import os
import pandas as pd
import streamlit as st
import openai
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Configuration
st.set_page_config(page_title="AInsightBuddy", page_icon="📊", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.image("ainsightbuddy_logo.jpg")
    st.title("AInsightBuddy")
    st.write("Upload your CSV and let AI uncover insights.")
    
    # OpenAI API Key Input
    openai.api_key = st.text_input("Enter OpenAI API token:", type="password")
    try:
        # Test OpenAI token with gpt-4o-mini
        openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Test message to validate the token."}]
        )
        st.success("Token validated! You may now proceed.", icon="✅")
    except openai.error.AuthenticationError:
        st.warning("Invalid OpenAI API token. Please enter a valid token.", icon="⚠️")
    except Exception:
        st.warning("Token validation failed. Please check your connection or API usage.", icon="⚠️")

# Initialize session states
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

# File Upload Section
st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    try:
        # Load CSV into a DataFrame
        csv_data = pd.read_csv(uploaded_file)
        st.session_state.csv_data = csv_data
        st.session_state.uploaded_file = uploaded_file.name
        st.success(f"File '{uploaded_file.name}' uploaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Display DataFrame if loaded
if st.session_state.csv_data is not None:
    st.subheader(f"Preview of {st.session_state.uploaded_file}")
    st.dataframe(st.session_state.csv_data)

# Let the user analyze the entire dataset or specific columns
st.subheader("Select Data for Analysis")
analysis_scope = st.radio(
    "Do you want to analyze the entire dataset or specific columns?",
    ("Entire Dataset", "Specific Columns")
)

# Sampling to limit the dataset size for AI analysis
MAX_ROWS_FOR_ANALYSIS = 500  # Adjust based on memory and processing requirements

if analysis_scope == "Specific Columns":
    columns = st.multiselect("Select columns for analysis", st.session_state.csv_data.columns)
    if columns:
        selected_data = st.session_state.csv_data[columns]
    else:
        st.warning("No columns selected for analysis.")
        selected_data = None
else:
    selected_data = st.session_state.csv_data

# If the dataset is too large, sample it
if selected_data is not None:
    if len(selected_data) > MAX_ROWS_FOR_ANALYSIS:
        st.warning(f"The dataset is too large for analysis. Sampling {MAX_ROWS_FOR_ANALYSIS} rows.")
        selected_data = selected_data.sample(MAX_ROWS_FOR_ANALYSIS, random_state=42)

    # Generate a summarized dataset for AI analysis
    summarized_data = selected_data.describe(include='all').reset_index().to_string()

    st.subheader("AI-Powered Insights")
    prompt = (
        f"Analyze the following dataset summary for trends, patterns, outliers, "
        f"and actionable insights:\n{summarized_data}"
    )

    # Call OpenAI for analysis
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        insights = response["choices"][0]["message"]["content"]
        st.markdown(insights)
    except Exception as e:
        st.error(f"Failed to generate insights: {e}")

    # Direct Calculation of Insights
    if selected_data is not None and not selected_data.empty:
        st.subheader("Data-Driven Insights")

        csv_string = selected_data.to_csv(index=False)
        prompt = f"""
        You are AInsightBuddy, a data analyst specializing in uncovering trends and actionable insights.
        Analyze the following dataset and provide:
        - Trends
        - Outliers
        - Patterns
        - Recommended visualizations and their purposes
        
        Please focus on ensuring that your output is well-structured and easy to read. Avoid unnecessary breaks or word concatenation. Provide insights in a clear and professional manner.
    

        Dataset:
        {csv_string}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful data analysis assistant. Please focus on ensuring that your output is well-structured and easy to read. Avoid unnecessary breaks or word concatenation. Provide insights in a clear and professional manner."}, {"role": "user", "content": prompt}],
                temperature=0.5,
            )
            insights = response["choices"][0]["message"]["content"]
            st.markdown(insights)
        except Exception as e:
            st.error(f"Failed to generate insights: {e}")

        # Visualization Recommendations Section
        st.subheader("Suggested Visualizations")
        st.info(
            """
            Common visualization options based on data structure:
            - **Histogram**: For visualizing distributions of a single numeric column.
            - **Box Plot**: For detecting outliers in numeric data.
            - **Bar Chart**: To compare aggregated values (e.g., categories vs. totals).
            - **Scatter Plot**: To explore correlations between two numeric variables.
            - **Pie Chart**: For visualizing proportions of categories.
            - **Heatmap**: To study relationships between variables in a grid format.
            """
        )

        # Visualization Section
        st.subheader("Create Visualizations")
        visualization_type = st.selectbox(
            "Choose a visualization type",
            [
                "Histogram",
                "Box Plot",
                "Bar Chart",
                "Scatter Plot",
                "Pie Chart",
                "Heatmap",
                "Line Chart",
            ],
        )
        x_axis = st.selectbox("Select X-axis", selected_data.columns)
        y_axis = st.selectbox(
            "Select Y-axis (if applicable)",
            [None] + list(selected_data.columns),
            index=0,
        )
        category = st.selectbox(
            "Select Category for Grouping (optional)", [None] + list(selected_data.columns), index=0
        )

        try:
            if visualization_type == "Histogram":
                fig = px.histogram(selected_data, x=x_axis, color=category, title="Histogram")
            elif visualization_type == "Box Plot":
                fig = px.box(selected_data, x=x_axis, y=y_axis, color=category, title="Box Plot")
            elif visualization_type == "Bar Chart":
                fig = px.bar(selected_data, x=x_axis, y=y_axis, color=category, title="Bar Chart")
            elif visualization_type == "Scatter Plot":
                fig = px.scatter(selected_data, x=x_axis, y=y_axis, color=category, title="Scatter Plot")
            elif visualization_type == "Pie Chart":
                fig = px.pie(selected_data, names=x_axis, values=y_axis, title="Pie Chart")
            elif visualization_type == "Heatmap":
                fig = px.density_heatmap(selected_data, x=x_axis, y=y_axis, color_continuous_scale="Viridis", title="Heatmap")
            elif visualization_type == "Line Chart":
                fig = px.line(selected_data, x=x_axis, y=y_axis, color=category, title="Line Chart")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
    else:
        st.warning("No data available for analysis. Please check your selection.")
else:
    st.info("Upload a CSV file to get started!")
