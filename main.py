import os
import streamlit as st
from transformers import pipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up Streamlit app
st.title("MyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
query = st.text_input("Ask your question:")
main_placeholder = st.empty()

# Helper function: Split text into chunks
def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks with a slight overlap for better context continuity."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    docs = text_splitter.split_text(text)
    return docs

# Load Hugging Face question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# If process URLs is clicked
if process_url_clicked:
    # Step 1: Load content from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data from URLs... ✅")
    data = loader.load()
    combined_text = " ".join([doc.page_content for doc in data])

    # Step 2: Split into manageable chunks
    main_placeholder.text("Splitting text into chunks... ✅")
    chunks = split_into_chunks(combined_text, chunk_size=500, overlap=50)

    st.sidebar.write(f"Generated {len(chunks)} chunks for processing.")

    # Step 3: Answer the query using each chunk
    if query:
        main_placeholder.text("Processing query on chunks...")
        results = []
        for chunk in chunks:
            try:
                response = qa_pipeline({"question": query, "context": chunk})
                results.append(response["answer"])
            except Exception as e:
                st.error(f"Error processing chunk: {e}")
        
        # Display the answers
        st.header("Answers from Chunks")
        for idx, answer in enumerate(results):
            st.subheader(f"Chunk {idx+1}")
            st.write(answer)

        # Aggregate the results for a final response (optional)
        final_answer = " ".join(results)
        st.header("Final Aggregated Answer")
        st.write(final_answer)
