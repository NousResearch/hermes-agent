# ArXiv Research Skill

This skill allows the Hermes agent to fetch, read, and analyze academic papers directly from ArXiv.org using the paper's URL.

## Description
The `arxiv_research` tool automates the process of:
1. Downloading the PDF of an academic paper.
2. Extracting text content from the PDF (using PyMuPDF).
3. Formatting the text for LLM analysis.

## Usage
The agent can use this tool when a user provides an ArXiv link and asks for a summary, specific methodology details, or a general explanation of the paper.

### Example Prompt
- "Can you read this arxiv paper for me and give me a comprehensive summary? https://arxiv.org/abs/1706.03762"
- "Explain the methodology used in this paper: https://arxiv.org/abs/2303.08774"

## Requirements
- `pymupdf`: For high-fidelity PDF text extraction.