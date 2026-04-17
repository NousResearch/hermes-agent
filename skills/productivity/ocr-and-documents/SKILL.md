---
name: ocr-and-documents
description: Unified document and vision OCR processing center. Extract text from PDFs, scanned documents, and images. Use web_extract for HTML/simple PDFs, paddleocr for high-precision OCR and complex layouts (tables/formulas), pymupdf for fast local pure-text PDFs, and marker-pdf for offline high-privacy parsing.
version: 3.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ocr, pdf, tables, image-to-text, document-parsing, text-recognition]
    related_skills: [powerpoint]
required_environment_variables:
  - name: PADDLEOCR_DOC_PARSING_API_URL
    description: Full endpoint URL for PaddleOCR layout parsing API
  - name: PADDLEOCR_OCR_API_URL
    description: Full endpoint URL for PaddleOCR text recognition API
  - name: PADDLEOCR_ACCESS_TOKEN
    description: Token for PaddleOCR API authentication
---

# Unified PDF & Document Extraction Center

This skill covers **pure-text PDFs, scanned documents, image OCR, and complex layouts (tables/formulas)**.

**Exclusions:**
- For DOCX: use `python-docx` (parses actual document structure, far better than OCR).
- For PPTX: see the `powerpoint` skill (uses `python-pptx` with full slide/notes support).

---

## Step 1: Core Decision Matrix

To choose the right tool, match the user's input type and goal against this matrix:

| Tool Name | Input Type | Core Capability & Responsibility | Best For | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **`web_extract`**<br>*(System Tool)* | URL (Webpage or PDF) | **Web scraping & long-text summarization**.<br>DOM-based parsing with built-in LLM summarizer (>5000 chars). | 1. Normal HTML webpages.<br>2. Arxiv or simple pure-text PDF URLs.<br>3. Quick extraction of core web content. | Cannot process images; cannot process scanned PDFs; loses complex table/formula layouts. |
| **`pymupdf`** | Local File (PDF) | **Lightweight local PDF text extraction**.<br>Extracts underlying text objects instantly. Zero dependencies. | 1. Local pure-text PDFs.<br>2. Splitting, merging, or searching PDFs.<br>3. Extracting embedded images from PDFs. | Fails on scanned documents (no text layer); cannot OCR images; tables often misaligned. |
| **`paddleocr_text`**<br>*(extract_paddleocr_text.py)* | URL or Local File (Image/PDF) | **High-precision pure-text OCR**.<br>Based on PP-OCRv5. Exceptionally high accuracy for mixed English/Chinese, small fonts, and handwriting. | 1. Screenshots, photos, scanned documents.<br>2. When only exact pure text is needed, ignoring paragraph layout.<br>3. Image text extraction in various languages. | Cannot reconstruct tables; cannot recognize math formulas; does not preserve reading order. |
| **`paddleocr_layout`**<br>*(extract_paddleocr_layout.py)* | URL or Local File (Image/PDF) | **Complex document layout analysis (VLM)**.<br>Based on PP-StructureV3/VL. Understands physical layout. **High-precision parsing**. | 1. Documents with complex tables (financial reports, invoices).<br>2. Academic papers with math formulas (LaTeX).<br>3. Multi-column newspapers/magazines.<br>4. Scenarios demanding high parsing accuracy. | Slower (large PDFs take minutes); consumes API quota; sends data to cloud API. |
| **`marker-pdf`** | Local File (Image/PDF) | **High-privacy offline document parsing**.<br>Local deep learning model. Data never leaves the machine. **Supports complex tables, formulas, and layouts.** | 1. **Highly sensitive/confidential documents** (must be processed offline).<br>2. User has a powerful GPU and doesn't mind downloading models.<br>3. Fallback when PaddleOCR API is unconfigured. | Requires 3-5GB model download; heavily consumes local CPU/GPU resources. |

> **Decision Path Examples:**
> - User provides an Arxiv link -> `web_extract` *(Note: This is a core Hermes Agent tool implemented in `tools/web_tools.py`, not a script in this skill)*
> - User provides a PDF with confidential company financial data -> `marker-pdf` (Privacy protection)
> - User provides a normal financial report screenshot (with tables), demands high accuracy -> `paddleocr_layout`
> - User provides a WeChat chat screenshot -> `paddleocr_text`
> - User provides a local 500-page novel PDF -> `pymupdf`

---

## Step 2: Tool References

### Tool 1: PaddleOCR API

**Dependencies:**
```bash
pip install httpx
```

**Configuration:**
Requires `PADDLEOCR_DOC_PARSING_API_URL`, `PADDLEOCR_OCR_API_URL`, and `PADDLEOCR_ACCESS_TOKEN`.
You can test your local setup by running:
```bash
# Test Layout Parsing
python scripts/extract_paddleocr_layout.py --file-url "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png" --pretty

# Test Text Recognition
python scripts/extract_paddleocr_text.py --file-url "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png" --pretty
```

**Layout Parsing (Tables, Formulas, Reading Order):**
```bash
# From URL
python scripts/extract_paddleocr_layout.py --file-url "https://example.com/report.pdf" --pretty

# From local file
python scripts/extract_paddleocr_layout.py --file-path "./invoice.png" --pretty
```
*Output is saved to a temp JSON file. Read the JSON and extract the `text` field for full content, or navigate `result.result.layoutParsingResults[n]` for structured page data.*

**Text Recognition (Pure Text OCR):**
```bash
# From URL
python scripts/extract_paddleocr_text.py --file-url "https://example.com/screenshot.jpg" --pretty

# From local file
python scripts/extract_paddleocr_text.py --file-path "./scan.pdf" --pretty
```
*Output is saved to a temp JSON file. Read the JSON and extract the `text` field.*

### Tool 2: PyMuPDF

```bash
pip install pymupdf pymupdf4llm
```

**Via helper script:**
```bash
python scripts/extract_pymupdf.py document.pdf              # Plain text
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # Tables
python scripts/extract_pymupdf.py document.pdf --images out/ # Extract images
python scripts/extract_pymupdf.py document.pdf --metadata    # Title, author, pages
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # Specific pages
```

### Tool 3: Marker-PDF

**Warning:** Requires ~5GB disk space for PyTorch and models.

```bash
# Check disk space first
python scripts/extract_marker.py --check

pip install marker-pdf
```

**Via helper script:**
```bash
python scripts/extract_marker.py document.pdf                # Markdown
python scripts/extract_marker.py document.pdf --json         # JSON with metadata
python scripts/extract_marker.py document.pdf --output_dir out/  # Save images
python scripts/extract_marker.py scanned.pdf                 # Scanned PDF (OCR)
```

### Tool 4: Web Extract

For standard webpages and online pure-text PDFs:
```python
web_extract(urls=["https://example.com/page", "https://arxiv.org/pdf/2402.03300"])
```

---

## Utilities

PyMuPDF handles PDF manipulation natively — use `execute_code` or inline Python:

```python
# Split: extract pages 1-5 to a new PDF
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# Merge multiple PDFs
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# Search for text across all pages
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"Page {i+1}: {len(results)} match(es)")
        print(page.get_text("text"))
```
