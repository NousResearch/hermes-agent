---
name: document-parse
description: Use existing Hermes tools to extract text from local PDFs, Office documents, images, and other non-plain-text files. Prefer `web_extract` for URLs, `read_file` for plain text, and `execute_code` with LiteParse for local document parsing, OCR, page screenshots, and layout-aware extraction.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Documents, PDF, OCR, LiteParse, Ingestion, Parsing]
---

# LiteParse Document Extraction

This skill packages the document parsing guidance without adding a registered tool.
Use it to decide when to reach for existing Hermes tools and when to call LiteParse
through `execute_code`.

## When to Use

- Use `read_file` for plain text files where line-oriented reading matters.
- Use `web_extract` first for remote URLs, including public PDF links.
- Use `vision_analyze` when the goal is understanding a single image rather than extracting structured document text.
- Use `execute_code` with LiteParse for local PDFs, DOCX, PPTX, XLSX, and image-based documents when OCR, page filtering, screenshots, or layout-aware output is needed.

## Recommended Workflow

1. Decide whether the document is remote or local.
2. For remote documents, try `web_extract` before any local parsing.
3. For local plain text files, use `read_file`.
4. For local rich documents, use `execute_code` and call LiteParse directly.
5. Return only the fields needed for the task so tool output stays small.

## LiteParse Setup

Install LiteParse only when needed:

```bash
pip install liteparse
```

If LiteParse is unavailable, fall back to:

- `web_extract` for URLs
- `read_file` for text-like local files
- `vision_analyze` for image interpretation

## Execute Code Patterns

### Basic extraction

Use this when the user wants the text from a local PDF, DOCX, PPTX, XLSX, or image-based document:

```python
from liteparse import LiteParse

parser = LiteParse()
result = parser.parse("/absolute/path/to/document.pdf")

print(result.text)
```

### Page-limited extraction

Use page filtering for large files or when only a subset matters:

```python
from liteparse import LiteParse

parser = LiteParse()
result = parser.parse(
    "/absolute/path/to/document.pdf",
    target_pages="1-5,10",
)

print(result.text)
```

### OCR-heavy documents

Use OCR for scans, photographed pages, and image-heavy PDFs:

```python
from liteparse import LiteParse

parser = LiteParse()
result = parser.parse(
    "/absolute/path/to/document.pdf",
    ocr_enabled=True,
    ocr_language="en",
    dpi=200,
)

print(result.text)
```

### Layout-aware extraction

When page structure matters, inspect pages and text items instead of only `result.text`:

```python
from liteparse import LiteParse

parser = LiteParse()
result = parser.parse("/absolute/path/to/document.pdf")

for page in result.pages:
    print({"page": page.pageNum, "text": page.text})
    for item in getattr(page, "textItems", [])[:10]:
        print({
            "text": getattr(item, "str", ""),
            "x": getattr(item, "x", None),
            "y": getattr(item, "y", None),
        })
```

### Page screenshots for downstream vision

When the user needs page images for review or `vision_analyze`, generate screenshots first:

```python
from pathlib import Path
from liteparse import LiteParse

out_dir = Path("/tmp/liteparse-pages")
out_dir.mkdir(parents=True, exist_ok=True)

parser = LiteParse()
shots = parser.screenshot(
    "/absolute/path/to/document.pdf",
    output_dir=str(out_dir),
    image_format="png",
    target_pages="1-3",
)

print([{"page": shot.page_num, "image_path": shot.image_path} for shot in shots])
```

## Fallback Guidance

- If LiteParse is unavailable, do not invent a new document parser tool.
- If the file is a URL, retry with `web_extract`.
- If the file is plain text or code, use `read_file`.
- If the input is an image and the user wants interpretation rather than raw extraction, use `vision_analyze`.
- If the task specifically requires LiteParse features, explain that LiteParse must be installed before continuing.

## Troubleshooting

- For scanned PDFs or photographed pages, keep OCR enabled and raise `dpi`.
- For huge documents, narrow the page range before summarizing.
- For slides and spreadsheets, preserve page-aware structure instead of flattening everything into one long string.
- For downstream vision workflows, generate page screenshots first and inspect those images separately.

## Reference

LiteParse library usage docs:
- `https://developers.llamaindex.ai/liteparse/guides/library-usage/`
