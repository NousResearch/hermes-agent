---
name: pdf
description: Extract text from PDFs, read metadata, search content, split and merge PDF files. Requires pypdf (pip install pypdf).
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [pdf, extract, text, metadata, search, split, merge, documents, pypdf]
    category: productivity
    requires_toolsets: [terminal]
---

# PDF Skill

Read, search, split, and merge PDF files.
6 commands: extract, metadata, info, search, split, merge.

Requires one dependency: `pip install pypdf`

---

## When to Use
- User wants to extract text from a PDF
- User wants to read PDF metadata (author, title, creation date...)
- User wants to search for a word or pattern inside a PDF
- User wants to split a PDF into specific pages
- User wants to merge multiple PDFs into one
- User wants to know page count, dimensions, or if a PDF has images

---

## Prerequisites
```bash
pip install pypdf
```

Script path: `~/.hermes/skills/productivity/pdf/scripts/pdf_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/productivity/pdf/scripts/pdf_client.py

python3 $SCRIPT extract document.pdf
python3 $SCRIPT extract document.pdf --pages 1-5 --output txt
python3 $SCRIPT extract document.pdf --pages 1,3,7 --output json
python3 $SCRIPT metadata document.pdf
python3 $SCRIPT info document.pdf
python3 $SCRIPT search document.pdf "revenue"
python3 $SCRIPT split document.pdf --pages 1-10 --output-dir ./output
python3 $SCRIPT merge file1.pdf file2.pdf file3.pdf --output combined.pdf
```

---

## Commands

### extract FILE [--pages PAGES] [--output txt|json]
Extract text from PDF. Default: all pages, plain text output.

Pages format: `1-5` (range), `1,3,7` (list), `all` (default)
Output: `txt` (plain text to stdout) or `json` (per-page JSON)
```bash
python3 $SCRIPT extract report.pdf
python3 $SCRIPT extract report.pdf --pages 1-3 --output json
```

### metadata FILE
Read PDF metadata: title, author, subject, creator, producer, dates, page count.
```bash
python3 $SCRIPT metadata report.pdf
```

### info FILE
Full document analysis: metadata + page dimensions (mm) + word count + image detection.
```bash
python3 $SCRIPT info report.pdf
```

### search FILE QUERY
Search for text pattern (case-insensitive, supports regex).
```bash
python3 $SCRIPT search report.pdf "quarterly revenue"
python3 $SCRIPT search report.pdf "\$[0-9]+"
```
Returns: {page, line_number, context} for each match.

### split FILE [--pages PAGES] [--output-dir DIR]
Extract specific pages to a new PDF.
```bash
python3 $SCRIPT split report.pdf --pages 1-5
python3 $SCRIPT split report.pdf --pages 1,3,7 --output-dir ./extracted
```
Output filename: `report_pages_1-5.pdf`

### merge FILE1 FILE2 [...] [--output OUTPUT]
Merge multiple PDFs into one file.
```bash
python3 $SCRIPT merge part1.pdf part2.pdf part3.pdf --output complete.pdf
```

---

## Pitfalls
- Scanned PDFs (image-only) produce no extractable text. Use OCR tools (tesseract) instead.
- Encrypted PDFs: tool tries empty password first. Password-protected files require manual decryption.
- Page numbers are 1-indexed in CLI arguments, 0-indexed internally.
- Very large PDFs may be slow on text extraction.

---

## Verification
```bash
pip install pypdf
python3 ~/.hermes/skills/productivity/pdf/scripts/pdf_client.py --help
```
