---
name: pdf-processing
description: Process PDF documents. Extract text, metadata, merge, split, and search within PDFs using CLI tools like poppler-utils (pdftotext, pdfinfo) and qpdf.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [PDF, Document, Extraction, Text, poppler-utils, qpdf]
    related_skills: []
---

# PDF Processing Skill

This skill teaches the agent to process PDF files using **standard CLI tools only** (`pdftotext`, `pdfinfo` from `poppler-utils`).

## ⛔ CRITICAL RULES — Read Before Anything Else

1. **NEVER** read a `.pdf` file directly with `cat`, `head`, or `less`. Binary PDF data will corrupt your terminal context.
2. **NEVER** use Python as a fallback (`pdfminer`, `pymupdf`, `pypdf2`, etc.). This skill exists precisely to avoid that. If `pdftotext` is not installed, **STOP** and ask the user to install it — do NOT write Python workarounds.
3. **NEVER** install packages without `sudo` (it will hang). Always check first, then ask the user for permission before installing.

---

## Step 1: Check Prerequisites (ALWAYS do this first)

```bash
command -v pdftotext && echo "OK" || echo "NOT INSTALLED"
command -v pdfinfo  && echo "OK" || echo "NOT INSTALLED"
```

**If tools are NOT installed:**
- Tell the user: *"pdftotext is not installed. Please run: `sudo apt-get install -y poppler-utils` (Linux) or `brew install poppler` (macOS), then try again."*
- **Do not proceed.** Do not attempt any Python alternative.

---

## Step 2: Read PDF Metadata

Always do this before reading content. It tells you the page count so you can read incrementally.

```bash
pdfinfo document.pdf
```

---

## Step 3: Extract Text

```bash
# Extract all text (good for small PDFs, < 10 pages)
pdftotext -layout document.pdf - 

# Extract specific pages only (REQUIRED for large PDFs)
pdftotext -f 1 -l 5 document.pdf -
```

> **Rule:** If the PDF has more than 10 pages, ALWAYS extract page ranges incrementally. Never extract all at once.

---

## Step 4: Search Within Extracted Text

```bash
# Extract to a temp file and grep for a keyword
pdftotext -layout document.pdf /tmp/pdf_output.txt
grep -n -C 3 "keyword" /tmp/pdf_output.txt
rm /tmp/pdf_output.txt   # Always cleanup!
```

---

## Step 5: Merge or Split PDFs (requires qpdf)

```bash
# Merge PDFs
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf

# Split into individual pages
qpdf --split-pages document.pdf page_%d.pdf
```

---

## Verification

After extraction, verify the output is readable plain text (not garbled binary). If the output looks like garbage, the PDF may be scanned (image-based). In that case, tell the user: *"This PDF appears to be image-based and requires OCR. Consider using the `ocr-and-documents` skill instead."*
