---
name: pdf
description: "Create, merge, split, fill, and secure PDF files."
version: 1.0.0
author: Anthropic (adapted by Nous Research)
license: Proprietary. LICENSE.txt has complete terms
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [PDF, Documents, Forms, Office, Productivity]
    category: productivity
    related_skills: [ocr-and-documents, nano-pdf, docx, xlsx]
---

# PDF Skill

Create, combine, split, transform, and secure PDF files — merging, page manipulation, form filling, watermarks, encryption, and text/table extraction. For heavy text extraction from scanned documents prefer the `ocr-and-documents` skill; for natural-language edits to existing PDF text prefer `nano-pdf`.

## When to Use

Use this skill whenever the user wants to do anything with PDF files: reading or extracting text/tables, combining or merging multiple PDFs, splitting PDFs apart, rotating pages, adding watermarks, creating new PDFs, filling PDF forms, encrypting/decrypting, extracting images, or OCR on scanned PDFs. If the user mentions a .pdf file or asks to produce one, use this skill.

## Prerequisites

```bash
pip install pypdf pdfplumber reportlab
which pdftotext || sudo apt install -y poppler-utils   # pdftotext, pdftoppm, pdfimages
which qpdf || sudo apt install -y qpdf                 # CLI merge/split/decrypt
```

macOS: `brew install poppler qpdf`. OCR extras: `pip install pytesseract pdf2image` + `sudo apt install -y tesseract-ocr`.

> Script paths below are relative to this skill's directory. Form filling has its own workflow — read [forms.md](forms.md) and follow it. Advanced library usage (pypdfium2, pdf-lib) and troubleshooting: [reference.md](reference.md).

## Quick Reference

| Task | Best Tool | Command/Code |
|------|-----------|--------------|
| Merge PDFs | pypdf | `writer.add_page(page)` per page |
| Split PDFs | pypdf | One page per file |
| Extract text | pdfplumber | `page.extract_text()` |
| Extract tables | pdfplumber | `page.extract_tables()` |
| Create PDFs | reportlab | Canvas or Platypus |
| Command-line merge/split | qpdf | `qpdf --empty --pages ...` |
| OCR scanned PDFs | pytesseract | Convert to images first (or use `ocr-and-documents`) |
| Fill PDF forms | see [forms.md](forms.md) | `scripts/fill_fillable_fields.py` etc. |
| Edit existing text | `nano-pdf` skill | `nano-pdf edit file.pdf <page> "<instruction>"` |

## Common operations

### Merge / split / rotate (pypdf)

```python
from pypdf import PdfReader, PdfWriter

# Merge
writer = PdfWriter()
for pdf_file in ["doc1.pdf", "doc2.pdf"]:
    for page in PdfReader(pdf_file).pages:
        writer.add_page(page)
with open("merged.pdf", "wb") as f:
    writer.write(f)

# Split: one file per page
reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    w = PdfWriter(); w.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as f:
        w.write(f)

# Rotate
page = reader.pages[0]
page.rotate(90)  # clockwise
```

### Extract text and tables (pdfplumber)

```python
import pdfplumber, pandas as pd

with pdfplumber.open("document.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    tables = [pd.DataFrame(t[1:], columns=t[0])
              for page in pdf.pages
              for t in page.extract_tables() if t]
```

### Create PDFs (reportlab)

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = [Paragraph("Report Title", styles["Title"]), Spacer(1, 12),
         Paragraph("Body text...", styles["Normal"]), PageBreak(),
         Paragraph("Page 2", styles["Heading1"])]
doc.build(story)
```

**Subscripts/superscripts:** never use Unicode sub/superscript characters (₀₁₂, ⁰¹²) — the built-in fonts lack the glyphs and render solid black boxes. Use `<sub>`/`<super>` markup inside `Paragraph` objects: `Paragraph("H<sub>2</sub>O", styles['Normal'])`. For canvas-drawn text, adjust font size and position manually.

### Command-line tools

```bash
pdftotext -layout input.pdf output.txt                     # text, layout preserved
pdftotext -f 1 -l 5 input.pdf output.txt                   # pages 1-5
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf     # merge
qpdf input.pdf --pages . 1-5 -- pages1-5.pdf               # split range
qpdf input.pdf output.pdf --rotate=+90:1                   # rotate page 1
qpdf --password=pw --decrypt encrypted.pdf decrypted.pdf   # remove password
pdfimages -j input.pdf img                                 # extract images
```

### Watermark

```python
from pypdf import PdfReader, PdfWriter

watermark = PdfReader("watermark.pdf").pages[0]
reader, writer = PdfReader("document.pdf"), PdfWriter()
for page in reader.pages:
    page.merge_page(watermark)
    writer.add_page(page)
with open("watermarked.pdf", "wb") as f:
    writer.write(f)
```

### Password protection

```python
writer.encrypt("userpassword", "ownerpassword")
```

### OCR scanned PDFs

```python
import pytesseract
from pdf2image import convert_from_path

pages = convert_from_path("scanned.pdf")
text = "\n\n".join(pytesseract.image_to_string(img) for img in pages)
```

For batch/structured extraction from scans, the `ocr-and-documents` skill (pymupdf, marker-pdf) is the better path.

## Form filling

Read [forms.md](forms.md) first — it distinguishes fillable (AcroForm) PDFs from flat scanned forms and walks through the helper scripts:

- `scripts/check_fillable_fields.py` — does the PDF have AcroForm fields?
- `scripts/extract_form_field_info.py` / `scripts/extract_form_structure.py` — enumerate fields
- `scripts/fill_fillable_fields.py` — fill AcroForm fields
- `scripts/fill_pdf_form_with_annotations.py` — overlay text on flat forms
- `scripts/check_bounding_boxes.py`, `scripts/create_validation_image.py` — verify placement visually

## Pitfalls

- `page.extract_text()` returns `None` on image-only pages — guard with `or ""` and fall back to OCR.
- pypdf preserves encryption flags: reading an encrypted PDF requires `PdfReader(path, password=...)` before pages are accessible.
- reportlab coordinates are bottom-left origin, points (1/72″) — not top-left.
- When filling flat forms by annotation overlay, always render a validation image and check the placement before delivering.

## Verification

1. Open the output with `PdfReader` and assert the expected page count.
2. Re-extract text from the output (`pdftotext` or pdfplumber) and confirm the content you added is present.
3. For anything visual (watermarks, filled forms, created reports): `pdftoppm -jpeg -r 100 output.pdf page` and inspect the images with `vision_analyze`.

## Related skills

`ocr-and-documents` (scanned-document text extraction), `nano-pdf` (NL text edits in place), `docx` (Word), `xlsx` (spreadsheets), `powerpoint` (decks).
