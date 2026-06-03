---
name: ocr-and-documents
description: "Extract text from PDFs/scans (pymupdf, marker-pdf)."
version: 2.4.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [PDF, Documents, Research, Arxiv, Text-Extraction, OCR]
    related_skills: [powerpoint]
---

# PDF & Document Extraction

For DOCX: use `python-docx` (parses actual document structure, far better than OCR).
For PPTX: see the `powerpoint` skill (uses `python-pptx` with full slide/notes support).
This skill covers **PDFs and scanned documents**.

## Step 1: Remote URL Available?

If the document has a URL, **always try `web_extract` first**:

```
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
web_extract(urls=["https://example.com/report.pdf"])
```

This handles PDF-to-markdown conversion via Firecrawl with no local dependencies.

Only use local extraction when: the file is local, web_extract fails, or you need batch processing.

## Step 2: Choose Local Extractor

| Feature | pymupdf (~25MB) | tesseract (system binary) | marker-pdf (~3-5GB) |
|---------|-----------------|---------------------------|---------------------|
| **Text-based PDF** | ✅ | ✅ (prefers text layer) | ✅ |
| **Scanned PDF (OCR)** | ❌ | ✅ (local/offline) | ✅ (90+ languages) |
| **Tables** | ✅ (basic) | ❌ | ✅ (high accuracy) |
| **Equations / LaTeX** | ❌ | ❌ | ✅ |
| **Code blocks** | ❌ | ❌ | ✅ |
| **Forms** | ❌ | ❌ | ✅ |
| **Headers/footers removal** | ❌ | ❌ | ✅ |
| **Reading order detection** | ❌ | ❌ | ✅ |
| **Images extraction** | ✅ (embedded) | ❌ | ✅ (with context) |
| **Images → text (OCR)** | ❌ | ✅ | ✅ |
| **EPUB** | ✅ | ❌ | ✅ |
| **Markdown output** | ✅ (via pymupdf4llm) | ❌ (plain text) | ✅ (native, higher quality) |
| **Install size** | ~25MB | ~0 (system pkg, no models) | ~3-5GB (PyTorch + models) |
| **Network** | none | none (fully offline) | downloads ~2.5GB models |
| **Speed** | Instant | ~0.1s/page (CPU) | ~1-14s/page (CPU), ~0.2s/page (GPU) |

**Decision tree**:
1. Remote/public URL → try `web_extract` first.
2. Text-based local PDF → `pdftotext -layout` / **pymupdf**; do **not** OCR selectable text.
3. One-off scanned image/PDF where plain text is enough → `extract_tesseract.py` quick helper.
4. Spearhead-scale or business batch OCR → `production_ocr_pipeline.py` canonical local pipeline: text-PDF short-circuit, 300 DPI render, `ces+eng`, PSM retry matrix (`3,6,11`), TSV-derived confidence metrics, layout outputs, manifests, and `needs_review` gates.
5. Need equations, forms, tables, complex layout, or markdown structure → **marker-pdf** or another layout-aware extractor.

If the user needs marker capabilities but the system lacks ~5GB free disk:
> "This document needs OCR/advanced extraction (marker-pdf), which requires ~5GB for PyTorch and models. Your system has [X]GB free. Options: free up space, provide a URL so I can use web_extract, or I can try pymupdf which works for text-based PDFs but not scanned documents or equations."

---

## pymupdf (lightweight)

```bash
pip install pymupdf pymupdf4llm
```

**Via helper script**:
```bash
python scripts/extract_pymupdf.py document.pdf              # Plain text
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # Tables
python scripts/extract_pymupdf.py document.pdf --images out/ # Extract images
python scripts/extract_pymupdf.py document.pdf --metadata    # Title, author, pages
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # Specific pages
```

**Inline**:
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## tesseract (lightweight local/offline OCR)

For **scanned PDFs and images** when you only need plain text, want to stay fully
offline, and don't have the GBs that marker-pdf needs. It calls the system
`tesseract` binary — there are **no Python model downloads and no network calls**.

**Optional dependency — NOT auto-installed.** Install it yourself (no sudo is run by
the script):
```bash
# Debian/Ubuntu: tesseract-ocr + per-language data + poppler-utils (for scanned PDFs)
apt-get install tesseract-ocr tesseract-ocr-ces poppler-utils   # ces = Czech, etc.
# macOS
brew install tesseract poppler
```
Scanned-PDF support also needs a rasterizer: `pdftoppm` (poppler-utils) **or** the
`pymupdf` package if it is already importable — the script auto-detects and prefers
`pdftoppm`.

**Check what's available first** (never errors, even with nothing installed):
```bash
python scripts/extract_tesseract.py --check        # JSON: which deps/langs are ready
python scripts/extract_tesseract.py --list-langs   # installed OCR languages
```

**Via helper script**:
```bash
python scripts/extract_tesseract.py image.png                 # OCR an image → stdout
python scripts/extract_tesseract.py image.png --lang ces+eng  # explicit mixed CZ/EN
python scripts/extract_tesseract.py scan.pdf                  # text layer first, OCR only if none
python scripts/extract_tesseract.py scan.pdf --force-ocr      # always OCR every page
python scripts/extract_tesseract.py doc.pdf --json            # structured result / errors
```

For directories, confidence/review gates, layout artifacts, or anything that may
become durable business evidence, use the production pipeline instead:

```bash
python scripts/production_ocr_pipeline.py \
  /path/to/docs-or-one-file \
  --output-dir ~/spearhead-execution/ocr-batch-YYYYMMDD \
  --lang ces+eng \
  --dpi 300 \
  --timeout 120

python scripts/production_ocr_pipeline.py --check  # dependency JSON; no input needed
```

Pipeline behavior: text PDFs short-circuit through `pdftotext -layout`; scans/images
render through Poppler + Tesseract; subprocesses run without shell and with timeouts;
Tesseract OpenMP is capped; PSM defaults are `3,6,11`; text/TSV/hOCR/ALTO/PAGE XML
are persisted per page; per-document and batch manifests record confidence metrics
and `needs_review` flags. See `references/production-ocr-pipeline.md`.

**Quick-helper routing**: `extract_tesseract.py` returns the embedded PDF text layer
when present (detected via pymupdf) and only OCRs when there is no text layer or
you pass `--force-ocr`. So one-off text PDFs stay fast and OCR is opt-in/fallback,
never the default.

**Quick-helper security limits** (defensive defaults; override with env vars):

| Env var | Default | Purpose |
|---------|---------|---------|
| `HERMES_OCR_TIMEOUT_S` | 120 | timeout per `tesseract`/`pdftoppm` call |
| `HERMES_OCR_MAX_FILE_BYTES` | 52428800 (50MB) | reject oversized inputs |
| `HERMES_OCR_MAX_PAGES` | 50 | cap pages rasterized/OCR'd per PDF |
| `HERMES_OCR_DPI` | 300 | rasterization resolution |

`production_ocr_pipeline.py` intentionally keeps a smaller explicit CLI surface:
`--check`, `--output-dir`, `--lang`, repeatable `--psm`, `--dpi`, `--timeout`, and
`--max-pages`. It does not expose `--force-ocr` or the quick-helper env override
surface; text PDFs always short-circuit through `pdftotext -layout` when enough
selectable text is present.

All external commands in both Tesseract helpers run with `shell=False` and explicit
argument lists. `extract_tesseract.py` returns structured errors with stable exit
codes under `--json`; the production pipeline records per-document errors in
manifests and exits non-zero when any document fails.

**Not for**: tables, equations, forms, complex layout, or markdown structure — use
marker-pdf for those.

---

## marker-pdf (high-quality OCR)

```bash
# Check disk space first
python scripts/extract_marker.py --check

pip install marker-pdf
```

**Via helper script**:
```bash
python scripts/extract_marker.py document.pdf                # Markdown
python scripts/extract_marker.py document.pdf --json         # JSON with metadata
python scripts/extract_marker.py document.pdf --output_dir out/  # Save images
python scripts/extract_marker.py scanned.pdf                 # Scanned PDF (OCR)
python scripts/extract_marker.py document.pdf --use_llm      # LLM-boosted accuracy
```

**CLI** (installed with marker-pdf):
```bash
marker_single document.pdf --output_dir ./output
marker /path/to/folder --workers 4    # Batch
```

---

## Arxiv Papers

```
# Abstract only (fast)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# Full paper
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# Search
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## Split, Merge & Search

pymupdf handles these natively — use `execute_code` or inline Python:

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

No extra dependencies needed — pymupdf covers split, merge, search, and text extraction in one package.

---

## Notes

- `web_extract` is always first choice for URLs
- pymupdf is the safe default — instant, no models, works everywhere
- tesseract is the lightweight local/offline OCR option — system binary, no models, no network; use when you need OCR but not marker's layout/equation features
- marker-pdf is for OCR, scanned docs, equations, complex layouts — install only when needed
- Both helper scripts accept `--help` for full usage
- marker-pdf downloads ~2.5GB of models to `~/.cache/huggingface/` on first use
- For Word docs: `pip install python-docx` (better than OCR — parses actual structure)
- For PowerPoint: see the `powerpoint` skill (uses python-pptx)
