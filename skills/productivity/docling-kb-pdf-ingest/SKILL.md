---
name: docling-kb-pdf-ingest
description: Docling-first local PDF ingest into the Hermes KB evidence layer with PyMuPDF4LLM fast path, Telegram drop handling, OCR fallback, and optional wiki/source promotion.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [PDF, Docling, OCR, KB, Telegram, Evidence, Wiki]
    related_skills: [ocr-and-documents]
---

# Docling KB PDF Ingest

Use this when a PDF should be routed into the Hermes KB evidence layer instead of being merely read once.

It provides a Docling-first ingest path with:
- PyMuPDF4LLM fast path for small born-digital PDFs
- OCR fallback for image-like or scan-like PDFs when Docling output is too thin
- immutable raw-PDF storage under the KB
- evidence/source/chunk/transform records
- optional promotion into `wiki/sources/`
- Telegram PDF drop ingestion

Linked scripts in `scripts/` are the canonical repo copy of the pipeline.

## Layout

The scripts write under the active `HERMES_HOME`:
- `kb/raw/pdf/`
- `kb/evidence/`
- `kb/wiki/sources/`
- `kb/wiki/queries/`
- `kb/staging/telegram/`

## Core scripts

- `scripts/pdf_ingest_pipeline.py` — main ingest entry point
- `scripts/telegram_pdf_drop_ingest.py` — Telegram-specific wrapper
- `scripts/pdf_extract_docling.py` — Docling extraction with OCR fallback
- `scripts/pdf_extract_pymupdf4llm.py` — fast-path extractor
- `scripts/pdf_promote_to_wiki.py` — promotion candidate and source-page writer
- `scripts/pdf_pilot_report.py` — pilot/evaluation report writer

## Typical usage

Run from repo root with the virtualenv active:

```bash
source venv/bin/activate
python skills/productivity/docling-kb-pdf-ingest/scripts/pdf_ingest_pipeline.py /absolute/path/to/file.pdf --ingress-channel manual_pilot --promote-source-page
```

Telegram drop wrapper:

```bash
source venv/bin/activate
python skills/productivity/docling-kb-pdf-ingest/scripts/telegram_pdf_drop_ingest.py /tmp/cached.pdf --chat-id 1632061707 --message-id 42 --original-filename drop.pdf
```

Pilot report:

```bash
source venv/bin/activate
python skills/productivity/docling-kb-pdf-ingest/scripts/pdf_pilot_report.py
```

## Routing rules

- If a text layer is present, page count is within the fast-path limit, and OCR is not needed: use PyMuPDF4LLM.
- Otherwise default to Docling.
- If Docling markdown is too thin, render page images and run local OCR via `ocrmac`, appending the recovered text under `OCR Fallback Supplement` instead of overwriting Docling output.

## Promotion rule

Promotion candidates can always be written into the derived evidence folder.
Only promote into `wiki/sources/` when the extracted markdown is genuinely promotion-ready. Weak OCR or identity-card style scans should stay in evidence only.

## Verification

After changes, run the targeted tests:

```bash
source venv/bin/activate
python -m pytest tests/tools/test_pdf_ingest_pipeline.py tests/tools/test_pdf_extract_docling.py tests/gateway/test_telegram_pdf_ingest_flow.py tests/gateway/test_telegram_documents.py -q
```
