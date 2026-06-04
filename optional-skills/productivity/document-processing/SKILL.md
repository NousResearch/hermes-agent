---
name: document-processing
description: "Document processing: PDF editing (nano-pdf), OCR/text extraction (pymupdf, marker-pdf), PowerPoint manipulation."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [PDF, Documents, OCR, PowerPoint, Text-Extraction, Editing, Productivity]
    umbrella: true
---

# Document Processing

Umbrella skill covering PDF manipulation, text extraction from documents/scans, and PowerPoint creation/editing.

## Skills

| Skill | Purpose | Best For |
|-------|---------|----------|
| [nano-pdf](references/nano-pdf.md) | Edit PDF text via NL prompts | Fixing typos, changing titles, updating text |
| [ocr-and-documents](references/ocr-and-documents.md) | Extract text from PDFs/scans | Text-based PDFs, scanned documents, OCR, equations |
| [powerpoint](references/powerpoint.md) | Create/read/edit .pptx files | Slide decks, presentations, templates |

## Overview

This cluster handles document-level operations:

- **Editing PDFs**: Use `nano-pdf` for text/typo corrections via natural language instructions
- **Extracting from PDFs/scans**: Use `ocr-and-documents` for text extraction, OCR, and layout analysis
- **Working with PowerPoint**: Use `powerpoint` for creating, reading, editing slide decks

### Decision Guide

| Need | Solution |
|------|----------|
| Fix a typo in a PDF | `nano-pdf edit <file.pdf> <page> "<instruction>"` |
| Extract text from a text-based PDF | `pymupdf` (via `ocr-and-documents`) |
| OCR a scanned document | `marker-pdf` (via `ocr-and-documents`) |
| Extract text from a URL PDF | `web_extract()` first |
| Create a presentation | `powerpoint` skill |
| Edit an existing .pptx | `powerpoint` skill |

## References

- [nano-pdf](references/nano-pdf.md) — Edit PDF text via CLI
- [ocr-and-documents](references/ocr-and-documents.md) — Extract text from PDFs/scans
- [powerpoint](references/powerpoint.md) — Create and edit PowerPoint decks