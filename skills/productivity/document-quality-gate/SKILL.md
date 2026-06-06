---
name: document-quality-gate
description: Use when converting, translating, editing, OCRing, or delivering PDFs, EPUBs, transcripts, subtitles, or long documents where legibility and structure matter more than a crude format dump.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [pdf, epub, ocr, documents, translation, quality]
    related_skills: [ocr-and-documents, document-translation, nano-pdf]
---

# Document Quality Gate

## Overview

Prevents ugly document dumps. Jaime expects readable structure: clean paragraphs, lists, headings, sane tables, validated output, and real inspection before delivery.

## When to Use

- PDF to EPUB/Markdown/HTML.
- OCR of scans.
- Long document translation.
- PDF typo/title editing.
- Transcript/subtitle cleanup.
- User complains about legibility or broken lines.

## Workflow

1. Inspect source: pages, language, layout, scan/text layer, tables, lists, footnotes, images, blockers.
2. Choose toolchain: text PDFs use extraction + cleanup; scans use OCR first; EPUBs validate; PDF edits use purpose-built tools.
3. Normalize structure: fix hard line breaks, preserve headings/lists/quotes, convert tables deliberately, keep page references if useful.
4. Quality sample before full run: process representative pages, inspect output, adjust rules.
5. Validate final artifact: beginning/middle/end samples, format validator, file size/path.

## User Simulation Tests

- EPUB opened on phone → readable paragraphs, no one-word lines.
- PDF bullets → bullets remain bullets.
- Scan has OCR uncertainty → flag uncertainty, don't invent text.
- Telegram delivery → file under safe document cache.
- Summary requested → no raw extraction dump.

## Common Pitfalls

1. Crude PDF-to-EPUB conversion.
2. Skipping sample inspection.
3. Losing lists/tables.
4. Delivering from unsafe path.

## Verification Checklist

- [ ] Source inspected.
- [ ] Toolchain selected for layout type.
- [ ] Representative sample checked.
- [ ] Beginning/middle/end final samples checked.
- [ ] Validator run where applicable.
- [ ] Delivered file path safe.
