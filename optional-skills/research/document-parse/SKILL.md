---
name: document-parse
description: This skill should be used when the user asks to "parse a PDF", "extract text from a document", "read a DOCX", "read a PPTX", "extract an XLSX", "OCR an image", "get page screenshots from a PDF", or needs help deciding when to use document_parse instead of read_file or web_extract.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Documents, PDF, OCR, LiteParse, Ingestion, Parsing]
---

# Document Parse

Use `document_parse` as the primary tool for local documents that are not well-served by `read_file`.

## When to Use

- Use `read_file` for plain text source files when line-oriented reading matters.
- Use `web_extract` for remote URLs and website content.
- Use `document_parse` for local PDFs, Office documents, images, and mixed-layout files.
- Use `document_parse` when OCR, page filtering, bounding boxes, or screenshots are needed.

## Recommended Workflow

1. Start with `document_parse(path=..., backend="auto")`.
2. Add `target_pages` when only a subset of pages matter.
3. Set `include_pages=true` when page-local output is needed.
4. Add `include_text_items=true` or `include_bounding_boxes=true` only together with `include_pages=true`, and only when layout-aware output is needed.
5. Enable `generate_screenshots=true` when page images are needed for downstream vision workflows.

## Important Options

- `backend`: use `auto` by default; force `liteparse` only when local LiteParse behavior is specifically required.
- `ocr_enabled`: disable for clean text PDFs to reduce work; enable for scans and image-heavy documents.
- `ocr_language`: set the correct language code for non-English documents.
- `target_pages`: use values like `1-5,10` to reduce noise and runtime.
- `dpi`: raise for difficult OCR cases; keep default for normal text documents.
- `precise_bounding_box`: enable when exact spatial structure matters.
- `preserve_small_text`: enable for slides, spreadsheets, or dense PDFs with tiny labels.

## Fallback Guidance

- If LiteParse is unavailable, `document_parse` still handles text-like local files with the basic parser.
- If a non-text document fails with the basic parser, install LiteParse and retry.
- If screenshot generation is requested without LiteParse, expect an error and switch to a LiteParse-capable environment.

## Troubleshooting

- For scanned PDFs or photographed pages: keep OCR enabled and raise `dpi`.
- For huge documents: limit with `target_pages` before asking for summaries.
- For layout-sensitive extraction: request pages plus bounding boxes instead of only top-level text.
- For downstream ingestion: prefer `include_pages=true` so later chunking can stay page-aware.

## Reference

LiteParse library usage docs:
- `https://developers.llamaindex.ai/liteparse/guides/library-usage/`
