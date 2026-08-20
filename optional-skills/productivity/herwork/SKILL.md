---
name: herwork
description: HerWork mode — docs, slides, sheets, PDFs, browser, files.
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [herwork, workspace, office, documents, browser, productivity]
    category: productivity
    related_skills: [docx, powerpoint, xlsx, pdf, ocr-and-documents, obsidian]
---

# HerWork Mode

A full work environment for Hermes: you act as a colleague with a shared
desk. The user hands you work — write a report, build a deck, fill a
spreadsheet, research on the web, organize a folder — and you deliver
finished files plus a short summary of what you did.

This skill is the orchestration layer. It does not replace the document
skills (`docx`, `powerpoint`, `xlsx`, `pdf`); it tells you which one to
load for each task and where files live.

## When to Use

- The user asks for a complete piece of work, not a code change: a Word
  report, a slide deck, a filled spreadsheet, a formatted PDF.
- The task mixes formats: research on the web, then write it up; read a
  folder of files, then summarize into a document.
- The user drops files for you to process (convert, merge, reformat,
  extract data from).
- The user says "herwork", "herwork mode", or invokes `/herwork`.
- Not for: editing this repository's own source code, or one-off answers
  that need no files.

## The shared desk (workspace layout)

All HerWork work happens under `~/herwork/`:

```
~/herwork/
  inbox/    user drops source files here; treat as read-only
  work/     your scratch space; intermediate drafts and extracted data
  output/   finished deliverables; one subfolder per task if several files
```

Create the three directories at the start of a herwork session if they are
missing. Check `inbox/` when the user's request references "the files I
gave you" without attaching paths.

## Workflow

1. **Intake.** Restate the deliverable in one sentence: what file(s),
   what format, who it's for. If the request names source material, read
   it first (inbox files, attached paths, or URLs via the browser tools).
2. **Plan briefly.** For anything with more than one artifact, list the
   artifacts you'll produce before producing them.
3. **Produce into `work/`, then promote.** Draft in `work/`; when a file
   is finished and verified, move or save the final copy into `output/`
   with a clear name (`q3-sales-report.docx`, not `draft2-final.docx`).
4. **Verify before delivering.** Re-open what you produced: read the
   `.docx` back, count the slides, recompute the sheet's totals, or open
   the PDF's first page. A file you haven't re-read is not done.
5. **Deliver.** End with the output paths and two or three sentences on
   what's inside. Offer the obvious next iteration (shorter, different
   tone, Arabic version, ...).

## Routing — which tool for which job

| Job | Use |
| --- | --- |
| Word documents (reports, letters, contracts) | `docx` skill |
| Slide decks | `powerpoint` skill |
| Spreadsheets, data tables, budgets | `xlsx` skill |
| Reading or producing PDFs | `pdf` skill |
| Scanned documents, images of text | `ocr-and-documents` skill |
| Web research, reading pages, filling web forms | native `browser_*` tools |
| GUI apps with no API (desktop clicks) | `computer-use` skill |
| Notes and knowledge bases | `obsidian` skill |
| Convert docx/pptx/xlsx → PDF | `soffice --headless --convert-to pdf <file>` (needs LibreOffice) |
| Convert markdown ↔ docx/html | `pandoc` (needs pandoc) |
| OCR scanned images (Arabic + English) | `tesseract <img> <out> -l ara+eng` (needs tesseract + lang packs) |
| Plain files: move, rename, organize, convert text | native file/terminal tools |

The three converters are optional enhancers: if one is missing and the
task needs it, follow the safety rules below (say what you'd install and
wait for approval).

Load the document skill for the format you're about to produce before
producing it — each one has scripts and conventions that prevent broken
files (e.g. the docx tracked-changes and templating CLIs).

## Arabic typography

Any deliverable that contains Arabic MUST use the **Cairo** font — the
office-suite default (Calibri) renders Arabic badly. Use the helper at
`scripts/arabic_style.py` in this skill's directory:

- docx: `style_docx(doc)` after building the Document
- pptx: `style_pptx(prs)` after building the Presentation (tables included)
- pdf (reportlab): `register_pdf_font()`, then draw with
  `canvas.setFont("Cairo", size)`; shape Arabic first with
  `arabic_reshaper.reshape(...)` + `bidi.algorithm.get_display(...)`

Setting `font.name` alone is NOT enough — Arabic is shaped from the
complex-script font slot (`w:cs` in docx, `a:cs` in pptx), which these
helpers also set. `register_pdf_font` locates the Cairo TTF across
Linux/macOS/Windows font directories and raises an install hint if Cairo
is missing (free at fonts.google.com/specimen/Cairo). Pure-English
deliverables may use the suite defaults.

## Safety rules (non-negotiable)

- **Write only inside `~/herwork/`** unless the user explicitly gives a
  target path. Never modify a user's original file in place — copy it to
  `work/` first and edit the copy.
- `inbox/` is read-only. Deliverables go to `output/`, never back into
  `inbox/`.
- No destructive operations outside the workspace. Inside it, prefer
  moving superseded drafts to `work/archive/` over deleting them.
- Browser work follows the normal website policy; never enter credentials
  or payment details into web forms.
- If a task needs software that isn't installed, say what you'd install
  and wait for approval before installing system-wide.
