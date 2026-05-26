---
name: word-docx
triggers:
  - Word
  - DOCX
  - python-docx
  - tracked changes
  - document template
  - numbering
  - styles
  - OOXML
description: >
  Best-practice instructions for AI agents handling Microsoft Word/DOCX
  documents. Covers OOXML structure (ZIP of XML parts), style preservation
  over direct formatting, numbering/list system, section-level page layout,
  track changes and comment anchors, TOC/field handling, round-trip
  compatibility, 12 common traps, and python-docx quick reference.
  Source: clawhub.ai/ivangdavila/word-docx
user-invocable: false
---

# Word / DOCX Skill

Best-practice instructions for AI agents handling Microsoft Word documents.
Source: clawhub.ai/ivangdavila/word-docx

## When to Use

Use when the main artifact is a `.docx` file, especially when tracked changes, comments, headers, numbering, fields, tables, templates, or compatibility matter.

## Core Rules

### 1. Treat DOCX as OOXML, not plain text

- A `.docx` file is a ZIP of XML parts — structure matters as much as visible text.
- Critical parts: `word/document.xml`, `styles.xml`, `numbering.xml`, headers, footers, and relationship files.
- Text may be split across multiple runs; never assume one word or sentence lives in one XML node.
- Reading, generating, and preserving an existing reviewed document are different jobs even when the format is the same.
- Legacy `.doc` inputs usually need conversion before you can trust modern `.docx` assumptions.

### 2. Preserve styles and direct formatting deliberately

- Prefer named styles over direct formatting so the document stays editable.
- Styles layer: paragraph styles, character styles, and direct formatting do not behave the same.
- Removing direct formatting is often safer than stacking more inline formatting on top.
- When editing an existing file, extend the current style system instead of inventing a parallel one.
- Copying content between documents can silently import foreign styles, theme settings, and numbering definitions.

### 3. Lists and numbering are their own system

- Bullets and numbering belong to Word's numbering definitions, not pasted Unicode characters.
- `abstractNum`, `num`, and paragraph numbering properties all matter — restart behavior is rarely "visual only".
- Indentation and numbering are related but not identical; a list can have broken numbering even if the indent looks right.
- A list that looks correct in one editor can restart, flatten, or renumber itself later if the underlying numbering state is wrong.

### 4. Page layout lives in sections

- Margins, orientation, headers, footers, and page numbering are section-level behavior.
- First-page and odd/even headers can differ inside the same document.
- Set page size explicitly — A4 and US Letter defaults change pagination and table widths.
- Use section breaks for layout changes; manual spacing and stray page breaks create drift.
- Header and footer media use part-specific relationships — copied IDs often break images or links.
- Table geometry depends on page width, margins, and fixed widths.

### 5. Track changes, comments, and fields need precise edits

- Visible text is not the full document when tracked changes are enabled.
- Insertions, deletions, and comments carry metadata that can survive careless edits.
- Deleted text may still exist in the XML even when it no longer appears on screen.
- Comment anchors and review ranges can break if edits move text without preserving surrounding structure.
- Tables of contents, page numbers, dates, cross-references, and mail merge placeholders are fields — edit the field source carefully and expect cached display values to lag until refresh.
- Hyperlinks, bookmarks, and references can break if IDs or relationships stop matching.
- For review workflows, make minimal replacements instead of rewriting whole paragraphs.
- In tracked-change workflows, only the changed span should look changed; broad rewrites create noisy reviews.
- For legal, academic, or business review documents, default to review-style edits over wholesale paragraph rewrites unless the user explicitly wants a rewrite.

### 6. Verify round-trip compatibility before delivery

- Complex documents can shift between Word, LibreOffice, Google Docs, and conversion tools.
- Tables, headers, embedded fonts, and copied styles are common sources of layout drift.
- Treat `.docm` as macro-bearing and higher risk; treat `.doc` as legacy input needing conversion.
- When layout matters, explicit table widths are safer than auto-fit or percentage behavior that different editors reinterpret.

## Common Traps

1. Copy-paste can import unwanted styles and numbering definitions.
2. Header/footer images use part-specific relationships — reusing IDs blindly breaks them.
3. Empty paragraphs used as spacing make templates fragile; spacing belongs in paragraph settings.
4. A clean-looking export can still hide unresolved revisions, comments, or stale field values.
5. Restarting lists "by eye" usually fails because numbering state lives outside paragraph text.
6. One visible phrase can be split across several runs, bookmarks, revision tags, or field boundaries.
7. Replacing a whole paragraph to change one clause often breaks review quality, bookmarks, comments, or nearby inline formatting.
8. Deleting all visible text from a paragraph can still leave behind an empty paragraph mark or unstable numbering.
9. Table auto-fit and percentage-like width behavior can look acceptable in Word and still drift in Google Docs or LibreOffice.
10. Compatibility mode can silently cap newer features or change pagination behavior.
11. A single change in page size or margin defaults can ripple through tables, headers, TOC, and cross-references.
12. TOC entries, footnotes, and cross-references can look correct until the recipient updates fields and exposes broken anchors.

## python-docx Quick Reference

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Create new document
doc = Document()
doc.add_heading("Title", level=0)
doc.add_paragraph("Body text with ", style="Normal").add_run("bold").bold = True
doc.add_table(rows=3, cols=3, style="Table Grid")
doc.save("output.docx")

# Edit existing document (preserve structure)
doc = Document("template.docx")
for para in doc.paragraphs:
    if "PLACEHOLDER" in para.text:
        for run in para.runs:
            run.text = run.text.replace("PLACEHOLDER", "actual value")
doc.save("filled.docx")
```

## Dependencies

- `python-docx` — reading, creating, and editing `.docx` files
- Install: `pip install python-docx`
