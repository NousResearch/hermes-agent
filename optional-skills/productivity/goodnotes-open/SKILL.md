---
name: goodnotes-open
description: Open .goodnotes notebooks from local or cloud storage.
version: 0.1.0
author: Emil, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [goodnotes, notes, pdf, conversion, cloud]
    related_skills: []
---

# GoodNotes (.goodnotes) Skill

Opens GoodNotes 5/6 notebook files (`.goodnotes`) so their content can be read, inspected, and converted to readable formats (PDF, SVG, JSON). A `.goodnotes` file is a proprietary container, so "opening" means locating it, unpacking it, and converting it with open-source tools — not rendering it natively. The file may live on the local disk or in a connected cloud service (Google Drive, Dropbox, OneDrive, iCloud, Goodnotes Cloud).

## When to Use

- User asks to open, read, view, or convert a `.goodnotes` file.
- User asks what is inside a `.goodnotes` file (pages, strokes, images, audio, PDFs).
- User points at a `.goodnotes` file in Google Drive, Dropbox, OneDrive, iCloud, or another configured cloud service.
- User wants a notebook exported to PDF/SVG/JSON without the GoodNotes app.

Don't use for: editing strokes in place (no reliable open-source writer exists), or opening `.note` (Notability) / `.rm` (reMarkable) files — those are different formats.

## Prerequisites

- Python 3.9+ with `pip` (for the conversion tools).
- One conversion tool installed (see Quick Reference). `goodnotes2pdf` is the simplest for PDF output; `parser-for-goodnotes` (`gn-export-*`) is best for JSON/SVG and deep inspection.
- For cloud files: a connected account for the relevant service (Google Drive, Dropbox, OneDrive, iCloud). Check with `manage_connections(action='status')`; connect missing ones with `manage_connections(action='connect', connectors=[...])`.
- The helper script `scripts/inspect_goodnotes.py` (ships with this skill) for quick inspection.

## How to Run

1. **Locate the file.** If it is in cloud storage, fetch it to a local path first (download via the service's connector/CLI), then work on the local copy. If it is local, note the path.
2. **Verify and inspect** with the helper script:
   `terminal(command="python3 scripts/inspect_goodnotes.py <file.goodnotes>", timeout=60)`
3. **Convert** to the requested format with the chosen tool (below).
4. **Deliver** the output file to the user via `MEDIA:/absolute/path`.

## Quick Reference

```
# Inspect (verify ZIP, list members, identify attachments)
python3 scripts/inspect_goodnotes.py notes.goodnotes

# Convert to PDF (goodnotes2pdf; needs pymupdf)
python3 goodnotes2pdf.py convert notes.goodnotes outdir

# Convert to PDF / SVG / JSON (parser-for-goodnotes; pip install -e .)
gn-export-pdf notes.goodnotes -o notes.pdf
gn-export-svg notes.goodnotes -o out_svgs/
gn-export-json notes.goodnotes -o notes.json
gn-inspect notes.goodnotes

# Extract embedded attachments (images/audio/PDFs)
gn-extract notes.goodnotes outdir
```

## Procedure

1. **Resolve the file location.**
   - Local path: use it directly.
   - Cloud (Google Drive, Dropbox, OneDrive, iCloud, Goodnotes Cloud): confirm the account is connected, download the file to a scratch path, and use that local copy. Never operate on a remote path directly.
2. **Confirm it is a `.goodnotes` file.** Run the inspect script. It must report a valid ZIP container and a `schema.pb` member. If it is not a ZIP, stop and tell the user the file is corrupt or not a GoodNotes notebook.
3. **Inspect the contents** (optional but recommended before converting): the script lists `notes/<UUID>` page files, `attachments/<UUID>` media, and the `index.*.pb` indexes, and identifies attachment types by magic bytes.
4. **Convert to the target format.**
   - PDF: `goodnotes2pdf.py convert <file> <outdir>` (or `gn-export-pdf`).
   - SVG/JSON: `gn-export-svg` / `gn-export-json`.
   - Extract media: `gn-extract`.
5. **Verify the output** exists and is non-empty (`read_file` or `search_files target='files'` on the output path).
6. **Deliver** the result to the user with `MEDIA:/absolute/path/to/output`.

## Pitfalls

- **`.goodnotes` is proprietary and undocumented.** The format was reverse-engineered; a future GoodNotes update can change it. If a tool fails on a newer file, say so rather than fabricating output.
- **Coordinates are in 1/132-inch units.** `goodnotes2pdf` scales to PDF points (72/132) by default — do not change `--scale` unless you know what you are doing.
- **Schema versions differ (24 vs 25).** Schema-25 (Mac exports) uses a journal structure in page files; some tools handle only one. If conversion fails, try a different tool.
- **Attachments have UUID filenames with no extension.** Identify them by magic bytes (the inspect script does this), not by name.
- **Cloud files must be downloaded first.** Connectors expose remote files; always materialize a local copy before converting.
- **No reliable open-source writer exists.** You can read/convert, but not faithfully edit and re-save a `.goodnotes` file. For editing, the user must use the GoodNotes app.
- **Auto-backup vs sync.** GoodNotes auto-backup (Google Drive/Dropbox/OneDrive) stores `.goodnotes` copies; iCloud sync and Goodnotes Cloud store the library differently. If a file is not where expected, ask which sync method the user uses.

## Verification

- The inspect script prints `Valid ZIP container` and lists members — proves the file opened.
- The conversion tool exits 0 and the output file exists and is non-empty.
- For PDF output, open it with `read_file` (PDF text layer) or `vision_analyze` on a rendered page to confirm content is present.
