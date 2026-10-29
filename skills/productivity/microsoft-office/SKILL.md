---
name: microsoft-office
description: Create, read, and edit Word, PowerPoint, and Excel files.
version: 2.0.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [office, word, docx, powerpoint, pptx, excel, xlsx, csv, documents, slides, spreadsheets]
    category: productivity
    related_skills: [pdf]
---

# Microsoft Office Skill

Create, read, edit, and template Word (`.docx`), PowerPoint (`.pptx`), and
Excel (`.xlsx`, `.csv`) files offline with small Python CLIs built on
python-docx, python-pptx, and openpyxl. Office itself is never needed. It does
not open legacy binary formats (`.doc`, `.ppt`, `.xls`) and does not render
pages without LibreOffice.

## When to Use

Pick the format, then read its reference with `skill_view` before working:

| File | Reference | Typical asks |
| --- | --- | --- |
| `.docx` | `references/word.md` | reports, letters, contracts, `{{token}}` templates, tracked changes, comments |
| `.pptx` | `references/powerpoint.md` | pitch decks, brand-template decks, chart refreshes, speaker notes, slide PNGs |
| `.xlsx`, `.csv` | `references/excel.md` | styled multi-sheet workbooks, formulas, charts, CSV conversion, row/column restructures |

Deeper notes: `references/revisions-and-comments.md` (Word review markup) and
`references/restructuring.md` (Excel reference shifting).

- Not for `.doc`/`.ppt`/`.xls`: convert first with
  `soffice --headless --convert-to docx|pptx|xlsx old-file` when LibreOffice is installed.
- Not for `.odt`/`.odp`/`.ods`, or WYSIWYG layout work.

## Prerequisites

- Python 3.10+ with `python-docx`, `python-pptx`, and `openpyxl` (import names
  `docx`, `pptx`, `openpyxl`).
- Pick the interpreter with `terminal`. Use `python` when it already imports
  the libraries (the Hermes Docker image ships them). Otherwise, when `hermes`
  is on PATH, let Hermes build its shared, pinned documents environment under
  the Hermes home: it survives container restarts, never touches the Hermes
  install itself, and is safe to run from several agents at once. When it is
  not (the terminal runs in a Docker sandbox, which has no Hermes), the block
  builds a venv in the sandbox home with the same pinned versions. Use this
  exact block in every document skill so they all share one environment:

  ```bash
  python -c "import docx, pptx, openpyxl" 2>/dev/null && PY=python || if command -v hermes >/dev/null 2>&1; then
    PY="$(hermes pm environment documents --extra documents --extra documents-pdf)"
  else
    PY="$HOME/.cache/hermes-docs-venv/bin/python"
    "$PY" -c "import docx, openpyxl, pptx, reportlab, pypdf, pdfplumber, pypdfium2" 2>/dev/null || (
      # Safe to run from several agents at once: each caller builds a private venv,
      # a symlink publishes the first finished one atomically, the others delete theirs;
      # a published build is replaced and deleted only while it still fails to import.
      mkdir -p "$HOME/.cache" && cd "$HOME/.cache" && stage="$(mktemp -d hermes-docs-venv.XXXXXX)" || exit 1
      old="$(readlink hermes-docs-venv)"  # the published build, if any (another agent may have just published it)
      python3 -m venv "$stage" &&
        "$stage/bin/python" -m pip install -q --disable-pip-version-check python-docx==1.2.0 openpyxl==3.1.5 python-pptx==1.0.2 reportlab==5.1.0 pypdf==6.21.2 pdfplumber==0.11.10 pypdfium2==5.14.0 &&
        "$stage/bin/python" -c "import docx, openpyxl, pptx, reportlab, pypdf, pdfplumber, pypdfium2" &&
        if [ -z "$old" ]; then python3 -c 'import os, sys; os.symlink(*sys.argv[1:])' "$stage" hermes-docs-venv
        elif [ "$(readlink hermes-docs-venv)" = "$old" ] && ! "$old/bin/python" -c "import docx, openpyxl, pptx, reportlab, pypdf, pdfplumber, pypdfium2" 2>/dev/null; then
          ln -s "$stage" "$stage.lnk" && python3 -c 'import os, sys; os.replace(*sys.argv[1:])' "$stage.lnk" hermes-docs-venv &&
            case "$old" in hermes-docs-venv.??????) rm -rf "$old" ;; esac
        fi 2>/dev/null
      [ "$(readlink hermes-docs-venv)" = "$stage" ] || rm -rf "$stage" "$stage.lnk"
    )
  fi
  export PY
  "$PY" scripts/docx_create.py --help
  ```

  Run every helper with `"$PY"`. In PowerShell (Hermes runs on that host, so
  it is on PATH):
  `$env:PY = hermes pm environment documents --extra documents --extra documents-pdf`.
  Never `pip install` into the interpreter that runs Hermes or into the
  `hermes pm environment` one; Hermes owns both.
- Optional: LibreOffice (`soffice`) for PDF export, Excel recalculation, and
  slide rendering, plus poppler (`pdftoppm` or `pdftocairo`) for slide PNGs.
  The helpers that need them report what is missing and exit 0.

## How to Run

Every helper lives in `scripts/`, is an argparse CLI with `--help`, prints
JSON to stdout, and exits non-zero on failure. Run them with `terminal` from
the skill directory. Author JSON specs with `write_file`; read results with
`read_file` or straight from stdout.

```bash
"$PY" scripts/docx_create.py spec.json out.docx
"$PY" scripts/pptx_create.py deck.json out.pptx
"$PY" scripts/xlsx_create.py spec.json report.xlsx
```

## Quick Reference

| Task | Word | PowerPoint | Excel |
| --- | --- | --- | --- |
| Create from JSON spec | `docx_create.py` | `pptx_create.py` | `xlsx_create.py`, `csv_to_xlsx.py` |
| Read back | `docx_read.py --text/--structure` | `pptx_read.py --outline` | `xlsx_read.py --sheets/--json` |
| Edit | `docx_edit.py` | `pptx_edit.py` | `xlsx_edit.py`, `xlsx_restructure.py` |
| Fill a template | `docx_template.py` | `pptx_from_template.py` | — |
| Review / check | `docx_revisions.py`, `docx_comments.py`, `docx_validate.py` | `pptx_render.py` (slide PNGs) | `xlsx_recalc.py`, `xlsx_to_csv.py` |

The per-format reference has the full flag table.

## Procedure

1. Identify the format from the file extension or the ask, and load its
   reference (`skill_view` with `file_path=references/<format>.md`).
2. Set `PY` with the Prerequisites block once per session.
3. Create, edit, or read with that format's helpers; pass an output path to
   keep the original whenever the helper supports one.
4. Verify the result as the reference's Verification section says, then give
   the user the exact output path.

## Converting to PDF

When LibreOffice is installed, convert headlessly:

```bash
soffice --headless --convert-to pdf --outdir out/ file.docx   # also .pptx, .xlsx
soffice --headless --convert-to csv --outdir out/ report.xlsx # first sheet only
```

Check first with `command -v soffice || command -v libreoffice`. Fonts not on
the host are substituted, so render-check decks before shipping. There is no
pure-Python path to PDF for any of these formats: if `soffice` is missing, say
so rather than approximating. For other sheets use
`xlsx_to_csv.py --sheet NAME`.

## Pitfalls

- **Never unzip-and-edit the XML.** These files are zip packages; raw text
  substitution corrupts them. Edit through the helpers (or the libraries) and
  use `patch`/`write_file` only for JSON inputs.
- **Text is split into runs.** Word and PowerPoint fragment text at edit
  boundaries; the replace helpers merge equal-format runs first, but a match
  across differently formatted runs takes the first run's formatting.
- **Nothing computes fields or formulas.** Word TOC/page fields and Excel
  formulas are filled in by the office app on open (or by `xlsx_recalc.py`).
- **Encoding.** Write specs as UTF-8; the helpers read UTF-8 explicitly.
- Format-specific traps are in each reference.

## Verification

- Re-read every output with the format's read helper and check the expected
  content is there and the old content is gone.
- Word: `docx_validate.py out.docx` exits 0 after revision, comment, or field
  surgery.
- PowerPoint: render with `pptx_render.py` and check each PNG with
  `vision_analyze` when the render tools exist.
- Excel: `xlsx_read.py --formulas` lists written formulas; run
  `xlsx_recalc.py` when cached values matter.
- The bundled tests are the full contract:
  `"$PY" -m pytest tests/ -q` (needs pytest in that interpreter).
