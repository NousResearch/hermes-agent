---
name: typst
description: Compile and scaffold Typst projects to PDF, HTML, and PNG.
version: 1.0.0
author: Thomas Bale (TumCucTom)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [typst, typesetting, pdf, latex-alternative, academic-writing, documents]
    related_skills: [pretext, research-paper-writing]
    category: creative
---

# Typst

A modern typesetting system designed as a successor to LaTeX. Compiles `.typ` source files to PDF, HTML, and PNG. Ships as a single Rust binary (`typst`) with a sub-second incremental compiler, built-in package management, and a markup-based syntax that reads closer to Markdown than to TeX.

This skill scaffolds Typst projects and compiles them through Hermes. Use it whenever the user wants a typeset document — papers, reports, slides, letters, CVs, theses — and especially when they ask for a "LaTeX alternative" or complain about LaTeX's compile times or syntax.

## When to Use

- User asks for a paper, article, report, thesis, slides, or any typeset document
- User mentions Typst by name
- User says "modern LaTeX" / "LaTeX alternative" / "faster than LaTeX"
- User wants HTML output alongside PDF (e.g. for a blog or web view)
- User asks to convert or rewrite a LaTeX document

Don't use for:
- Plain Markdown documents — write `.md` directly
- Spreadsheets or data tables — use a different tool
- Image-only output (Posters, banners) — use the image generation stack
- Heavy bibliographic workflows with `.bib` files and BibTeX-style citations — Typst's `hayagriva` is simpler but the model is different; only use if the user opts in

## Prerequisites

1. **Install the Typst CLI** (Rust binary, ~30 MB):

   ```bash
   # macOS / Linux — official installer (downloads to ~/.local/bin)
   curl -fsSL https://typst.community/typst-install/install.sh | bash

   # macOS via Homebrew
   brew install typst

   # Windows via winget
   winget install --id Typst.Typst

   # Verify
   typst --version
   ```

   Hermes calls `typst` through the `terminal` tool. If the binary is on `PATH` and `typst --version` succeeds, the skill works. If not, the verification step at the bottom of this skill will fail — install first, then retry.

2. **No Python deps.** The helper script in `scripts/compile.py` uses only stdlib. There is nothing to `pip install`.

## How to Run

All operations go through the `terminal` tool, invoking `typst` directly or calling `scripts/compile.py` for project scaffolding:

- **Scaffold a new project**: `python scripts/new_project.py <name> --type article|report|slides`
- **Compile once**: `typst compile main.typ main.pdf`
- **Compile and watch** (re-render on save): `typst watch main.typ main.pdf`
- **Compile to HTML**: `typst compile --format html main.typ main.html`
- **Compile to PNG** (one image per page): `typst compile --format png main.typ main-{p}.png`
- **Install a Typst package from the registry** (e.g. `@preview/charged-ieee`): `typst init @preview/charged-ieee`

The agent drives `typst` from the `terminal` tool — never call it through Python unless wrapping it (as `compile.py` does).

## Quick Reference

| Command | What it does |
|---|---|
| `typst compile <in> <out>` | One-shot compile `<in>.typ` to `<out>.pdf` |
| `typst watch <in> <out>` | Compile and re-render on save (Ctrl+C to stop) |
| `typst compile --format html <in> <out>` | HTML output (single self-contained file) |
| `typst compile --format png <in> <out>` | PNG per page (use `--ppi` for resolution) |
| `typst init @preview/<pkg>` | Add a package to the current project |
| `typst fonts` | List fonts available to the compiler |
| `typst help <cmd>` | Built-in help for any subcommand |

Format strings: `--format pdf` (default), `html`, `png`, `svg`.

## Procedure

### 1. Decide the project type

Ask (or infer) what the user wants:
- `article` — single-column paper, default for most writing
- `report` — multi-section long-form with chapters
- `slides` — landscape PDF for presentations (Typst has native slide support)
- `custom` — start from `templates/minimal.typ` and let the user direct

### 2. Scaffold the project directory

```bash
python scripts/new_project.py my-paper --type article
```

This creates `my-paper/` with `main.typ`, `refs.bib` (placeholder), `figures/` (empty), and a `.gitignore`. The user can open `main.typ` immediately and start writing.

### 3. Compile on demand

After the user has written or edited `main.typ`:

```bash
cd my-paper && typst compile main.typ main.pdf
```

Report any compile errors verbatim — Typst's error messages include the file, line, and column, and they're usually self-explanatory.

### 4. Watch for live editing (optional)

If the user is iterating:

```bash
cd my-paper && typst watch main.typ main.pdf
```

This blocks. Run it in a backgrounded `terminal` call if the agent needs to keep working. Stop with `process(action='kill', id=<id>)` when the user is done iterating.

### 5. Hand off the PDF

Return the file path. The user can open it with `MEDIA:/path/to/main.pdf` or by reading the path directly.

## Pitfalls

- **Binary not on `PATH`.** The most common failure. If `typst --version` returns "command not found", the install step above didn't reach the user's shell — direct them to either add `~/.local/bin` to `PATH` or use the Homebrew/winget install.
- **Missing font.** Typst errors with `unknown font` if a font referenced in the source isn't installed. Either install the font or change the `#set text(font: "...")` declaration. For LaTeX-style fonts (Computer Modern, Latin Modern), use the `cmbright` or `lipsum` packages from the registry.
- **Package not found.** `import "@preview/foo:1.2.3"` requires the version to be in the registry. If the error says "package not found", the user is either offline, the name is wrong, or they need `typst init @preview/foo` first.
- **Watched compile blocks.** `typst watch` does not exit — it stays attached. Always background it when calling from a long-lived `terminal` session, and surface the `terminal` call's ID so the user (or agent) can kill it later.
- **HTML output loses some features.** `typst compile --format html` doesn't render custom `#raw` blocks or 3rd-party packages that depend on the PDF backend. Stick to PDF unless the user explicitly wants web output.
- **Cross-platform path separators.** Typst accepts `/` in `#include` and image paths on every platform. Don't use `pathlib.Path` for Typst source paths — feed it strings with forward slashes.
- **First compile is slow on big projects.** Initial compile scans the package cache. Subsequent compiles (and `typst watch`) are sub-second. If the user complains about a slow first build, that is normal.

## Verification

A single command proves the skill is wired up:

```bash
typst --version && python scripts/compile.py --self-test
```

Expected: `typst 0.x.y` (or similar) followed by `OK: typst binary invoked successfully`. If `typst` is missing, the first command fails with "command not found" — fix by installing. If `compile.py --self-test` fails, the script is broken and needs a re-read of `scripts/compile.py`.
