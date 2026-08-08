---
name: genoffice
description: "Create and byte-preserving-edit .docx/.pptx via the GenOffice MCP server."
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Office, Documents, Presentations, MCP, Docx, Pptx]
    homepage: https://github.com/criptogus/mcp-genoffice
    related_skills: [powerpoint, docx, office-document-generation]
---

# GenOffice MCP — byte-preserving Office editing

Create and edit `.docx` / `.pptx` files with **byte-preserving roundtrips**
through the GenOffice engines (pure TypeScript, Apache-2.0, no Electron
needed). Only the blocks/elements you touch are regenerated; everything else
in the file keeps its original bytes — layout, styles, headers, comments and
zip parts survive. This is the fidelity edge over python-docx/python-pptx,
which rewrite files wholesale.

## Install

```bash
hermes mcp install genoffice    # catalog entry (git-installs mcp-genoffice)
```

First tool call takes ~40s (one-time: the server clones the pinned GenOffice
engines into `~/.cache/mcp-genoffice/src` and runs `npm install`).

## Tools

| Tool | Purpose |
| --- | --- |
| `genoffice_extract_text` | Read text from .docx/.xlsx/.pptx/.pdf (structure preserved) |
| `genoffice_docx_blocks` | List docx blocks (index, type, style, text) — read before patching |
| `genoffice_docx_patch` | Rewrite paragraphs (byte-preserving; writes a NEW file) |
| `genoffice_docx_watermark` | Set/remove a text watermark (header only, body untouched) |
| `genoffice_docx_create` | New .docx from scratch, optional initial paragraphs |
| `genoffice_docx_delete` | Delete blocks — remaining blocks keep original bytes |
| `genoffice_pptx_slides` | List slides + text elements (id, name, type, text) |
| `genoffice_pptx_patch` | Replace element text on a slide (element-level byte-preserving) |
| `genoffice_pptx_create` | New .pptx from scratch (one blank slide) |
| `genoffice_pptx_delete` | Delete elements from a slide |
| `genoffice_app_status` | Is the GenOffice desktop app installed / CDP port up |
| `genoffice_app_launch` | Launch the app with the CDP debug port (handles auto-update relaunch) |
| `genoffice_app_open_file` | Open a file in the app (macOS `open -a`, registered doc types) |
| `genoffice_app_screenshot` | PNG screenshot of the app window over CDP |
| `genoffice_app_eval` | Evaluate read-only JS in the app page context (DOM) |

## Workflow

1. **Read** — `genoffice_docx_blocks` / `genoffice_pptx_slides` to map the file.
2. **Edit** — `genoffice_docx_patch` / `genoffice_pptx_patch` with the indexes
   / element names from step 1. Multi-line text: `\n` in pptx = new paragraph.
3. **Verify** — re-run the read tool on the output path; confirm untouched
   content is identical.
4. All edits write a **new file** (`<name>.patched.docx` etc. by default) —
   the original is never modified.
5. **Visual check (optional)** — `genoffice_app_launch` + `genoffice_app_open_file`
   + `genoffice_app_screenshot` to inspect the result in the real app.

## Pitfalls

- **pptx element ids are NOT stable across saves** (engine renumbers
  `cNvPr`). Address elements by **name** (`Title 1`, `Subtitle 2`), not id.
- pptx patch uses the element's **first run as the formatting template** —
  font/size/color/alignment/bullets of the first paragraph carry to the new
  text.
- xlsx: text extraction works; **editing xlsx is not yet covered** (the
  GenOffice sheets engine is a Rust sidecar — roadmap).
- The GenOffice **app** (optional) auto-updates via Squirrel and relaunches
  without CDP flags — the headless tools above don't touch the app.
- Errors are actionable: out-of-range indexes/elements tell you to re-run the
  read tool.

## References

- Server repo: https://github.com/criptogus/mcp-genoffice
- Engine source: https://github.com/genspark-ai/genoffice (Apache-2.0)
