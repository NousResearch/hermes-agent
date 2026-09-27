---
name: litkit-deliverable
description: Build, quote-check, and commit work product to LitKit.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litkit, deliverable, docx, pdf, quote-check]
    related_skills: [litkit-corpus-pull, legal-cite-check, deposition-prep-package, discovery-letter-brief]
---

# LitKit Deliverable Skill

Turn a finished draft into a matter record: build the .docx or .pdf from a source file with the host's tool venv, check every quotation against the matter's documents, then commit it through LitKit's deliverable gates and report what the gates found. The gates are a floor, not the verification: the drafting skill's own checks run first.

## When to Use

- A memo, letter, outline, brief, or analysis is ready to hand to the case team.
- A new version of a document already in the matter's LitSpace (`documentId`).

## Prerequisites

- The `litkit` toolset on a matter host.
- The tool venv on the host with `python-docx`, PyMuPDF, and LibreOffice (`soffice`) for PDF rendering.
- The draft's source (Markdown plus a build script) in the thread's working directory.

## What reaches the thread

Two different acts, two tools:

- **Attach to the reply** (`litco_deliver_local`): the file is handed to the lawyer with this turn's answer. No gates, nothing committed to the matter.
- **Commit to the matter** (`litkit_deliver`): the file goes through LitKit's deliverable gates and becomes a matter record.

A turn attaches exactly (1) every file registered with `litco_deliver_local path=<file> [name=<shown name>] [deliverableClass=<class>]`, and (2) any other file created or changed in `deliverables/` whose type is .docx, .xlsx, .pptx, .pdf, .md, .txt, .csv, .png, or .jpg. Everything else stays on the host: dotfiles, `*.spec.json`, `*.tmp`, Office lock files (`~$*`), JSON, scripts, zips. Keep build specs, data, and scratch outside `deliverables/` (for example in `build/`), and register every file you mean to hand over, including one of an allowed type, so its class travels with it. A zip or JSON the lawyer asked for ships only when registered.

## Quick Reference

| Step | Tool |
|---|---|
| Build .docx / render .pdf | `terminal` with the tool venv (`python build_x.py`, `soffice --headless --convert-to pdf`) |
| Verify it exists | `terminal` `ls -l deliverables/<file>` (size above zero), or `read_file` |
| Look at the rendered pages | `vision_analyze` on PNGs from PyMuPDF |
| Quote check | `litkit_quote_check` path=deliverables/<file> deliverableClass=<class> |
| Commit | `litkit_deliver` path=deliverables/<file> deliverableClass=<class> [documentId=<uuid>] |
| Attach to the reply | `litco_deliver_local` path=deliverables/<file> deliverableClass=<class> |
| Tell someone | `litkit_notify` userId=<member> title=… link=… |

The order is fixed: **build → verify the file exists → quote-check → deliver.** `litkit_deliver` and `litkit_quote_check` read a file that is already on disk; called on a path that has not been written, they return `file_missing: true` and do nothing.

Classes: `draft`, `memo`, `brief`, `filing`, `pleading`, `production` (aliases `letter`, `motion`, `complaint`, `client_memo`, `work_product`). Pick the class the document is; `pleading` and `filing` block harder.

## Procedure

1. **Build from source.** Generate the .docx from its Markdown or data with a build script kept in the working directory (the script and any spec file outside `deliverables/`). Revisions change the source and regenerate; never hand-edit a generated .docx, because the file and its source must not diverge. Render a PDF with `soffice --headless --convert-to pdf` and inspect the pages that carry tables, signature blocks, and tab-stop layouts as images.
2. **Verify the file exists.** Check that the build wrote the file where you will point the tools (`ls -l deliverables/<file>`, size above zero). A build command that exits non-zero shows as a failed tool call; read its error and rebuild before going on. If a tool answers `file_missing`, the build did not write that path: fix the build, do not retry the tool.
3. **Run the drafting skill's own checks first.** Quotations copied programmatically from `texts/` and substring-matched against the cited file; transcript cites against the certified final; docket cites against the as-filed ECF copy. The LitKit gate re-verifies quotations it can bind to matter documents; it cannot see sources outside the matter.
4. **Quote check.** `litkit_quote_check` on the built file. It writes nothing to LitKit and saves full findings under `qa/`. Fix every `unverified` and `hardFail` item in the source, rebuild, and rerun until clean, or mark the item for the user.
5. **Commit.** `litkit_deliver` with the file, the class, and `documentId` when this is a new version. The response carries `blocked`, the gate summary (quotes checked and verified, pleading, citation, and prose flags), and `fullFindings` (the saved response under `qa/`).
6. **When blocked.** Nothing was saved. Report the gate findings to the user in plain terms (which quotation, which allegation, which citation) and fix the source or ask. Do not resubmit the same bytes, and do not switch to a softer class to get past a block.
7. **Attach.** Register the committed file, and every other file the lawyer should receive, with `litco_deliver_local` (same class). Do not register scratch.
8. **Report.** Tell the user where the document landed (`path`, version number), what the gates checked (counts, not adjectives), what remains open, and attach nothing that was not built from the committed source. If someone else should see it, `litkit_notify` them with the link.

## Pitfalls

- `quoteGateError` means the quote pass could not run: the quotations are unchecked, not verified. Say so.
- `bytesRewritten` means a text-format gate appended a strike list or appendix to the stored copy; tell the user.
- A text deliverable (.md, .txt) or a new version needs a matter with a LitSpace binding (`409 not_bound` otherwise); deliver the .docx or .pdf instead.
- `permission_denied` means the lawyer on the turn lacks `repository.write` on the matter. Attach the file with `litco_deliver_local` (it still reaches the thread) and say who can commit it.

## Verification

- The committed file is the one the last quote check ran on (compare `sha256`).
- The report to the user states the gate counts and every open item.
