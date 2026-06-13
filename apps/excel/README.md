# Hermes for Excel

A local Excel sidecar for Hermes: an Office.js task pane plus a small
zero-dependency Node bridge that connects a live workbook to the Hermes
`api_server` platform. Chat with Hermes in a side panel while it reads sheet
data, writes tables and formulas, formats ranges, and parses attached
documents — without ever touching your filesystem.

```
Excel task pane (Office.js)  →  bridge :8787 (node, no deps)  →  Hermes api_server :8642/v1
```

See [PROTOCOL.md](./PROTOCOL.md) for the full bridge protocol, the actions
schema, the read loop, and the formula-anchoring contract.

## Features

- **Model-first chat** with per-workbook conversation memory (last 12 turns,
  persists across pane reopens) and a Clear button.
- **Workbook actions**: write cells, create formatted sheets, apply styles,
  run scoped Office.js — all via a structured JSON contract, applied by the
  pane, never by agent tools.
- **Cross-sheet reads**: Hermes sees every sheet's used range and can request
  cell values (`read_range`, up to 5 rounds) for tie-outs and reconciliations.
- **Correct formula placement**: the model authors table formulas A1-relative;
  the bridge rebases them to wherever the table lands (`=B2*C2` written at
  H23 becomes `=I24*J24`; `$`-anchors and cross-sheet refs are never shifted).
- **Post-write verification**: the pane re-reads written ranges and warns on
  error cells or all-zero formula columns.
- **Undo** (last 10 changes), **Cancel** for in-flight requests, and an
  optional **review-before-apply** mode (Apply/Discard per change set).
- **Attachments**: drag/drop PDF/Office/CSV/text/images; TXT/CSV parse
  locally, the rest through a [Docling](https://github.com/docling-project/docling)
  service when available.
- **Honest failures**: if the model is unreachable, the bridge explains what
  failed — it never writes invented numbers into a workbook.
- **JSON resilience**: stray/missing-bracket repair plus one corrective retry
  for near-valid model output.

## Requirements

- Excel for Windows (desktop) with add-in sideloading allowed.
- Node.js 18+ on PATH.
- A running Hermes gateway with the `api_server` platform enabled on
  `http://127.0.0.1:8642/v1` (`hermes gateway run`).
- **Recommended — tool containment**: run the api_server platform with
  file/terminal/code-execution toolsets disabled, so the workbook actions JSON
  is the agent's only output channel:

  ```yaml
  # config.yaml
  platform_toolsets:
    api_server:
      - skills
      - memory
      - vision
  ```

- Optional: Docling at `http://127.0.0.1:8200` for PDF/DOCX/XLSX/image parsing.

## Run

From this directory:

```powershell
node broker\server.mjs
```

Then sideload in Excel: **Home → Add-ins → More Add-ins → Upload My Add-in →
`manifest.xml`**, and open the pane from the **Hermes** ribbon group.

Configuration is environment-variable based (defaults shown):

```text
PORT=8787
HERMES_EXCEL_LLM_BASE_URL=http://127.0.0.1:8642/v1
HERMES_EXCEL_LLM_MODEL=hermes-agent
HERMES_EXCEL_LLM_API_KEY=        # read from %LOCALAPPDATA%\hermes\config.yaml on Windows if unset
HERMES_EXCEL_LLM_TIMEOUT_MS=180000
HERMES_EXCEL_LLM_MAX_TOKENS=8000
HERMES_EXCEL_DOCLING_URL=http://127.0.0.1:8200
HERMES_EXCEL_MAX_EXTRACTED_CHARS_PER_FILE=32000
HERMES_EXCEL_MAX_EXTRACTED_CHARS_TOTAL=96000
HERMES_EXCEL_UPLOADS_TTL_MS=604800000
```

On macOS/Linux hosts set `HERMES_EXCEL_LLM_API_KEY` explicitly (the config.yaml
key autodetect is Windows-only). The bridge binds 127.0.0.1 only.

## Tests

Dependency-free, no install step:

```powershell
node --test broker\server.test.mjs     # 27 unit tests (parsing, actions, formula rebasing)
node broker\smoke.mjs                  # live regression cases against a running bridge
node broker\debug-llm.mjs broker\debug-body-h23.json   # dump a raw model reply for a saved request
```

There is intentionally no `package.json` here: the bridge has zero
dependencies, and omitting it keeps this directory out of the repo's npm
workspace/lockfile graph.

## Notes

- Attachment temp files land in `uploads/` (pruned after 7 days); explicit CSV
  exports land in `exports/`. `/api/export` is the bridge's only file-writing
  path and runs only on an explicit export action.
- The default output style is tuned for accounting/finance tables (header
  rows, total rows, currency formats, formulas over hardcoded totals); see the
  system prompt in `broker/server.mjs` to adjust.
