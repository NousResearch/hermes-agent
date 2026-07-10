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

- Excel (desktop) on Windows or macOS with add-in sideloading allowed.
- Node.js 18+ on PATH.
- A running Hermes gateway with the `api_server` platform enabled on
  `http://127.0.0.1:8642/v1` (`hermes gateway run`). The key is read automatically
  from the Hermes `config.yaml` for your platform (Windows `%LOCALAPPDATA%\hermes`,
  macOS `~/Library/Application Support/hermes`, Linux `$XDG_CONFIG_HOME/hermes`);
  set `HERMES_EXCEL_LLM_API_KEY` to override.
- **Tool containment (defense-in-depth)**: the bridge NEVER executes model tool
  calls — only the workbook-actions JSON it returns — so the agent's tools are
  already inert here. Still, run the api_server platform with
  file/terminal/code-execution toolsets disabled so nothing runs agent-side either:

  ```yaml
  # config.yaml
  platform_toolsets:
    api_server:
      - skills
      - memory
      - vision
  ```

- Optional: [Docling](https://github.com/docling-project/docling) for
  PDF/DOCX/XLSX/image parsing. Set `HERMES_EXCEL_DOCLING_MODE` to match how it runs
  relative to the bridge: `wsl` (default on Windows — Docling in WSL), `native`
  (shares this host's filesystem), or `docker` (mounted). Default is `native` off
  Windows.

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
HERMES_EXCEL_DATA_DIR=                     # uploads/exports/logs root, OUTSIDE the web root (default: per-user app-data)
HERMES_EXCEL_BRIDGE_TOKEN=                 # when set, every /api/* call requires it (the installer sets one per box)
HERMES_EXCEL_ALLOWED_ORIGINS=              # extra comma-separated CORS origins (loopback origins always allowed)
HERMES_EXCEL_LLM_BASE_URL=http://127.0.0.1:8642/v1
HERMES_EXCEL_LLM_MODEL=hermes-agent
HERMES_EXCEL_LLM_API_KEY=                  # auto-read from the platform Hermes config.yaml if unset
HERMES_EXCEL_LLM_TIMEOUT_MS=180000         # per model call
HERMES_EXCEL_LLM_REQUEST_BUDGET_MS=420000  # end-to-end deadline per /api/chat (Docling + all retries)
HERMES_EXCEL_LLM_MAX_TOKENS=8000
HERMES_EXCEL_MAX_PROMPT_CHARS=180000       # assembled-prompt cap (trims oldest history first)
HERMES_EXCEL_LOCK_TOOLS=                   # 1 => also send tool_choice:none to the gateway
HERMES_EXCEL_DOCLING_URL=http://127.0.0.1:8200
HERMES_EXCEL_DOCLING_MODE=                 # wsl | native | docker (default: wsl on Windows, else native)
HERMES_EXCEL_WSL_DISTRO=Ubuntu-24.04       # used only when DOCLING_MODE=wsl
HERMES_EXCEL_DOCLING_OUTPUT_DIR=           # if set, Docling result paths must resolve under it
HERMES_EXCEL_MAX_EXTRACTED_CHARS_PER_FILE=32000
HERMES_EXCEL_MAX_EXTRACTED_CHARS_TOTAL=96000
HERMES_EXCEL_MAX_UPLOAD_BYTES=26214400     # 25 MB per attachment
HERMES_EXCEL_MAX_UPLOAD_FILES=12
HERMES_EXCEL_UPLOADS_TTL_MS=604800000
HERMES_EXCEL_DISABLE_CSP=                  # 1 => omit the pane Content-Security-Policy (troubleshooting only)
```

The bridge binds 127.0.0.1 only, serves **only** the pane's own files (never the
broker source, uploads, exports, or manifest), restricts CORS to an origin
allowlist, rejects foreign `Host` headers, and — when a token is set — requires it
on every `/api/*` call. The api_server key autodetect is cross-platform.

### Fleet install (Windows)

For an unattended per-box install — Node bootstrap, an autostarting bridge
supervisor, Office sideload registration, and a post-install health check — see
[install/README.md](./install/README.md):

```powershell
powershell -ExecutionPolicy Bypass -File install\apply.ps1
```

The installer selects the first available port starting at `8787`, rewrites the
sideloaded manifest to match, and fails if an explicitly requested port belongs
to another service. Autostart uses one tracked Scheduled Task supervisor.

## Tests

Dependency-free, no install step:

```powershell
node --test broker\server.test.mjs     # 39 unit tests (parsing, actions, formula rebasing, model-retry branches)
node broker\smoke.mjs                  # live regression + security cases against a running bridge
```

There is intentionally no `package.json` here: the bridge has zero
dependencies, and omitting it keeps this directory out of the repo's npm
workspace/lockfile graph.

## Notes

- Attachment temp files and explicit CSV exports land under `HERMES_EXCEL_DATA_DIR`
  (`uploads/`, pruned after 7 days; `exports/`) — a per-user directory **outside**
  the HTTP-served root, so neither is ever reachable over HTTP. `/api/export` is the
  bridge's only file-writing path and runs only on an explicit export action.
- The default output style is tuned for accounting/finance tables (header
  rows, total rows, currency formats, formulas over hardcoded totals); see the
  system prompt in `broker/server.mjs` to adjust.
