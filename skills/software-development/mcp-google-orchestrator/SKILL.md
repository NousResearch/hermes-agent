---
name: mcp-google-orchestrator
description: "Use when Anton runs `python agent.py` in `~/Downloads/MCP/` to install, check, or diagnose Gmail/Calendar/Drive/Sheets MCP servers in Hermes. Covers the 5-agent architecture, OAuth flow, the run_with_retries backoff, and the dynamic tools-count discovery (3 strategies) that replaced hardcoded `expected_tools`."
version: 1.0.0
author: Hermes Agent (audit session 2026-07-18)
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [mcp, google, gmail, calendar, drive, sheets, hermes, oauth, audit]
    related_skills: [mcp-server-setup, mcp-server-troubleshooting, hermes-agent]
---

# mcp-google-orchestrator

Operating guide for Anton's `~/Downloads/MCP/` workflow — the Python orchestrator that installs and maintains the four Google MCP servers (`gmail`, `calendar`, `google-drive`, `google-sheets`) wired into Hermes.

**When to use this skill:** user asks anything about MCP Google installation, OAuth flow, OAuth keys location, service failures, drift in tools counts, or runs `python agent.py` from `~/Downloads/MCP/`.

**Skip this skill:** question is about a *different* MCP server, a *different* framework (Claude Code, Codex), or unrelated Hermes topics.

## Architecture (5 agents)

Located at `C:\Users\anton\Downloads\MCP\`. Backup at `MCP.bak.<YYYYMMDD-HHMMSS>\`.

| File | Role |
|---|---|
| `agent.py` | Launcher: menu, preflight, routing, diagnostic |
| `run_mcp_workflow.py` | Common engine: detection, install, OAuth, config, tests, retry x5 |
| `install-mcp-gmail.py` | Gmail-dedicated installer (uses `@gongrzhe/server-gmail-autoauth-mcp`) |
| `install-mcp-calendar.py` | Calendar-dedicated installer (uses `@gongrzhe/server-calendar-autoauth-mcp`) |
| `install-mcp-drive.py` / `install-mcp-sheets.py` | Thin wrappers over the workspace engine (uses `@us-all/google-drive-mcp` via `GD_TOOLS=drive` or `GD_TOOLS=sheets`) |

Both Gmail and Calendar use the dedicated `@gongrzhe/*` packages (separate OAuth keys, separate credentials.json). Drive and Sheets share a single workspace package (`@us-all/google-drive-mcp`) — OAuth keys and credentials live in `~/.google-workspace-mcp/`, the PowerShell wrapper selects the tool category via `GD_TOOLS=`.

## Standard paths (use verbatim)

- `@user` = `C:\Users\<name>` (Anton's machine)
- `@mcp` = `@user\Downloads\MCP` (orchestrator root)
- `@gmail` = `@user\.gmail-mcp`
- `@calendar` = `@user\.calendar-mcp`
- `@workspace` = `@user\.google-workspace-mcp`
- `@hermes_config` = `@user\AppData\Local\hermes\config.yaml`
- npm packages land in `@user\.local\share\<service>-mcp-server\node_modules\<scoped-pkg>\`

## Three executable commands (from `@mcp`)

```bash
python agent.py                          # Interactive QCM menu
python agent.py --install <service|all>  # Headless install
python agent.py --check                  # Non-destructive state check
python agent.py --diagnose               # Full preflight + per-service state
```

**Never run install commands without backup.** Pre-flight check is read-only and safe.

## OAuth flow (the 5 steps that actually matter)

1. Verify `@<service>` directory exists, copy client OAuth keys from `client_secret_*.json` if missing.
2. Play `winsound.MessageBeep` (or terminal bell on non-Windows) — then open the OAuth window.
3. User clicks Authorize in the browser. Script polls `credentials.json` for `refresh_token`.
4. **NEVER print tokens.** `credentials.json` is the durable artifact; everything else is transient.
5. Backup `@hermes_config` before any write — pattern `.bak.<YYYYMMDD_HHMMSS>` is already in `backup_file()`.

If OAuth fails, do **not** retry blindly. Check: (a) port 3000 free, (b) client OAuth is the right project, (c) API is enabled in Google Cloud Console (`gmail.googleapis.com`, `calendar-json.googleapis.com`, `drive.googleapis.com`, `sheets.googleapis.com`).

## Dynamic tools-count discovery (3 strategies, in priority order)

Replaces the hardcoded `expected_tools` per service. Calibration baseline (2026-07-18):

| Service | Count | Strategy |
|---|---|---|
| gmail | 19 | Strategy 1: regex `name:\s*"<tool>"` on `dist/index.js` |
| calendar | 5 | Strategy 1: same regex on `build/index.js` |
| google-drive | 20 | Strategy 2: count `.js` files in `dist/tools/` |
| google-sheets | 20 | Strategy 2: same count |

**Strategy 1:** read `<pkg>/dist/index.js` (or `build/index.js`) and count unique matches of `name:\s*"([a-z_][a-z_0-9]*)"`. Exclude noise tokens (`gmail`, `calendar`, `main`, `index`, `default`, `module`).

**Strategy 2:** count files in `<pkg>/dist/tools/*.js` excluding `.js.map`, `.d.ts`, and `index*`. Works for the workspace package's per-tool architecture.

**Strategy 3 (fallback):** keep the hardcoded `expected_tools` value when both strategies fail.

**Why drift is legitimate for Drive/Sheets:** the workspace package exposes 95 tools total across categories (`drive`, `sheets`, `docs`, `slides`, `shared-drives`, `labels`, `approvals`, `meta`). The wrapper selects one category via `GD_TOOLS=`. 20 observed = `GD_TOOLS=drive` (or `sheets`) only. If the user runs with `GD_TOOLS=drive,sheets`, observed count jumps to 40+ — script reports drift upward, not failure.

## Backoff on rate limits (added 2026-07-18)

In `run_with_retries()`, before `repair_from_error`, detect rate limit by message content (case-insensitive substring match on `429`, `rate limit`, `quota`, `user rate`). When detected, sleep `min(60, 2 ** attempt)` seconds — exponential 2s, 4s, 8s, 16s, 32s. Without this, the retry loop just re-plants on Google's quota and burns the 5-attempt budget in seconds.

## Mavis is OPTIONAL

Anton's machine has no Mavis CLI installed. Steps 9/10/11 of `agent.md` (configure + test Mavis) silently no-op when `~/.mavis/mcp/mcp.json` is absent. Do NOT recommend installing Mavis unless Anton asks explicitly. The architecture supports it but Anton chose Hermes-only.

## GOOGLE_DRIVE_ALLOW_WRITE

NEVER default to `true` in `.env`. The Drive/Sheets wrappers inject `$env:GOOGLE_DRIVE_ALLOW_WRITE = "true"` at run time inside the PowerShell wrapper (`run-google-drive-mcp.ps1` line ~`$env:GOOGLE_DRIVE_ALLOW_WRITE = "true"`), so the env is scoped per-process, not global. This is the right safety pattern: write capability on demand, off by default in `.env`.

## Pre-flight checks (`preflight_checks()` in `agent.py`)

Checks: Python 3.11+, Node, npm, npx, Hermes CLI, Mavis CLI (optional — skip if absent), PyYAML, requests, PowerShell 7, internet to `www.google.com:443`. Mavis missing is **not** a fatal preflight failure.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Tools exposes 0/19` despite valid install | Wrong path in `pkg_dir` candidates | Check `@user/.local/share/<service>-mcp-server/node_modules/<pkg>` exists |
| `Tools exposes: 20/25 (drift)` after npm upgrade | Package gained/lost tools | Recount via `python agent.py --check`, update `expected_tools` if drift > 20% |
| `hermes mcp test <name>` returns KO but creds are valid | Server crashed on OAuth refresh | Re-run `npx -y <pkg> auth` interactively |
| Port 3000 already in use | Another process holds the OAuth callback port | Kill the process or change the OAuth client redirect URI |
| Gmail/Calendar `Tools exposes 0/N` | Package version changed its bundle layout | Inspect `<pkg>/dist/` or `<pkg>/build/` for current file structure |

## What NOT to do

- Do NOT modify `credentials.json` directly — it contains the refresh_token and any edit invalidates the OAuth flow.
- Do NOT add Mavis-specific configuration unless Anton asks.
- Do NOT change `expected_tools` without re-running `python agent.py --check` first to confirm the drift is real.
- Do NOT delete `~/Downloads/MCP.bak.<timestamp>/` directories — they are the rollback path if a patch breaks the orchestrator.

## Versioning the orchestrator

The Python files (`agent.py`, `run_mcp_workflow.py`, `install-mcp-*.py`) have no version field. Track changes by:
1. Backup the directory: `cp -r ~/Downloads/MCP ~/Downloads/MCP.bak.<YYYYMMDD-HHMMSS>` before any patch.
2. Patch with `patch` tool (preferred) or `write_file` (full rewrite).
3. Run `python agent.py --check` as the smoke test — never skip.
4. If green, document the change in `agent.md` with a dated section header (pattern: `## <Topic> (<YYYY-MM-DD>)`).

## Quick verification checklist (run after every patch)

```bash
cd ~/Downloads/MCP
python agent.py --check
```

Expected: 4 services, each with `Tools exposes: N/N (conforme, methode: <strategy>)`. Any `drift` or `decouverte impossible` line is a signal to investigate before declaring the patch successful.

## Reference: 2026-07-18 audit fixes (do not re-apply blindly)

Four patches were applied during the audit session that produced this skill:

1. `STARTUP_QUESTION` in `agent.py` — old: `"Vous voulez-vous un MCP Calendar, Google Sheet, Drive..."` → new: `"Quel MCP Google voulez-vous installer ? (Gmail, Calendar, Drive, Sheets)"`.
2. Dynamic tools-count discovery — 3 strategies replacing hardcoded counts.
3. Exponential backoff in `run_with_retries()` — 2s to 32s on rate limit / 429 / quota.
4. `agent.md` updated to reflect Mavis as optional + new section explaining the dynamic counter.

The complete pre-patch state lives in `~/Downloads/MCP.bak.20260718-214244/`. If you see code that doesn't match this skill's description, that's why — and the backup is the rollback path.