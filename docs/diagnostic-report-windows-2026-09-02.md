# Diagnostic Report — Hermes Agent on Windows

**Environment:** Windows 11 Pro · Hermes Agent v0.21.0
**Period:** 2026-09-01 – 2026-09-02
**Scope:** defects observed in the project itself while installing, configuring, and running Hermes Agent (CLI, web dashboard, desktop app) on a Windows machine, plus the local fixes and workarounds applied.

---

## Identified Issues (Project Bugs)

### 1. Broken dashboard chat — "Chat connection interrupted (code 1006)"
**Severity: Critical** — Prevented web chat usage entirely.

- **Symptom:** The dashboard chat never came up; `gui.log` showed the terminal WebSocket (`/api/pty`) being accepted and killed every ~3 seconds in an infinite reconnection loop.
- **Root Cause:** Hermes spawns the chat TUI via **pywinpty**, which fails with `WinptyError: not a valid Win32 application` when the executable path contains **spaces** — and the Hermes-managed Node runtime lives under the user's profile directory (`C:\Users\<username with spaces>\AppData\Local\hermes\node\node.exe`). Proven by isolation: `cmd.exe` spawns fine, `node.exe` fails; the exact same `node.exe` copied to a path without spaces spawns successfully. The WebSocket handler then died without sending a close frame, which browsers surface as code 1006.
- **Project Bug:** Yes — Hermes neither guards against paths with spaces when spawning the PTY nor catches the pywinpty exception on that code path (the handler dies without a close frame instead of surfacing the error).

### 2. Auxiliary models configured with `provider: main` do not use the main provider
**Severity: High** — Crashed the chat on every message.

- **Symptom:** Recurring entries in `errors.log`: `Auxiliary: marking openrouter unhealthy for 60s (payment / credit error)`.
- **Root Cause:** The value `main` in the auxiliary slots (compression, title generation, session search, web extraction, vision) **does not** resolve to the configured primary provider — it resolves to `OPENAI_BASE_URL` + `OPENAI_API_KEY` from `.env`. When that legacy key has no credit, the failure falls back to OpenRouter and the session breaks. The `main` naming is misleading — an upstream design/documentation flaw.

### 3. Messaging gateway crashes on Windows — `asyncio.start_unix_server`
**Severity: Medium** — Affects only messaging integrations (Telegram/Discord, etc.), not the chat.

- **Symptom:** Gateway process crashing intermittently with `SystemExit: 75` and `AttributeError: asyncio.start_unix_server` in `shutdown_watchdog.py`.
- **Root Cause:** POSIX-specific code (Unix domain socket) executed without a Windows platform guard, where `asyncio.start_unix_server` does not exist. A portability bug.

### 4. Bundled runtime ships SQLite 3.45.1 (WAL-reset bug)
**Severity: Low** — Automatically mitigated.

- **Symptom:** Warning in the logs: the SQLite library linked into the venv's Python is affected by the WAL-reset corruption bug.
- **Root Cause:** The installer-bundled Python runtime ships SQLite 3.45.1. Hermes already mitigates this by enforcing `journal_mode=DELETE` (rollback journal) on its databases, so no corruption occurs — but the runtime remediation step of `hermes update` did not complete (see #7).

### 5. `terminal.backend` spontaneously changed to `ssh`
**Severity: Medium.**

- **Symptom:** The terminal backend flipped from `local` to `ssh` without user action.
- **Root Cause:** One of the frontends (desktop app/dashboard) rewrote `config.yaml` with its own defaults. The project lacks per-layer config isolation, allowing one interface to overwrite another's choices.

### 6. Setup wizard fails in non-TTY environments
**Severity: Low.**

- **Symptom:** The installer exited with code 1 and `NoConsoleScreenBufferError`.
- **Root Cause:** The interactive setup wizard assumes an interactive console (TTY prompts) and has no graceful degradation for non-interactive execution — breaking headless or scheduled sessions, a common Windows scenario.

### 7. `hermes update` completes partially when files are in use
**Severity: Low.**

- **Symptom:** The first update run ended "partially complete" without upgrading the venv.
- **Root Cause:** The running dashboard process held a lock on a `.pyd` file inside the venv. File locking is a Windows OS constraint, but the updater does not detect its own running processes and stop them (or refuse to proceed) before updating.

### 8. Session token rotates on every server restart
**Severity: Low** — Expected behavior with confusing UX.

- **Symptom:** After a dashboard restart, already-open browser tabs show connection errors and the log registers `pty auth rejected reason=token_mismatch` until the tab is manually refreshed (F5).
- **Root Cause:** A fresh token is minted per process launch and injected into the HTML — secure, but stale tabs get no automatic re-authentication path.

### 9. Minor observations
- `hermes dashboard --stop` prints a snippet of the CLI usage text even on successful execution (cosmetic).
- The v38 → v40 config migration warned that the `teams` and `google_chat` platforms reference unknown toolsets (`hermes-teams`, `hermes-google_chat`), leaving those platforms without tools until reconfigured via `hermes tools`.
- One isolated log entry: `event loop stalled 3167.5s (GIL pressure suspected)` on the web server — no recurrence observed; kept under monitoring.

---

## Applied Fixes & Mitigations

| # | Resolution | Status |
|---|------------|--------|
| 1 | Created a directory junction (e.g. `C:\hermes-node`) pointing to the Hermes-managed Node directory — a path without spaces — and set the persistent user environment variable `HERMES_NODE` to the `node.exe` inside the junction. Hermes prioritizes `HERMES_NODE` when building the chat command. Validated end-to-end: PTY connects and the TUI renders in the browser. | **Resolved** |
| 2 | Pointed all 5 `auxiliary` slots in `config.yaml` directly at the NVIDIA endpoint: `base_url: https://integrate.api.nvidia.com/v1` + `api_key: ${NVIDIA_API_KEY}` (an env-var reference resolved from `.env` — no key material in the file) + `nvidia/nemotron-3-super-120b-a12b`. No further OpenRouter errors logged; chat tested and responding. | **Resolved** |
| 3 | Gateway restarted by the update; the crash is intermittent and only affects messaging integrations. No local fix possible — needs an upstream platform guard in `shutdown_watchdog.py`. | **Mitigated** |
| 4 | Databases running in rollback-journal mode (Hermes' automatic mitigation). Definitive fix would be rebuilding the venv with a `uv`-managed Python carrying a patched SQLite — optional, no active risk. | **Mitigated** |
| 5 | Ran `hermes config set terminal.backend local` — verified in `config.yaml`. | **Resolved** |
| 6 | Configuration done without the wizard, via `hermes config set` and direct `config.yaml` edits. Standing rule: never run `hermes setup` in non-TTY sessions. | **Workaround** |
| 7 | Re-ran the update with dashboard/gateway/desktop stopped → completed (config migrated v38→v40, cua-driver 0.21.0→0.23.2). Adopted procedure: stop all Hermes processes before updating. | **Resolved** |
| 8 | Documented: refresh the tab (F5) after a server restart. | **Documented** |
| — | Additional: the monorepo's `node_modules` was corrupted by parallel npm builds during setup (EBUSY/ENOTEMPTY); recovered with `npm ci` and sequential builds. Desktop app and dashboard built and operational. | **Resolved** |

**Final state:** web dashboard serving with a fully functional chat (verified over WebSocket), model inference and tool calling operational, desktop application built.

---

## Suggested follow-ups for the project

Issues 1, 2, 3 and 5 describe upstream bugs that would be better tracked as individual issues with their log excerpts:

1. Guard the PTY spawn against paths containing spaces (or surface the pywinpty error cleanly instead of dropping the WebSocket without a close frame).
2. Make `provider: main` in auxiliary slots actually resolve to the configured primary provider, or rename/document it as "the `OPENAI_*` env credentials".
3. Add a Windows platform guard around the Unix-socket code in `shutdown_watchdog.py`.
4. Isolate per-frontend writes to `config.yaml` so one surface can't silently overwrite another's settings.
