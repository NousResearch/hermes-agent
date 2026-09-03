# Diagnostic Report — Hermes Agent

**Environment:** Windows 11 Pro · Hermes Agent v0.21.0
**Period:** 09/01–09/02/2026

---

## Identified Issues (Inherent Project Bugs)

### 1. Broken dashboard chat — "Chat connection interrupted (code 1006)"
**Severity: Critical** — Prevented web chat usage.

- **Symptom:** The dashboard chat (http://127.0.0.1:9119) would not open; `gui.log` showed the terminal WebSocket (`/api/pty`) being accepted and terminated every ~3 seconds in an infinite reconnection loop.
- **Root Cause:** Hermes spawns the chat TUI via **pywinpty**, which fails with `WinptyError: not a valid Win32 application` when the executable path contains **spaces** — and the managed Node runtime resides in `C:\Users\Thinkin pad 8g\...` (a username containing a space). Proven by isolation: `cmd.exe` spawns properly, `node.exe` fails; copying the exact same `node.exe` to a path without spaces succeeds. The handler crashed without sending a WebSocket close frame, which the browser reports as error code 1006.
- **Project Bug:** Yes — Hermes does not handle pywinpty exceptions along this code path (the process dies without a close frame) and does not account for the `path with space + pywinpty` combination.

### 2. Auxiliary models configured with `provider: main` do not use the main provider
**Severity: High** — Crashed the chat on every message.

- **Symptom:** Recurring errors in `errors.log`: `Auxiliary: marking openrouter unhealthy for 60s (payment / credit error)`.
- **Root Cause:** The `main` value in auxiliary slots (compression, title generation, session search, web extraction, vision) **does not** point to the configured main provider — it resolves to `OPENAI_BASE_URL` + `OPENAI_API_KEY` from `.env`. The legacy OpenAI key had exhausted credits; the failure triggered a fallback to OpenRouter (which also had no balance), breaking the session. The `main` naming is misleading — this is an upstream design/documentation flaw.

### 3. Messaging gateway crashes on Windows — `asyncio.start_unix_server`
**Severity: Medium** — Affects only messaging integrations (Telegram/Discord, etc.), not the chat.

- **Symptom:** Gateway process crashing intermittently with `SystemExit: 75` and `AttributeError: asyncio.start_unix_server` in `shutdown_watchdog.py`.
- **Root Cause:** POSIX-specific code (Unix domain socket) executed without a Windows platform guard, where `start_unix_server` does not exist. A portability bug within the project.

### 4. Bundled runtime includes SQLite 3.45.1 (WAL-reset bug)
**Severity: Low** — Automatically mitigated.

- **Symptom:** Warning logged across all log files: the SQLite library linked in the venv Python environment suffers from the WAL-reset database corruption bug.
- **Root Cause:** The Python runtime bundled with the installer ships with the vulnerable SQLite 3.45.1. Hermes mitigates this on its own by enforcing `journal_mode=DELETE` (rollback journal) across databases, preventing corruption — however, the runtime remediation step failed during update.

### 5. `terminal.backend` spontaneously changed to `ssh`
**Severity: Medium.**

- **Symptom:** The terminal backend switched from `local` to `ssh` without user intervention.
- **Root Cause:** One of the application layers (desktop app/dashboard) rewrote `config.yaml` with its own defaults. The project lacks layer-isolated configuration, allowing one frontend interface to overwrite choices made by another.

### 6. Setup wizard fails in non-TTY environments
**Severity: Low.**

- **Symptom:** The installer exited with code 1 and `NoConsoleScreenBufferError`.
- **Root Cause:** The interactive setup wizard requires an interactive console (TTY prompts) and lacks graceful degradation for non-interactive execution — breaking headless or scheduled Windows sessions.

### 7. `hermes update` incomplete when files are in use
**Severity: Low.**

- **Symptom:** The initial update ended "partially complete" without upgrading the venv.
- **Root Cause:** The running dashboard process held an exclusive lock on a `.pyd` file in the venv; the updater does not terminate active Hermes processes prior to updating (file locking is a Windows OS limitation, but the updater should detect running instances and prompt or handle it).

### 8. Session token rotates on every server restart
**Severity: Low** (Expected behavior, confusing UX).

- **Symptom:** After restarting the dashboard, existing open browser tabs report connection failures and logs register `pty auth rejected reason=token_mismatch` until the user manually refreshes (F5).
- **Root Cause:** A fresh token is generated on each process launch and injected into the HTML — secure, but lacking automatic re-authentication or rehydration in active tabs.

### 9. Minor Observations
- `hermes dashboard --stop` outputs a snippet of the CLI usage/help text even upon successful execution (cosmetic).
- Updating migrated configuration from v38 → v40 with warnings that `teams` and `google_chat` platforms reference missing toolsets (`hermes-teams`, `hermes-google_chat`) — remaining without tools until reconfigured via `hermes tools`.
- An isolated log entry of `event loop stalled 3167.5s (GIL pressure suspected)` on the web server — no recurrence observed; kept under observation.

---

## Applied Fixes & Mitigations

| # | Resolution | Status |
|---|------------|--------|
| 1 | Created directory junction `C:\hermes-node` → Hermes Node directory (path without spaces) + added persistent user environment variable `HERMES_NODE=C:\hermes-node\node.exe`, which Hermes prioritizes when building the chat command. Validado end-to-end: PTY connects and TUI renders in browser. | **Resolved** |
| 2 | Pointed all 5 `auxiliary` slots in `config.yaml` directly to NVIDIA: `base_url: https://integrate.api.nvidia.com/v1` + `api_key: ${NVIDIA_API_KEY}` + `nvidia/nemotron-3-super-120b-a12b`. No further OpenRouter errors logged; chat tested and returning responses successfully. | **Resolved** |
| 3 | Gateway restarted via update; crashes are intermittent and only impact messaging. No local fix possible — requires upstream fix (platform guard check in `shutdown_watchdog.py`). | **Mitigated** |
| 4 | Databases operating with rollback journal (Hermes automated mitigation). Pending: rebuild venv with Python managed via `uv` to replace SQLite — optional, no active risk. | **Mitigated** |
| 5 | Executed `hermes config set terminal.backend local` — verified in `config.yaml`. | **Resolved** |
| 6 | Completed configuration bypassing wizard: utilized `hermes config set` and direct edits to `config.yaml`. Permanent guideline: do not invoke `hermes setup` in non-TTY sessions. | **Workaround** |
| 7 | Re-executed update with dashboard/gateway/desktop stopped → succeeded (config migrated v38→v40, cua-driver updated 0.21.0→0.23.2). Adopted operational procedure: stop all processes before updating. | **Resolved** |
| 8 | Documented: refresh browser tab (F5) following server restart. | **Documented** |
| — | Additional: monorepo `node_modules` was corrupted due to parallel npm builds during layer installation (EBUSY/ENOTEMPTY); restored via `npm ci` and sequential builds. Desktop app and dashboard compiled and operational. | **Resolved** |

**Final State:** Dashboard running at http://127.0.0.1:9119 with fully functional chat (verified via WebSocket), model inference and tool calling operational, desktop application compiled.
'@

[System.IO.File]::WriteAllText("C:\Users\Thinkin pad 8g\Desktop\relatorio-hermes-agent.md", $content, [System.Text.Encoding]::UTF8)
`
Ran command: `$content = @'
# Diagnostic Report — Hermes Agent

**Environment:** Windows 11 Pro · Hermes Agent v0.21.0
**Period:** 09/01–09/02/2026

---

## Identified Issues (Inherent Project Bugs)

### 1. Broken dashboard chat — "Chat connection interrupted (code 1006)"
**Severity: Critical** — Prevented web chat usage.

- **Symptom:** The dashboard chat (http://127.0.0.1:9119) would not open; `gui.log` showed the terminal WebSocket (`/api/pty`) being accepted and terminated every ~3 seconds in an infinite reconnection loop.
- **Root Cause:** Hermes spawns the chat TUI via **pywinpty**, which fails with `WinptyError: not a valid Win32 application` when the executable path contains **spaces** — and the managed Node runtime resides in `C:\Users\Thinkin pad 8g\...` (a username containing a space). Proven by isolation: `cmd.exe` spawns properly, `node.exe` fails; copying the exact same `node.exe` to a path without spaces succeeds. The handler crashed without sending a WebSocket close frame, which the browser reports as error code 1006.
- **Project Bug:** Yes — Hermes does not handle pywinpty exceptions along this code path (the process dies without a close frame) and does not account for the `path with space + pywinpty` combination.

### 2. Auxiliary models configured with `provider: main` do not use the main provider
**Severity: High** — Crashed the chat on every message.

- **Symptom:** Recurring errors in `errors.log`: `Auxiliary: marking openrouter unhealthy for 60s (payment / credit error)`.
- **Root Cause:** The `main` value in auxiliary slots (compression, title generation, session search, web extraction, vision) **does not** point to the configured main provider — it resolves to `OPENAI_BASE_URL` + `OPENAI_API_KEY` from `.env`. The legacy OpenAI key had exhausted credits; the failure triggered a fallback to OpenRouter (which also had no balance), breaking the session. The `main` naming is misleading — this is an upstream design/documentation flaw.

### 3. Messaging gateway crashes on Windows — `asyncio.start_unix_server`
**Severity: Medium** — Affects only messaging integrations (Telegram/Discord, etc.), not the chat.

- **Symptom:** Gateway process crashing intermittently with `SystemExit: 75` and `AttributeError: asyncio.start_unix_server` in `shutdown_watchdog.py`.
- **Root Cause:** POSIX-specific code (Unix domain socket) executed without a Windows platform guard, where `start_unix_server` does not exist. A portability bug within the project.

### 4. Bundled runtime includes SQLite 3.45.1 (WAL-reset bug)
**Severity: Low** — Automatically mitigated.

- **Symptom:** Warning logged across all log files: the SQLite library linked in the venv Python environment suffers from the WAL-reset database corruption bug.
- **Root Cause:** The Python runtime bundled with the installer ships with the vulnerable SQLite 3.45.1. Hermes mitigates this on its own by enforcing `journal_mode=DELETE` (rollback journal) across databases, preventing corruption — however, the runtime remediation step failed during update.

### 5. `terminal.backend` spontaneously changed to `ssh`
**Severity: Medium.**

- **Symptom:** The terminal backend switched from `local` to `ssh` without user intervention.
- **Root Cause:** One of the application layers (desktop app/dashboard) rewrote `config.yaml` with its own defaults. The project lacks layer-isolated configuration, allowing one frontend interface to overwrite choices made by another.

### 6. Setup wizard fails in non-TTY environments
**Severity: Low.**

- **Symptom:** The installer exited with code 1 and `NoConsoleScreenBufferError`.
- **Root Cause:** The interactive setup wizard requires an interactive console (TTY prompts) and lacks graceful degradation for non-interactive execution — breaking headless or scheduled Windows sessions.

### 7. `hermes update` incomplete when files are in use
**Severity: Low.**

- **Symptom:** The initial update ended "partially complete" without upgrading the venv.
- **Root Cause:** The running dashboard process held an exclusive lock on a `.pyd` file in the venv; the updater does not terminate active Hermes processes prior to updating (file locking is a Windows OS limitation, but the updater should detect running instances and prompt or handle it).

### 8. Session token rotates on every server restart
**Severity: Low** (Expected behavior, confusing UX).

- **Symptom:** After restarting the dashboard, existing open browser tabs report connection failures and logs register `pty auth rejected reason=token_mismatch` until the user manually refreshes (F5).
- **Root Cause:** A fresh token is generated on each process launch and injected into the HTML — secure, but lacking automatic re-authentication or rehydration in active tabs.

### 9. Minor Observations
- `hermes dashboard --stop` outputs a snippet of the CLI usage/help text even upon successful execution (cosmetic).
- Updating migrated configuration from v38 → v40 with warnings that `teams` and `google_chat` platforms reference missing toolsets (`hermes-teams`, `hermes-google_chat`) — remaining without tools until reconfigured via `hermes tools`.
- An isolated log entry of `event loop stalled 3167.5s (GIL pressure suspected)` on the web server — no recurrence observed; kept under observation.

---

## Applied Fixes & Mitigations

| # | Resolution | Status |
|---|------------|--------|
| 1 | Created directory junction `C:\hermes-node` → Hermes Node directory (path without spaces) + added persistent user environment variable `HERMES_NODE=C:\hermes-node\node.exe`, which Hermes prioritizes when building the chat command. Validated end-to-end: PTY connects and TUI renders in browser. | **Resolved** |
| 2 | Pointed all 5 `auxiliary` slots in `config.yaml` directly to NVIDIA: `base_url: https://integrate.api.nvidia.com/v1` + `api_key: ${NVIDIA_API_KEY}` + `nvidia/nemotron-3-super-120b-a12b`. No further OpenRouter errors logged; chat tested and returning responses successfully. | **Resolved** |
| 3 | Gateway restarted via update; crashes are intermittent and only impact messaging. No local fix possible — requires upstream fix (platform guard check in `shutdown_watchdog.py`). | **Mitigated** |
| 4 | Databases operating with rollback journal (Hermes automated mitigation). Pending: rebuild venv with Python managed via `uv` to replace SQLite — optional, no active risk. | **Mitigated** |
| 5 | Executed `hermes config set terminal.backend local` — verified in `config.yaml`. | **Resolved** |
| 6 | Completed configuration bypassing wizard: utilized `hermes config set` and direct edits to `config.yaml`. Permanent guideline: do not invoke `hermes setup` in non-TTY sessions. | **Workaround** |
| 7 | Re-executed update with dashboard/gateway/desktop stopped → succeeded (config migrated v38→v40, cua-driver updated 0.21.0→0.23.2). Adopted operational procedure: stop all processes before updating. | **Resolved** |
| 8 | Documented: refresh browser tab (F5) following server restart. | **Documented** |
| — | Additional: monorepo `node_modules` was corrupted due to parallel npm builds during layer installation (EBUSY/ENOTEMPTY); restored via `npm ci` and sequential builds. Desktop app and dashboard compiled and operational. | **Resolved** |

**Final State:** Dashboard running at http://127.0.0.1:9119 with fully functional chat (verified via WebSocket), model inference and tool calling operational, desktop application compiled.
'@

[System.IO.File]::WriteAllText("C:\Users\Thinkin pad 8g\Desktop\relatorio-hermes-agent.md", $content, [System.Text.Encoding]::UTF8)
`
Ran command: `Get-Content -Path "C:\Users\Thinkin pad 8g\Desktop\relatorio-hermes-agent.md" -Head 15`

O arquivo [relatorio-hermes-agent.md](file:///C:/Users/Thinkin%20pad%208g/Desktop/relatorio-hermes-agent.md) foi traduzido e atualizado com sucesso para o inglês.

Abaixo está o conteúdo traduzido:

```markdown
# Diagnostic Report — Hermes Agent

**Environment:** Windows 11 Pro · Hermes Agent v0.21.0
**Period:** 09/01–09/02/2026

---

## Identified Issues (Inherent Project Bugs)

### 1. Broken dashboard chat — "Chat connection interrupted (code 1006)"
**Severity: Critical** — Prevented web chat usage.

- **Symptom:** The dashboard chat (http://127.0.0.1:9119) would not open; `gui.log` showed the terminal WebSocket (`/api/pty`) being accepted and terminated every ~3 seconds in an infinite reconnection loop.
- **Root Cause:** Hermes spawns the chat TUI via **pywinpty**, which fails with `WinptyError: not a valid Win32 application` when the executable path contains **spaces** — and the managed Node runtime resides in `C:\Users\Thinkin pad 8g\...` (a username containing a space). Proven by isolation: `cmd.exe` spawns properly, `node.exe` fails; copying the exact same `node.exe` to a path without spaces succeeds. The handler crashed without sending a WebSocket close frame, which the browser reports as error code 1006.
- **Project Bug:** Yes — Hermes does not handle pywinpty exceptions along this code path (the process dies without a close frame) and does not account for the `path with space + pywinpty` combination.

### 2. Auxiliary models configured with `provider: main` do not use the main provider
**Severity: High** — Crashed the chat on every message.

- **Symptom:** Recurring errors in `errors.log`: `Auxiliary: marking openrouter unhealthy for 60s (payment / credit error)`.
- **Root Cause:** The `main` value in auxiliary slots (compression, title generation, session search, web extraction, vision) **does not** point to the configured main provider — it resolves to `OPENAI_BASE_URL` + `OPENAI_API_KEY` from `.env`. The legacy OpenAI key had exhausted credits; the failure triggered a fallback to OpenRouter (which also had no balance), breaking the session. The `main` naming is misleading — this is an upstream design/documentation flaw.

### 3. Messaging gateway crashes on Windows — `asyncio.start_unix_server`
**Severity: Medium** — Affects only messaging integrations (Telegram/Discord, etc.), not the chat.

- **Symptom:** Gateway process crashing intermittently with `SystemExit: 75` and `AttributeError: asyncio.start_unix_server` in `shutdown_watchdog.py`.
- **Root Cause:** POSIX-specific code (Unix domain socket) executed without a Windows platform guard, where `start_unix_server` does not exist. A portability bug within the project.

### 4. Bundled runtime includes SQLite 3.45.1 (WAL-reset bug)
**Severity: Low** — Automatically mitigated.

- **Symptom:** Warning logged across all log files: the SQLite library linked in the venv Python environment suffers from the WAL-reset database corruption bug.
- **Root Cause:** The Python runtime bundled with the installer ships with the vulnerable SQLite 3.45.1. Hermes mitigates this on its own by enforcing `journal_mode=DELETE` (rollback journal) across databases, preventing corruption — however, the runtime remediation step failed during update.

### 5. `terminal.backend` spontaneously changed to `ssh`
**Severity: Medium.**

- **Symptom:** The terminal backend switched from `local` to `ssh` without user intervention.
- **Root Cause:** One of the application layers (desktop app/dashboard) rewrote `config.yaml` with its own defaults. The project lacks layer-isolated configuration, allowing one frontend interface to overwrite choices made by another.

### 6. Setup wizard fails in non-TTY environments
**Severity: Low.**

- **Symptom:** The installer exited with code 1 and `NoConsoleScreenBufferError`.
- **Root Cause:** The interactive setup wizard requires an interactive console (TTY prompts) and lacks graceful degradation for non-interactive execution — breaking headless or scheduled Windows sessions.

### 7. `hermes update` incomplete when files are in use
**Severity: Low.**

- **Symptom:** The initial update ended "partially complete" without upgrading the venv.
- **Root Cause:** The running dashboard process held an exclusive lock on a `.pyd` file in the venv; the updater does not terminate active Hermes processes prior to updating (file locking is a Windows OS limitation, but the updater should detect running instances and prompt or handle it).

### 8. Session token rotates on every server restart
**Severity: Low** (Expected behavior, confusing UX).

- **Symptom:** After restarting the dashboard, existing open browser tabs report connection failures and logs register `pty auth rejected reason=token_mismatch` until the user manually refreshes (F5).
- **Root Cause:** A fresh token is generated on each process launch and injected into the HTML — secure, but lacking automatic re-authentication or rehydration in active tabs.

### 9. Minor Observations
- `hermes dashboard --stop` outputs a snippet of the CLI usage/help text even upon successful execution (cosmetic).
- Updating migrated configuration from v38 → v40 with warnings that `teams` and `google_chat` platforms reference missing toolsets (`hermes-teams`, `hermes-google_chat`) — remaining without tools until reconfigured via `hermes tools`.
- An isolated log entry of `event loop stalled 3167.5s (GIL pressure suspected)` on the web server — no recurrence observed; kept under observation.

---

## Applied Fixes & Mitigations

| # | Resolution | Status |
|---|------------|--------|
| 1 | Created directory junction `C:\hermes-node` → Hermes Node directory (path without spaces) + added persistent user environment variable `HERMES_NODE=C:\hermes-node\node.exe`, which Hermes prioritizes when building the chat command. Validated end-to-end: PTY connects and TUI renders in browser. | **Resolved** |
| 2 | Pointed all 5 `auxiliary` slots in `config.yaml` directly to NVIDIA: `base_url: https://integrate.api.nvidia.com/v1` + `api_key: ${NVIDIA_API_KEY}` + `nvidia/nemotron-3-super-120b-a12b`. No further OpenRouter errors logged; chat tested and returning responses successfully. | **Resolved** |
| 3 | Gateway restarted via update; crashes are intermittent and only impact messaging. No local fix possible — requires upstream fix (platform guard check in `shutdown_watchdog.py`). | **Mitigated** |
| 4 | Databases operating with rollback journal (Hermes automated mitigation). Pending: rebuild venv with Python managed via `uv` to replace SQLite — optional, no active risk. | **Mitigated** |
| 5 | Executed `hermes config set terminal.backend local` — verified in `config.yaml`. | **Resolved** |
| 6 | Completed configuration bypassing wizard: utilized `hermes config set` and direct edits to `config.yaml`. Permanent guideline: do not invoke `hermes setup` in non-TTY sessions. | **Workaround** |
| 7 | Re-executed update with dashboard/gateway/desktop stopped → succeeded (config migrated v38→v40, cua-driver updated 0.21.0→0.23.2). Adopted operational procedure: stop all processes before updating. | **Resolved** |
| 8 | Documented: refresh browser tab (F5) following server restart. | **Documented** |
| — | Additional: monorepo `node_modules` was corrupted due to parallel npm builds during layer installation (EBUSY/ENOTEMPTY); restored via `npm ci` and sequential builds. Desktop app and dashboard compiled and operational. | **Resolved** |

**Final State:** Dashboard running at http://127.0.0.1:9119 with fully functional chat (verified via WebSocket), model inference and tool calling operational, desktop application compiled.
```