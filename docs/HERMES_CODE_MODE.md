# Hermes Code Mode

**Status:** active on `feature/hermes-code-mode`  
**Current state schema:** `v17` (`hermes_state.SCHEMA_VERSION = 17`)  
**Primary UI:** Hermes CLI home screen + HermesWeb Code Cockpit at `http://localhost:3001/code`  
**Backend API:** `http://localhost:9119`

Hermes Code Mode turns Hermes Agent into an AI development console. It keeps the normal conversational agent loop, tools, skills, memory, approvals, and terminal backends, but adds a coding-focused home screen, HermesWeb cockpit links, workspace/session awareness, REST-backed Code Mode services, and quick slash commands for day-to-day development work.

This document is the consolidated reference for the local `hermes-code` work. The historical phase reports remain in `docs/HERMES_CODE_MODE_PHASE_*.md` and `docs/HERMES_CODE_MODE_POST_PHASE_10_AUDIT.md`.

---

## 1. What Code Mode Adds

Code Mode adds four user-visible layers:

1. **CLI Code Console home screen**
   - Rendered by `build_hermes_code_console()` in `hermes_cli/banner.py`.
   - Used by the interactive CLI banner in `cli.py`.
   - Shows provider, model, profile, workspace, branch, session, backend status, web cockpit URL, DB schema, tool count, skill count, active code sessions, and pending approvals.

2. **Code Mode slash commands**
   - Registered in `hermes_cli/commands.py` under the `Code Mode` category.
   - Handled by `HermesCLI` in `cli.py`.
   - Commands: `/code`, `/web`, `/workspace`, `/session`, `/approvals`, `/skills-code`.

3. **HermesWeb Code Cockpit**
   - Main route: `http://localhost:3001/code`.
   - Uses the backend API on `http://localhost:9119`.
   - Current frontend behavior is REST polling. Backend WebSocket support exists, but the frontend WebSocket client is not the primary integration path yet.

4. **Backend Code services**
   - Workspaces, sessions, command execution, Git status/diff, model routing, diagnostics/LSP, multi-agent coding flows, coding skills, approvals, and provider selection.
   - Persisted in Hermes state storage with schema version `v17`.

---

## 2. Quick Start

### Start the CLI

```bash
source venv/bin/activate
hermes
```

On startup, the interactive CLI displays the Hermes Code Console banner. The banner is safe when the backend is offline; it falls back to local state and static defaults.

### Open the Code Cockpit

```text
http://localhost:3001/code
```

Use `/web` in the CLI to print the expected local URLs and log paths.

### Typical local services

| Service | URL / path | Purpose |
|---------|------------|---------|
| HermesWeb frontend | `http://localhost:3001` | Web UI shell |
| Code Cockpit | `http://localhost:3001/code` | Coding dashboard |
| Backend API | `http://localhost:9119` | REST API for HermesWeb and Code Mode |
| Backend status | `http://localhost:9119/api/status` | Health/status endpoint |
| Backend log | `/tmp/hermes-backend.log` | Local backend logs |
| Frontend log | `/tmp/hermes-frontend.log` | Local frontend logs |

Actual service startup can vary by local scripts/configuration. Check the branch-specific scripts or process manager before assuming both services are already running.

---

## 3. CLI Home Screen

The Code Mode home screen intentionally replaces the old huge tool/skill lists with a compact development dashboard.

### Full-width layout

When the terminal is wide enough, the banner shows:

- Hermes Code ASCII logo and Code Mode title.
- Version label rebranded from `Hermes Agent ...` to `Hermes Agent Code ...`.
- Provider and model.
- Active profile.
- Workspace path from `TERMINAL_CWD` or `os.getcwd()`.
- Current Git branch when the workspace is a Git repository.
- Session ID, or `new` if a fresh session.
- Backend status (`online :9119` or `offline :9119`).
- Web Cockpit URL.
- DB schema label (`v17` locally on this branch).
- Context usage when available.
- Active code session count and pending approval count when non-zero.
- Quick Actions for Code Mode slash commands.
- Tool and skill counts instead of the full verbose lists.

### Narrow terminal layout

For narrower terminals, the banner collapses to a line-oriented compact view while retaining:

- `HERMES CODE · AI Development Console`
- version label
- provider/model/profile
- workspace and branch
- backend/web/DB status
- quick command list
- tool/skill counts

### Safe fallback behavior

The home screen must never block CLI startup. It uses short timeouts and guarded fallbacks:

| Data | Primary source | Fallback |
|------|----------------|----------|
| Workspace | `TERMINAL_CWD`, then `os.getcwd()` | current process directory |
| Branch | `git rev-parse --abbrev-ref HEAD` | `not a git repo` |
| Backend status | `GET /api/status` | `offline` |
| DB schema | `~/.hermes/state.db` schema row | `hermes_state.SCHEMA_VERSION` |
| Code sessions | `GET /api/code/sessions` | `0` |
| Approvals | `GET /api/approvals` | `0` |
| Provider | explicit CLI provider or inferred from model | `unknown` / `local` |
| Model | current CLI model | `unknown` |

---

## 4. Code Mode Slash Commands

These commands are CLI-only helpers for development workflows. They do not replace the agent; they expose the current Code Mode state quickly.

| Command | Purpose | Behavior |
|---------|---------|----------|
| `/code` | Show Code Mode help | Prints cockpit URL, backend URL, and useful related commands. |
| `/web` | Show local web/backend URLs | Prints frontend, Code Cockpit, backend, status, agents, sessions, and log paths. |
| `/workspace` | Show current workspace | Uses `TERMINAL_CWD`/`os.getcwd()` and Git branch detection. Does not crash outside Git repos. |
| `/session` | Show active code sessions | Calls `GET /api/code/sessions`; accepts both list and `{sessions,total}` response envelopes. Backend offline/auth-required states degrade gracefully. |
| `/approvals` | Show pending approvals | Calls `GET /api/approvals`; accepts both list and `{approvals,total}` envelopes. Backend offline/auth-required states degrade gracefully. |
| `/skills-code` | List coding skills | Shows the seven built-in coding workflow shortcuts. Alias: `/skillscode`. |

### `/skills-code` list

| Skill | Intended use |
|-------|--------------|
| `fix_build` | Diagnose and fix build failures. |
| `review_diff` | Review a Git diff or code change. |
| `stabilize_hanging_task` | Recover from a hanging or stuck agent task. |
| `fix_runtime_error` | Diagnose runtime exceptions and propose/fix root causes. |
| `implement_feature` | Scaffold and implement a focused feature. |
| `refactor_react_page` | Refactor a React page/component while preserving behavior. |
| `benchmark_provider` | Compare model providers for speed/cost/quality. |

---

## 5. Architecture Overview

```text
Hermes CLI
  ├─ Code Console banner (`hermes_cli/banner.py`)
  ├─ Code Mode slash commands (`cli.py`, `hermes_cli/commands.py`)
  ├─ normal AIAgent loop (`run_agent.py`)
  ├─ tools / skills / approvals
  └─ optional HermesWeb/backend status polling

HermesWeb (`web/`)
  └─ Code Cockpit route `/code`
       ├─ REST API client
       ├─ workspaces and sessions
       ├─ command output and artifacts
       ├─ Git status/diff panels
       ├─ diagnostics/LSP views
       ├─ provider/model controls
       ├─ multi-agent flows
       ├─ coding skills
       └─ approvals UI

Backend (`hermes_cli/web_server.py`)
  ├─ `/api/code/*` REST endpoints
  ├─ `/api/providers*` endpoints
  ├─ `/api/approvals*` endpoints
  ├─ `/ws` realtime hub support
  └─ Hermes state DB / services
```

### State and persistence

Code Mode persists data through Hermes state services and SQLite migrations. Current local schema is `v17`. The schema includes Code Mode entities such as workspaces, sessions, commands, Git snapshots, provider routing/cost data, diagnostics, agent flows, and skill runs.

### Approvals

Dangerous or sensitive operations still go through Hermes approval infrastructure. Code Mode surfaces pending approvals both in the CLI (`/approvals`) and in HermesWeb.

### WebSocket vs polling

The backend has `/ws` support and emits realtime events such as session updates, command events, and artifact events. The current HermesWeb Code Cockpit integration is considered valid with REST polling; a dedicated frontend WebSocket client remains a future optimization.

---

## 6. Backend Endpoint Summary

The following endpoints are present in `hermes_cli/web_server.py` on this branch.

### Workspaces and sessions

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/workspaces` |
| `POST` | `/api/code/workspaces/open` |
| `GET` | `/api/code/workspaces/{workspace_id}` |
| `POST` | `/api/code/workspaces/{workspace_id}/refresh` |
| `GET` | `/api/code/sessions` |
| `POST` | `/api/code/sessions` |
| `GET` | `/api/code/sessions/{code_session_id}` |
| `PATCH` | `/api/code/sessions/{code_session_id}` |
| `POST` | `/api/code/sessions/{code_session_id}/cancel` |
| `POST` | `/api/code/sessions/{code_session_id}/complete` |
| `GET` | `/api/code/sessions/{code_session_id}/events` |
| `POST` | `/api/code/sessions/{code_session_id}/events` |
| `GET` | `/api/code/sessions/{code_session_id}/artifacts` |
| `POST` | `/api/code/sessions/{code_session_id}/artifacts/{artifact_id}/link` |

### Commands

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/sessions/{code_session_id}/commands` |
| `POST` | `/api/code/sessions/{code_session_id}/commands/run` |
| `GET` | `/api/code/commands/{command_id}` |
| `POST` | `/api/code/commands/{command_id}/cancel` |

### Git service

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/workspaces/{workspace_id}/git/status` |
| `GET` | `/api/code/workspaces/{workspace_id}/git/diff` |
| `GET` | `/api/code/workspaces/{workspace_id}/git/branch` |
| `GET` | `/api/code/workspaces/{workspace_id}/git/remote` |
| `POST` | `/api/code/workspaces/{workspace_id}/git/snapshot` |
| `POST` | `/api/code/workspaces/{workspace_id}/git/branch/prepare` |
| `POST` | `/api/code/workspaces/{workspace_id}/git/branch` |
| `POST` | `/api/code/workspaces/{workspace_id}/git/commit/prepare` |

### Provider/model routing and cost

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/providers` |
| `POST` | `/api/providers/select` |
| `POST` | `/api/providers/test` |
| `POST` | `/api/providers/add` |
| `GET` | `/api/code/sessions/{code_session_id}/model` |
| `POST` | `/api/code/sessions/{code_session_id}/model/select` |
| `PUT` | `/api/code/sessions/{code_session_id}/model` |
| `GET` | `/api/code/sessions/{code_session_id}/presets` |
| `POST` | `/api/code/sessions/{code_session_id}/presets` |
| `DELETE` | `/api/code/sessions/{code_session_id}/presets/{preset_id}` |
| `POST` | `/api/code/sessions/{code_session_id}/cost` |
| `GET` | `/api/code/sessions/{code_session_id}/cost` |
| `GET` | `/api/code/sessions/{code_session_id}/cost/entries` |

### Diagnostics and language services

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/workspaces/{workspace_id}/diagnostics` |
| `GET` | `/api/code/workspaces/{workspace_id}/diagnostics/file` |
| `GET` | `/api/code/workspaces/{workspace_id}/languages` |
| `POST` | `/api/code/workspaces/{workspace_id}/lsp/restart` |

### Multi-agent coding flows

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/agent-flows` |
| `POST` | `/api/code/agent-flows` |
| `GET` | `/api/code/agent-flows/{flow_id}` |
| `POST` | `/api/code/agent-flows/{flow_id}/run` |
| `POST` | `/api/code/agent-flows/{flow_id}/cancel` |
| `POST` | `/api/code/agent-flows/{flow_id}/resume` |
| `GET` | `/api/code/sessions/{code_session_id}/agent-flows` |

### Coding skills

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/code/skills` |
| `GET` | `/api/code/skill-runs` |
| `POST` | `/api/code/skill-runs` |
| `GET` | `/api/code/skill-runs/{run_id}` |
| `POST` | `/api/code/skill-runs/{run_id}/run` |
| `POST` | `/api/code/skill-runs/{run_id}/cancel` |
| `POST` | `/api/code/skill-runs/{run_id}/resume` |
| `GET` | `/api/code/sessions/{code_session_id}/skill-runs` |
| `POST` | `/api/code/skills/{skill_name}/run` |

### Approvals and realtime

| Method | Endpoint |
|--------|----------|
| `GET` | `/api/approvals` |
| `POST` | `/api/approvals/{approval_id}/approve` |
| `POST` | `/api/approvals/{approval_id}/reject` |
| `WS` | `/ws` |

---

## 7. Verification Commands

Activate the virtual environment before running Python commands:

```bash
source venv/bin/activate
```

Targeted Code Mode tests:

```bash
python3 -m pytest \
  tests/hermes_cli/test_banner.py \
  tests/hermes_cli/test_code_mode_commands.py \
  --override-ini="addopts=" --tb=short
```

Broader backend Code Mode regression set:

```bash
python3 -m pytest \
  tests/hermes_cli/test_coding_skills.py \
  tests/hermes_cli/test_multi_agent_coding.py \
  tests/hermes_cli/test_lsp_service.py \
  tests/hermes_cli/test_provider_router.py \
  tests/hermes_cli/test_git_service.py \
  tests/hermes_cli/test_command_runner.py \
  tests/hermes_cli/test_code_session_service.py \
  tests/hermes_cli/test_workspace_service.py \
  tests/test_artifacts.py \
  tests/test_hermes_state.py \
  --override-ini="addopts=" --tb=short
```

Website docs validation, when Node dependencies are already installed:

```bash
cd website
npm run build
```

Do not install dependencies just to validate docs unless the task explicitly allows it.

---

## 8. Known Limitations

- **Frontend WebSocket client:** backend `/ws` exists, but the Code Cockpit currently works through REST polling. This is acceptable for daily use; WebSocket integration is an optimization.
- **E2E validation:** full browser/backend validation requires the backend, frontend, auth/config, and local state to be running together.
- **Auth-required endpoints:** `/session` and `/approvals` intentionally treat `401`/`403` as graceful empty/auth-required fallbacks in the CLI.
- **Backend offline:** the CLI home screen still renders when `localhost:9119` is unavailable.
- **Existing lint caveats:** previous post-phase audit documented frontend lint issues in legacy pages. Treat those as separate from Code Mode docs unless re-audited.

---

## 9. Maintenance Notes

When changing Code Mode, update all of the following together:

1. `hermes_cli/commands.py` if slash commands are added/renamed.
2. `cli.py` command handlers and output examples.
3. `hermes_cli/banner.py` home screen fields and fallbacks.
4. `hermes_cli/skin_engine.py` if semantic colors change.
5. `docs/HERMES_CODE_MODE.md` and public website docs.
6. `website/docs/reference/slash-commands.md` for user-visible command changes.
7. Tests in `tests/hermes_cli/test_banner.py` and `tests/hermes_cli/test_code_mode_commands.py`.

Keep the CLI safe under degraded conditions: no backend, no Git repository, missing state DB, narrow terminal, or missing optional packages must not prevent `hermes` from starting.
