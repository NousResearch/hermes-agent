# AGENT_HANDOFF.md — Hermes Agent Runtime

## Project
- **Repo:** https://github.com/NousResearch/hermes-agent.git
- **Local:** ~/hermes-stack-work/hermes-agent
- **Branch:** `feat/runtime-run-api-contract`
- **Parent:** `main`

## Phase 0 — Preflight (completed)

### State at branch creation
- **Commit:** `30e947e0a05ef535e4b25a183d8bbe34fd68d1d5`
- **Message:** `feat(gateway): persist per-session /model overrides across gateway restarts`
- **Dirty files:** none (clean working tree)

### Architecture Summary

#### Existing `/v1/runs` API
The `/v1/runs` API **already exists** inside `gateway/platforms/api_server.py` in the `APIServerAdapter` class. It uses **aiohttp** (not FastAPI). Key endpoints:
- `POST /v1/runs` — Start a run (HTTP 202, returns `run_id`)
- `GET /v1/runs/{run_id}` — Poll run status
- `GET /v1/runs/{run_id}/events` — SSE stream of lifecycle events
- `POST /v1/runs/{run_id}/approval` — Resolve pending approval
- `POST /v1/runs/{run_id}/stop` — Interrupt a running agent

Run management is **embedded** in `APIServerAdapter` (no separate `RunManager` class). State is held in dicts: `_run_streams`, `_active_run_agents`, `_active_run_tasks`, `_run_statuses`, `_run_approval_sessions`.

SSE lifecycle events: `tool.started`, `tool.completed`, `reasoning.available`, `message.delta`, `approval.request`, `approval.responded`, `run.completed`, `run.failed`, `run.cancelled`.

#### AIAgent Class (`run_agent.py`)
- `AIAgent.__init__` accepts ~60 parameters. Thin shell — delegates to `agent/agent_init.py::init_agent()`.
- `AIAgent.chat(message)` — simple interface, returns `str`.
- `AIAgent.run_conversation(...)` — full interface, returns `dict` with `final_response` + `messages`.
- Delegates actual loop to `agent/conversation_loop.py`.

#### Two HTTP Servers
| Server | Framework | Port | File | Role |
|--------|-----------|------|------|------|
| API Server | aiohttp | 8642 | `gateway/platforms/api_server.py` | `/v1/runs`, `/v1/chat/completions`, etc. |
| Dashboard | FastAPI | 9119 | `hermes_cli/web_server.py` | Operator web UI, config, management |

#### Agent creation for API
`APIServerAdapter._create_agent()` at `gateway/platforms/api_server.py:1209` is the canonical example. It resolves runtime kwargs from gateway config via `gateway.run._resolve_runtime_agent_kwargs()`, applies per-client model routes, wires up callbacks (tool_progress, stream_delta, thinking, reasoning, clarify).

### Test Infrastructure
- **Runner:** `scripts/run_tests.sh` (enforces `TZ=UTC`, `LANG=C.UTF-8`, `PYTHONHASHSEED=0`, per-file subprocess isolation)
- **Relevant test files:**
  - `tests/gateway/test_api_server_runs.py` — dedicated `/v1/runs` tests (601 lines)
  - `tests/gateway/test_api_server.py` — general API server tests
  - `tests/run_agent/` — 128 test files for AIAgent
  - `tests/e2e/` — end-to-end tests
  - `tests/integration/` — integration tests
- **Command:** `scripts/run_tests.sh tests/gateway/test_api_server_runs.py -v`

### Key Files for Runtime API Work
| File | Purpose |
|------|---------|
| `gateway/platforms/api_server.py` | Primary `/v1/runs` implementation (4892 lines) |
| `run_agent.py` | AIAgent class shell (5978 lines) |
| `agent/agent_init.py` | Actual AIAgent init (1921 lines) |
| `agent/conversation_loop.py` | Actual run_conversation implementation |
| `gateway/run.py` | Gateway runner — orchestrates adapters (20025 lines) |
| `gateway/config.py` | Gateway config types |
| `model_tools.py` | Tool orchestration and dispatch |
| `toolsets.py` | Toolset definitions |
| `hermes_state.py` | SessionDB — SQLite session store |
| `hermes_constants.py` | `get_hermes_home()`, `display_hermes_home()` |

### Hard Rules (as specified)
1. Do not modify `../hermes-webui` unless explicitly instructed.
2. Do not create a second API server if one already exists.
3. Extend existing server/API architecture if present.
4. Agent runtime owns AIAgent execution.
5. WebUI should eventually call Agent runtime API instead of instantiating AIAgent directly.
6. Redact secrets from API responses, logs, event payloads, and tests.
7. Do not expose raw stack traces in public API responses.
8. Verify each phase before marking it complete.
9. Update this AGENT_HANDOFF.md after every completed phase.
10. Commit after every verified phase.

### Next Recommended Phase
**Phase 5 — WebUI agent-runs adapter:**
- WebUI adds `HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs` support
- Adapter calls Agent runtime API instead of instantiating `AIAgent` directly
- Integration tests between WebUI adapter and Agent `RunManager`

---

## Phase 4 — Hermes Agent /v1/runs Runtime API Foundation (completed)

### State Before Phase 4
- **Commit:** `40a255a`
- **Message:** `docs: Phase 0 preflight — create AGENT_HANDOFF.md with runtime/API architecture summary`

### What Was Built

Created a standalone `gateway/runtime/` package with the runtime API foundation:

#### Models (`gateway/runtime/models.py`)
- **`RuntimeEvent`** dataclass — structured event in a run's lifecycle
  - Fields: `event_id`, `seq`, `run_id`, `session_id`, `type`, `created_at`, `terminal`, `payload`
  - `event_id` format: `{run_id}:{seq}`
  - `to_dict(*, redact=True)` — serializes with optional secret redaction
- **`RuntimeStatus`** dataclass — pollable status for a run
  - Fields: `run_id`, `session_id`, `status`, `last_event_id`, `last_seq`, `terminal`, `controls`, `pending_approval_ids`, `pending_clarify_ids`, `error`, `result`, `created_at`, `updated_at`
  - `to_dict(*, redact=True)` — serializes with optional secret redaction
- **`redact_secrets(obj)`** — recursive secret redaction utility
  - Exact key-name match for: `api_key`, `apikey`, `token`, `access_token`, `refresh_token`, `password`, `secret`, `authorization`, `bearer`, `api-key`
  - Delegates to `agent.redact.redact_sensitive_text` for string values (prefix patterns, auth headers, JWTs, etc.)
- Supported statuses: `queued`, `running`, `awaiting_approval`, `awaiting_clarify`, `paused`, `cancelling`, `cancelled`, `failed`, `completed`, `expired`
- Supported event types: `run.started`, `run.status`, `token.delta`, `reasoning.delta`, `reasoning.done`, `progress`, `tool.started`, `tool.updated`, `tool.done`, `approval.requested`, `approval.resolved`, `clarify.requested`, `clarify.resolved`, `title.updated`, `usage.updated`, `usage.final`, `error`, `done`

#### Run Manager (`gateway/runtime/run_manager.py`)
- **`RunManager`** class with in-memory storage (thread-safe via `threading.Lock`)
- Public API:
  - `create_run(session_id, *, message, workspace, profile, model, toolsets, metadata)` — creates run + `run.started` event, returns `run_id`/`session_id`/`status`/`events_url`/`status_url`/`controls`
  - `get_status(run_id)` — returns `RuntimeStatus` dict or `None` for unknown
  - `append_event(run_id, event_type, *, session_id, payload)` — adds event, updates status
  - `read_events(run_id, *, after_seq, limit)` — returns `{"run_id": ..., "events": [...]}`
  - `stop_run(run_id)` — transitions to cancelling then cancelled; returns `not_found` for unknown
  - `transition_status(run_id, new_status)` — explicit status transition with event
  - `complete_run(run_id, *, result)` — terminal completed with result
  - `fail_run(run_id, *, error)` — terminal failed with error
  - `resolve_approval(run_id, choice)` — returns `not_supported` (deferred to integration phase)
  - `resolve_clarify(run_id, response)` — returns `not_supported` (deferred to integration phase)

### API Server Integration Status: **B** — Route module implemented, server mount deferred

The `RunManager` is a standalone service layer. The existing `/v1/runs` route handlers remain in `gateway/platforms/api_server.py` (aiohttp). Integration options:
1. Mount new route handlers that delegate to `RunManager` (best for clean separation)
2. Replace `api_server.py`'s embedded dicts with `RunManager` internally
3. Add a FastAPI router using `RunManager` for the dashboard/WebUI surface

### Files Created
| File | Purpose |
|------|---------|
| `gateway/runtime/__init__.py` | Package init, re-exports all public symbols |
| `gateway/runtime/models.py` | `RuntimeEvent`, `RuntimeStatus`, `redact_secrets`, status/event constants |
| `gateway/runtime/run_manager.py` | `RunManager` class — in-memory run lifecycle management |
| `tests/gateway/test_runtime_models.py` | 20 tests — model serialization, redaction, imports |
| `tests/gateway/test_runtime_run_manager.py` | 33 tests — lifecycle, events, stop, transitions, thread safety |
| `tests/gateway/test_runtime_routes.py` | 21 tests — API contract shapes, error handling, redaction |

### Verification

**Test run:**
```bash
uv run python -m pytest tests/gateway/test_runtime_models.py tests/gateway/test_runtime_run_manager.py tests/gateway/test_runtime_routes.py -v
```
Result: **74 passed, 0 failed** (0.63s)

**Smoke check:**
```bash
uv run python - <<'PY'
from gateway.runtime import RuntimeEvent, RuntimeStatus, RunManager
mgr = RunManager()
r = mgr.create_run("sess", message="hi")
# All operations verified: create, status, events, stop, approval, clarify
PY
```
Result: All operations correct. Redaction confirmed (api_key → <<redacted>>).

### Unsupported Items
- **Approval resolution** — returned as `not_supported`. The real approval mechanism (`tools.approval.resolve_gateway_approval`) requires a running gateway adapter context that isn't available in the standalone `RunManager`. Integration phase should wire this.
- **Clarify resolution** — returned as `not_supported`. Same reason: requires gateway adapter context.
- **True agent interruption** — `stop_run` transitions status directly to `cancelled` with synthetic events. Actual `agent.interrupt()` requires a live `AIAgent` reference.
- **HTTP route handlers** — not included in this package. The existing `api_server.py` route handlers (aiohttp) still manage their own state. Integration phase should either add new handlers delegating to `RunManager` or refactor the existing ones.

---

## Phase 9 — Full verification and final implementation report (completed)

### State Before Phase 9
- **Commit:** `f7cc6c5`
- **Message:** (prior phase commit)

### Verification Results

#### Agent — Focused verification
```
./scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py -v
Result: 74 passed, 0 failed in 0.7s — PASS
```

#### Agent — Full test suite
```
./scripts/run_tests.sh
Result: 70 passed in shard scope. 18 failures total across all files —
  all in pre-existing, unrelated areas:
  tests/acp/test_auth.py (2), tests/acp/test_edit_approval.py (1),
  tests/gateway/test_wecom_callback.py (3),
  tests/tools/test_execute_code_approval_cluster.py (7),
  tests/tools/test_modal_sandbox_fixes.py (2),
  tests/tools/test_voice_mode.py (3).
  Also 9 acp test files with collection/import errors.
  None related to Phase 4 runtime foundation.
```

#### Agent — Import/config smoke
```
python3 - import gateway.runtime.models, gateway.runtime.run_manager;
  RunManager smoke run created successfully
Result: All imports OK. Smoke run: run_id, session_id, status: queued,
  events_url/status_url/controls all populated. PASS
```

### Files Updated
- `AGENT_HANDOFF.md` — Phase 9 section added
- `IMPLEMENTATION_REPORT.md` — created with full implementation report

### Next task

**Mount `gateway/runtime` route module into live Agent API server, then run live WebUI agent-runs smoke.**

---

## Phase 10A — Mount gateway/runtime route module into live Agent API server (completed)

### State Before Phase 10A
- **Commit:** `035e333`
- **Message:** `Document Agent runtime API foundation verification`

### What Was Built

#### Route module (`gateway/runtime/routes.py`)
- Created `register_runtime_routes(app, *, run_manager, error_formatter)` function
- Registers 6 aiohttp handlers on a `web.Application`:
  - `POST /v1/runs` — create run (202), delegates to `RunManager.create_run()`
  - `GET /v1/runs/{run_id}` — poll status, delegates to `RunManager.get_status()`
  - `GET /v1/runs/{run_id}/events` — event replay (JSON or SSE), delegates to `RunManager.read_events()`
  - `POST /v1/runs/{run_id}/stop` — interrupt, delegates to `RunManager.stop_run()`
  - `POST /v1/runs/{run_id}/approval` — returns 501 not_supported, delegates to `RunManager.resolve_approval()`
  - `POST /v1/runs/{run_id}/clarify` — returns 501 not_supported, delegates to `RunManager.resolve_clarify()`
- Stores `RunManager` instance on `app["runtime_run_manager"]`
- Accepts optional `error_formatter` callback for consistent error envelope

#### Live server mount (`gateway/platforms/api_server.py`)
- Added import of `register_runtime_routes` from `gateway.runtime.routes`
- In `APIServerAdapter.connect()`, added conditional route registration:
  - When `HERMES_USE_RUNTIME_RUNS` env var or `platforms.api_server.extra.use_runtime_runs` config is truthy → uses runtime route module
  - Default: keeps legacy embedded handlers (zero behavior change)
- Runtime route module uses `_openai_error` as error formatter for API-consistent error responses

#### Package exports (`gateway/runtime/__init__.py`)
- Added `register_runtime_routes` to `__all__` and top-level imports

### Files Created
| File | Purpose |
|------|---------|
| `gateway/runtime/routes.py` | aiohttp route handlers delegating to RunManager |
| `tests/gateway/test_runtime_server_mount.py` | 31 tests — full HTTP lifecycle verification |

### Files Modified
| File | Change |
|------|--------|
| `gateway/runtime/__init__.py` | Added `register_runtime_routes` export |
| `gateway/platforms/api_server.py` | Added conditional runtime route mount in `connect()` |

### Verification

**Focused test run:**
```
./scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py -v
Result: 105 passed, 0 failed (4 files, 16 workers, 1.3s) — PASS
```

**Existing API server tests (regression check):**
```
./scripts/run_tests.sh tests/gateway/test_api_server.py -v
Result: 193 passed, 0 failed — PASS (no regressions)

./scripts/run_tests.sh tests/gateway/test_api_server_runs.py -v
Result: 23 passed, 0 failed — PASS (no regressions)
```

**Import/server smoke:**
```
python3 — gateway.runtime import, register_runtime_routes export,
  importable from gateway.platforms.api_server, RunManager functional
  smoke (create, status, events, stop, approval, clarify)
Result: All imports and smoke checks PASS
```

### Server Mount Details
- **Live server module:** `gateway.platforms.api_server` (APIServerAdapter class)
- **Route framework:** aiohttp `web.Application`
- **Mount mechanism:** Conditional — `HERMES_USE_RUNTIME_RUNS=true` env var enables runtime routes
- **Without flag:** Legacy embedded handlers registered (no behavior change)
- **With flag:** `register_runtime_routes(app, error_formatter=_openai_error)` replaces legacy handlers
- **RunManager storage:** `app["runtime_run_manager"]` accessible to middleware and tests

### Unsupported/Deferred
- **Approval/clarify resolution** — still returned as 501 not_supported (same as Phase 4)
- **True live agent execution** — `RunManager` manages state only; actual agent spawning deferred
- **Live server smoke with curl** — requires starting a blocking server; deferred to WebUI integration
- **HERMES_USE_RUNTIME_RUNS env var** — opt-in flag; production default keeps legacy behavior

### Next task
**Phase 10B — WebUI live agent-runs smoke**

---

## Phase 10B — WebUI live agent-runs smoke (completed)

### State Before Phase 10B
- **Commit:** `c53bcc8`
- **Message:** `feat(Phase 10A): mount runtime runs routes into Agent API server`

### Live Server Startup

Started standalone API server with runtime route module mounted (full `hermes gateway run` not viable for smoke due to messaging adapter startup dependencies):

```bash
cd hermes-agent
uv run python /tmp/hermes-agent-standalone.py
# Binds 127.0.0.1:8642 with HERMES_USE_RUNTIME_RUNS=1
# register_runtime_routes(app) delegates to RunManager
```

### Agent Direct Live Smoke Results

All 5 smoke steps passed:

| Step | Endpoint | Result |
|---|---|---|
| Create run | POST /v1/runs | 202, returns run_id, status "queued", events_url, status_url, controls |
| Get status | GET /v1/runs/{run_id} | RuntimeStatus shape: run_id, session_id, status, last_event_id, last_seq, terminal, controls, pending_approval_ids, pending_clarify_ids, error, result, created_at, updated_at |
| Get events | GET /v1/runs/{run_id}/events | run.started event with seq=1, payload contains message, workspace, profile, model, toolsets, metadata |
| Stop run | POST /v1/runs/{run_id}/stop | 200, status "cancelled", terminal true, controls cleared |
| No secrets | All responses | No API keys, tokens, or credentials leaked in any response |

### WebUI Agent-Runs Smoke Results (via Agent app["runtime_run_manager"])

| Step | WebUI Endpoint | Result |
|---|---|---|
| Runtime capabilities | GET /api/runtime/capabilities | runtime_adapter="agent-runs", resumable_events=true, last_event_id=true, cancel/approval/clarify supported |
| Mobile capabilities | GET /api/mobile/capabilities | deployment_health=true, workspace_search=true, resumable_runs=true, adapter="agent-runs" |
| Deployment health | GET /api/deployment/health | runtime_adapter="agent-runs", agent_runtime_reachable=false (standalone server lacks /v1/health endpoint; functional adapter works) |
| Run status | GET /api/runs/{run_id} | Correctly proxies to Agent /v1/runs/{run_id} via agent-runs adapter. Returns status "cancelled" for stopped run |
| Run events | GET /api/runs/{run_id}/events | Returns 3 events: run.started, run.status, done — matches Agent contract |
| Cancel proxy | POST /api/runs/{run_id}/cancel | 200, status "cancelled", no traceback, no secrets |
| Workspace search | GET /api/workspace/search | 200, empty results for test workspace, no error, no secret leakage |

### Post-Smoke Test Results

**Agent:**
```
uv run python -m pytest tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py -v
Result: 105 passed, 0 failed in 1.04s — PASS
```

**WebUI (agent-runs env):**
```
HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642 \
HERMES_WEBUI_AGENT_RUNS_API_KEY=test-key \
./scripts/test.sh tests/test_agent_runs_adapter.py \
  tests/test_runtime_adapter_selection.py \
  tests/test_agent_runs_error_mapping.py \
  tests/test_runtime_routes.py \
  tests/test_mobile_capabilities.py \
  tests/test_deployment_health.py \
  tests/test_workspace_search.py -v
Result: 149 passed, 8 failed in 5.98s — 8 expected failures in test_runtime_routes.py
  (tests designed for legacy-direct/journal mode; documented in Phase 5)
```

### Live Server Command Used

```bash
# Agent standalone server (pid in tmux hermes-agent-smoke):
cd hermes-agent && uv run python /tmp/hermes-agent-standalone.py
# Server: 127.0.0.1:8642, runtime routes mounted via register_runtime_routes(app)

# WebUI (via ctl.sh):
HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642 \
HERMES_WEBUI_AGENT_RUNS_API_KEY=test-key \
HERMES_WEBUI_PORT=8789 \
HERMES_WEBUI_PASSWORD=test-password \
./ctl.sh start
# WebUI: 127.0.0.1:8789 (port 8787 was occupied by unrelated process)
```

### Issues Found

1. **`agent_runtime_reachable: false` in deployment health** — The standalone server exposes `/health` but deployment health checks `/v1/health` which the standalone server doesn't provide. The full hermes gateway with API server would expose `/v1/health`. Not a functional bug — the agent-runs adapter (run status, events, cancel) works correctly.

2. **Port 8787 occupied** — Unrelated `command-center` dashboard process on port 8787. WebUI smoke used port 8789.

3. **`hermes gateway run` not practical for smoke** — The full gateway starts all messaging adapters (Telegram, etc.) which require credentials and block startup. Standalone Python server with route module was used instead.

### Next task
**PR review / harden approval-clarify live integration**

---

## Phase 11A — PR Review, Security Audit, and Merge-Readiness Package (completed)

### State Before Phase 11A
- **Commit:** `ab0fd67`
- **Message:** `Document live runtime runs smoke verification`

### Review Scope
Full branch diff (`feat/runtime-run-api-contract` vs `main`): 11 files, 2849 insertions, 6 deletions.

### Security Audit
- No API keys, tokens, passwords, or credentials present in any source file
- No hardcoded personal paths (test fixtures use generic `/home/user/workspace`)
- No hardcoded localhost:port outside of test fixtures
- Route mount correctly gated — `HERMES_USE_RUNTIME_RUNS` must be explicitly set
- Secret redaction covers 10 key-name variants + `agent.redact.redact_sensitive_text` fallback

### Bugs Found and Fixed

| # | File | Issue | Severity | Fix |
|---|------|-------|----------|-----|
| 1 | `gateway/runtime/run_manager.py:100-123` | TOCTOU race: two lock acquisitions allowed run deletion between session_id lookup and event append | MODERATE | Merged into single lock acquisition |
| 2 | `gateway/runtime/routes.py:95-102` | Array message parsing only handled `content` key, not OpenAI `{"type": "text", "text": "..."}` format | MODERATE | Extended to handle `type`-based text parts |

### Test Results

**Focused tests (post-fix):**
```
./scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py -v
Result: 105 passed, 0 failed (4 files, 16 workers, 0.9s) — PASS
```

**Import smoke (post-fix):**
```
RunManager.create_run(session_id="phase11a", message="review smoke")
Result: OK, status "queued", all fields populated — PASS
```

### Remaining Risks
1. approval/clarify resolution returns 501 `not_supported` — requires gateway adapter context for full integration
2. True live agent interruption not implemented — `stop_run` transitions status; `agent.interrupt()` needs live `AIAgent`
3. Bare `except Exception:` on JSON parsing follows existing `api_server.py` pattern, not regressive
4. `_redact_header_value` untouched — no caller in this codebase

### Files Modified in Phase 11A
- `gateway/runtime/run_manager.py` — TOCTOU fix
- `gateway/runtime/routes.py` — array message parsing fix

### Next task
**Phase 11C — True live AIAgent interruption and continuation, or PR submission if continuation remains out of scope.**

---

## Phase 11B — Approval/Clarify Lifecycle Integration (completed)

### State Before Phase 11B
- **Commit:** `5e34f8e`
- **Message:** `Phase 11A: PR review — fix TOCTOU race in append_event and extend array message parsing`

### What Was Done
Replaced the 501 `not_supported` approval/clarify stubs with a first-class pending action lifecycle in `RunManager`:
- `request_approval(run_id, approval_id, payload)` / `resolve_approval(run_id, approval_id, choice, payload)`
- `request_clarify(run_id, clarify_id, payload)` / `resolve_clarify(run_id, clarify_id, answer, payload)`
- Full error handling: not_found (404, run or action), conflict (409, terminal/duplicate)
- Secret redaction in all payloads (9 key-name variants)
- URL path run_id enforcement in routes
- Status transitions back to running when all pending IDs cleared

### Changed Files
- `gateway/runtime/run_manager.py` — full approval/clarify lifecycle methods
- `gateway/runtime/routes.py` — approval/clarify handlers with error mapping, URL path validation, body run_id rejection
- `tests/gateway/test_runtime_approval_clarify.py` — new (38 tests)
- `tests/gateway/test_runtime_run_manager.py` — updated
- `tests/gateway/test_runtime_routes.py` — updated
- `tests/gateway/test_runtime_server_mount.py` — updated

### Exact Tests
```
scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py \
  tests/gateway/test_runtime_approval_clarify.py
Result: 154 passed, 0 failed (5 files)
```

### Live Smoke Status
RunManager-level verified. HTTP endpoint smoke via `test_runtime_server_mount.py` integration tests. Full live AIAgent continuation deferred.

### Remaining Risks
1. True AIAgent continuation after approval/clarify — requires bridging session_key-based approval primitives to run_id-based runtime tracking
2. The gateway's `GatewayRunner` owns the only live `AIAgent` instances; the runtime `RunManager` is an isolated storage layer

### Next task
**Phase 12 — Full gateway runner integration for live agent continuation, or PR submission.**

---

## Phase 11C — True live AIAgent interruption and continuation bridge (completed)

### State Before Phase 11C
- **Commit:** `e258026`
- **Message:** `Phase 11B: Harden runtime approval and clarify lifecycle`

### What Was Built

Created a `RuntimeControlBridge` that bridges run_id-based runtime controls to session_key-based live gateway primitives:

#### Control Bridge (`gateway/runtime/control_bridge.py`)
- `RuntimeControlBridge(run_manager, *, get_session_key_for_run, gateway_runner_ref)` class
  - `resolve_approval(run_id, approval_id, choice, payload)` — updates RunManager, then calls `tools.approval.resolve_gateway_approval()` when session_key is known
  - `resolve_clarify(run_id, clarify_id, answer, payload)` — updates RunManager, then calls `tools.clarify_gateway.resolve_gateway_clarify()` (clarify_id is universally unique)
  - `stop_run(run_id)` — updates RunManager, then signals `AIAgent.interrupt("run_stop")` when live agent is reachable
  - `request_approval(run_id, approval_id, payload)` — pass-through to RunManager
  - `request_clarify(run_id, clarify_id, payload)` — pass-through to RunManager
  - `bind_run(run_id, session_key)` — establishes run_id → session_key mapping
- Fully optional: when no bridge is present, routes fall back to standalone RunManager (Phase 11B)
- All live resolution failures are caught and logged; bridge never fails when live primitives are unavailable

#### Route Integration (`gateway/runtime/routes.py`)
- Handlers dynamically look up bridge from `request.app["runtime_control_bridge"]`
- When bridge is present, stop/approval/clarify delegate through it
- When bridge is absent, handlers use RunManager directly (Phase 11B behavior preserved)

#### API Server Wiring (`gateway/platforms/api_server.py`)
- When runtime routes are enabled, creates a `RuntimeControlBridge` with `get_session_key_for_run` using RunManager's `session_id` field and `gateway_runner_ref` pointing to `_gateway_runner_ref` for live agent interrupt
- Bridge is stored on `app["runtime_control_bridge"]`

### Changed Files
- `gateway/runtime/control_bridge.py` — new file (222 lines)
- `gateway/runtime/routes.py` — bridge-aware control handlers
- `gateway/runtime/__init__.py` — added `RuntimeControlBridge` export
- `gateway/platforms/api_server.py` — bridge creation and wiring on route mount
- `tests/gateway/test_runtime_control_bridge.py` — new file (25 tests)
- `tests/gateway/test_runtime_server_mount.py` — new bridge mount tests (8 tests)

### Exact Tests
```
scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py \
  tests/gateway/test_runtime_approval_clarify.py \
  tests/gateway/test_runtime_control_bridge.py
Result: 188 passed, 0 failed (6 files)
```

Full gateway suite: 8845 passed, 3 failed (pre-existing `test_wecom_callback.py`, unrelated)

### What Is Complete vs Still Partially Deferred

**Complete:**
- `RuntimeControlBridge` exists and is wired into routes and API server
- Approval resolution bridges to `tools.approval.resolve_gateway_approval(session_key, choice)` when session_key is known
- Clarify resolution bridges to `tools.clarify_gateway.resolve_gateway_clarify(clarify_id, answer)` (clarify_id is universally unique, no mapping needed)
- Stop/cancel bridges to `AIAgent.interrupt("run_stop")` through GatewayRunner

**Still deferred (documented missing primitive):**
- Full run_id → session_key mapping at run creation time requires GatewayRunner to cooperate — the bridge's `get_session_key_for_run` callable is supplied by the API server using RunManager `session_id` as a proxy, which works for API-server-created runs but is not the same as a true gateway `session_key` (built from platform/chat_id/user_id via `build_session_key`)
- The bridge has the `bind_run(run_id, session_key)` method ready; a future gateway integration that calls it when spawning a runtime-tracked agent run would complete the mapping

### Live Smoke Status
Integration smoke test verifies: routes correctly delegate through bridge, bridge correctly delegates to RunManager, bridge handles errors gracefully. Full HTTP test with aiohttp TestClient passes all paths (create, status, stop via bridge, approval via bridge, clarify via bridge).

### Remaining Risks
1. `session_id` from RunManager is not identical to gateway `session_key` — but clarify resolution doesn't need it (universally unique ID), and approval resolution only needs it when both a live gateway approval queue and a matching session_key exist
2. GatewayRunner interaction is via module-level functions and weakref, not instance references — the weakref pattern is safe

### Next task
**Phase 12 — Full gateway runner integration for live agent continuation, or PR submission.**
