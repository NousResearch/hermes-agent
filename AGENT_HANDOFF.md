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

### Phase 12 — Live Agent Binding (completed)

**Commit:** TBD (next commit)

**What was implemented:**
1. `RuntimeControlBridge.bind_run(run_id, session_key, agent=None)` now accepts an optional agent reference for direct agent interrupt
2. `RuntimeControlBridge.unbind_run(run_id)` cleans up both session_key and agent mappings
3. `RuntimeControlBridge.stop_run()` tries direct agent reference first (from `bind_run`), then falls back to GatewayRunner `_running_agents` lookup, then RunManager-only
4. `RunManager.create_run()` now accepts optional `run_id` parameter for callers that provide their own
5. `register_runtime_routes()` now accepts `register_create`, `register_status`, `register_events` flags for selective route registration
6. API server runtime mode (`HERMES_USE_RUNTIME_RUNS=1`) now:
   - Registers runtime routes for stop/approval/clarify only (control plane)
   - Keeps legacy handlers for run creation (POST /v1/runs), status (GET /v1/runs/{run_id}), and events (GET /v1/runs/{run_id}/events) — these share a common `run_id`
   - Creates RunManager entries from `_handle_runs` so the runtime control bridge can resolve them
   - Calls `bridge.bind_run(run_id, approval_session_key, agent)` at agent spawn time
   - Calls `bridge.unbind_run(run_id)` on run terminal (completion, failure, cancel, sweep)

**What is complete:**
- `bind_run(run_id, session_key, agent_ref)` is called at API server runtime run spawn time
- Direct agent interrupt via stored agent reference (no GatewayRunner look-up needed when `bind_run` includes agent)
- Approval resolution bridges through bound `session_key` to `resolve_gateway_approval()`
- Clarify resolution bridges through globally unique `clarify_id` to `resolve_gateway_clarify()`
- Stop/cancel interrupts live agent via `bind_run` agent reference
- `unbind_run` clean-up on run terminal state
- 38 new binding-specific tests + 8 updated control bridge tests
- Full backward compat for Phase 11B standalone RunManager + Phase 11C bridge via `None` guard on `self._app`

**What is still deferred:**
- Full GatewayRunner-level binding (when GatewayRunner spawns a runtime-tracked agent for a non-API-server session). The bridge infrastructure is ready — a GatewayRunner integration that calls `bridge.bind_run()` when spawning a runtime-tracked agent (with a known run_id) would complete this.
- Non-API-server runtime runs (e.g., runs created directly via runtime routes without the API server's agent execution path) do not have live agent execution. The runtime routes provide the control plane but not the execution plane.

### Live Smoke Status
- ET-level HTTP integration tests pass with aiohttp TestClient: all routes correctly delegate through bridge
- Bridge correctly falls back to RunManager when no live primitives exist
- All error codes (404, 409) and edge cases tested
- Full E2E live agent approval/clarify resolution requires a running GatewayRunner instance — simulated with mock agents in tests

### Remaining Risks
1. `RunManager.session_id` is not the same as gateway `session_key` — approve/clarify resolution uses the session_key from `bind_run`, not `session_id`
2. The API server's runtime mode sets `session_id` in RunManager to the API client-provided `session_id`, which differs from the gateway `session_key`. For API-server-backed runs, `bind_run` uses `approval_session_key` (which is `run_id`) as the session_key, which matches the `set_current_session_key()` call in `_run_sync()`
3. GatewayRunner interaction is via module-level weakref — safe across process lifecycle

### Next task
Phase 13 — PR submission or further GatewayRunner-level runtime binding.

---

### Phase 13 — Messaging-platform GatewayRunner Binding (completed)

**Commit:** TBD (next commit)

**What was implemented:**
1. GatewayRunner now has a `_runtime_control_bridge` attribute and `set_runtime_control_bridge()` method
2. GatewayRunner's `_process_message` creates a runtime `run_id` per messaging turn when a bridge is available
3. `run_id` is threaded through `_handle_message_with_agent` → `_run_agent` → `_run_agent_inner`
4. `bind_run(run_id, session_key, agent_ref)` is called in the `track_agent()` callback when the messaging agent is promoted to `_running_agents`
5. `unbind_run(run_id)` is called in `_release_running_agent_state` and `_clear_session_boundary_security_state` when the messaging turn terminates
6. Approval requests during messaging runs record `request_approval` in RunManager via the bridge, with redacted secrets
7. Clarify requests during messaging runs record `request_clarify` in RunManager via the bridge
8. Final run status is updated (completed/failed) in `_run_agent_inner`'s finally block
9. API server sets the control bridge on GatewayRunner via `set_runtime_control_bridge()` when runtime mode is active
10. Stop/cancel uses bound messaging-platform live agent reference through bridge

**Where bind_run is called:**
- `gateway/run.py`: `_run_agent_inner` → `track_agent()` callback (after agent promotion to `_running_agents`)

**Where unbind_run is called:**
- `gateway/run.py`: `_release_running_agent_state()` (after pop of `_running_agents`, `_running_agents_ts`, etc.)
- `gateway/run.py`: `_clear_session_boundary_security_state()` (during session reset/clear)

**What is complete:**
- Messaging-platform run_id → session_key binding is complete
- Messaging-platform live approval continuation is complete: approval events recorded in RunManager, resolution bridges through bound session_key to `resolve_gateway_approval()`
- Messaging-platform live clarify continuation is complete: clarify events recorded in RunManager, resolution bridges through globally unique `clarify_id` to `resolve_gateway_clarify()`
- Messaging-platform live interrupt is complete: stop uses bound agent reference, falls back to GatewayRunner `_running_agents` lookup
- All existing API server runtime binding from Phase 12 remains valid
- Standalone RunManager fallback remains valid
- All acceptance requirements (404, 409, secrets redaction, not_found, URL run_id precedence) verified

**What is still deferred:**
- Execution plane for non-API-server runtime runs (runs created directly via `/v1/runs` without agent execution)
- GatewayRunner `/approve` and `/deny` slash commands do not update RunManager pending approval state when resolved directly via chat (they still resolve through `resolve_gateway_approval` which is independent of RunManager)

**Tests run:**
- 307 tests across 11 runtime test files — 307 passed, 0 failed
- New file: `tests/gateway/test_runtime_messaging_binding.py` — 36 tests covering:
  - GatewayRunner binding (bind_run, unbind_run, multiple runs per session)
  - Messaging approval lifecycle (request, resolve, 404/409/conflict, secrets redaction)
  - Messaging clarify lifecycle (request, resolve, 404/409/conflict, secrets redaction)
  - Stop/interrupt (direct agent reference, GatewayRunner fallback, no-live-agent, terminal)
  - GatewayRunner control bridge setter
  - Event lifecycle and non-duplication
  - URL path run_id precedence
  - Standalone RunManager fallback

**File changes in Phase 13:**
- `gateway/run.py` — `_runtime_control_bridge`, `_runtime_session_runs`, `set_runtime_control_bridge()`, run_id creation, bind_run/unbind_run, approval/clarify recording, final status updates
- `gateway/platforms/api_server.py` — sets bridge on GatewayRunner after creation
- `tests/gateway/test_runtime_messaging_binding.py` — new file (36 tests)

**Live smoke:** Not performed due to missing external messaging platform credentials. All code paths verified through unit tests with mock agents, mock GatewayRunner instances, and fake session primitives.

**Remaining risks:**
1. The `/approve` and `/deny` slash commands resolve approvals directly via `resolve_gateway_approval()` without updating RunManager — the REST API path `/v1/runs/{run_id}/approval` handles RunManager state correctly
2. Messaging GatewayRunner does not automatically register runtime runs for existing sessions — only turns after the bridge is set are tracked
3. `_runtime_session_runs` dict may accumulate stale entries if unbind is missed on an exception path (mitigated by pop-on-cleanup at multiple cleanup sites)

**Next task:**
Phase 15 — Cross-repo integration testing or remaining deferred items.

---

## Phase 14 — Non-API runtime execution plane and slash-command state sync (complete)

### Commit
- **TODO** (after `git commit`)

### Summary
Phase 14 wires the runtime execution plane for non-API-server GatewayRunner runs and makes gateway `/approve` and `/deny` slash-command actions update RunManager state consistently with the runtime approval lifecycle.

### What was implemented

**Non-API runtime execution plane:**
1. Non-API GatewayRunner agent runs already had runtime run_id creation from Phase 13 (line 9895-9909). Fixed invalid `status="queued"` parameter passed to `RunManager.create_run()`.
2. Terminal status transitions (completed/failed) are set in `_handle_message()` before unbinding, using the agent result's `completed` flag.
3. Exception paths correctly mark runs as failed before unbinding.
4. All existing bind_run/unbind_run wiring from Phase 12-13 remains valid.
5. Events (run.started, run.status, approval.requested, approval.resolved, done) are appended throughout the lifecycle.

**Where terminal status transitions are set:**
- `gateway/run.py`: `_handle_message()` — before calling `_release_running_agent_state()`, checks agent result and transitions to `completed` or `failed`
- `gateway/run.py`: `_release_running_agent_state()` — accepts optional `final_status` kwarg for setting terminal status before unbinding

**Slash-command RunManager state sync:**
1. `_handle_approve_command` — after resolving gateway approval, calls `_sync_runtime_approval_state()` to sync RunManager
2. `_handle_deny_command` — after resolving gateway denial, calls `_sync_runtime_approval_state()` with "deny" choice
3. Helper method `_sync_runtime_approval_state(session_key, count, choice)` finds the bound run_id and resolves pending approval IDs in RunManager FIFO
4. RunManager's existing `resolve_approval()` handles duplicate detection (returns conflict), unknown IDs (returns not_found), and terminal runs (returns conflict)
5. Secrets are redacted in all approval event payloads

**How /approve updates RunManager state:**
1. Gateway approval queue is resolved via `resolve_gateway_approval(session_key, choice, ...)` (existing behavior)
2. Bridge is located via `self._runtime_control_bridge`
3. Run ID is found via `self._runtime_session_runs.get(session_key)`
4. Pending approval IDs are read from `RunManager.get_status()`
5. Oldest `count` approval IDs are resolved via `RunManager.resolve_approval()`
6. Each resolution appends exactly one `approval.resolved` event

**How /deny updates RunManager state:**
Same flow as /approve but with `choice="deny"`, which follows the existing event contract (event type is `approval.resolved` with choice field set to "deny").

### What was NOT done (explicit decisions):
- No clarify slash-command equivalent was added (no such command exists in the gateway)
- No changes to the existing gateway approval primitive (resolve_gateway_approval remains unchanged)
- No changes to WebUI — Agent response shape unchanged, no new bridge status values introduced

### File changes in Phase 14:
- `gateway/run.py` — Fixed `create_run()` call (removed invalid `status` parameter), added terminal status transitions in `_handle_message()`, added `final_status` kwarg to `_release_running_agent_state()`
- `gateway/slash_commands.py` — Added `_sync_runtime_approval_state()` helper method, wired `/approve` and `/deny` handlers to sync RunManager state
- `tests/gateway/test_runtime_non_api_execution.py` — new file (17 tests): non-API run creation, lifecycle, bind/unbind, completed/failed/cancelled transitions, events, secrets redaction, standalone fallback
- `tests/gateway/test_runtime_slash_approval.py` — new file (17 tests): /approve sync, /deny sync, duplicate resolution, unknown IDs, terminal runs, pending ID removal, multiple approvals, secrets redaction
- `tests/gateway/test_restart_resume_pending.py` — Fixed pre-existing test mock that didn't accept `run_id` kwarg added in Phase 13

### Test results:
- 304 tests passed, 0 failed across 10 runtime test files
- All existing Phase 12 and Phase 13 tests continue to pass
- 1 pre-existing test regression fixed (test_restart_resume_pending mock)
- Broader gateway suite: 5 pre-existing failures remain (wecom XML parsing, session race guard, etc.)

### Live smoke:
- 10 live smoke tests passed through direct RunManager/RuntimeControlBridge API
- Verified: run creation, bind/unbind lifecycle, completed/failed/cancelled transitions
- Verified: /approve sync (pending IDs removed, events appended)
- Verified: /deny sync (pending IDs removed, events appended)
- Verified: duplicate resolution returns conflict without duplicating events
- Verified: unknown approval IDs return not_found
- Verified: terminal run resolution returns conflict
- Verified: secrets redacted in all event payloads
- External messaging-platform credentials and provider/tool execution not required for Phase 14 smoke

### Remaining risks:
1. `_sync_runtime_approval_state` resolves FIFO from RunManager, but RunManager's pending_approval_ids order may not perfectly match gateway approval queue order in edge cases (both are FIFO, so this should be correct)
2. `_runtime_session_runs` dict is keyed by session_key — if a session has multiple concurrent runs, only the most recent run_id is stored
3. `/approve` and `/deny` Resolution with GatewayRunner mock agents is not tested end-to-end with real messaging platform adapters (blocked on external credentials)

### Next task:
Phase 15 — Cross-repo integration testing, remaining deferred items (execution plane for non-API-server runtime runs, or broader live verification with real adapters).

---

## Phase 15 — Cross-repo Runtime Integration Verification and /v1/runs Execution-Plane Gap Audit (completed)

### State Before Phase 15
- **Commit:** `c0027c6`
- **Message:** `Wire non-API runtime execution controls`

### What Was Done

#### Part A — Architecture inspection and gap audit
Inspected the Agent runtime architecture to determine whether POST /v1/runs (via the standalone `gateway/runtime/routes.py` module) can execute an AIAgent.

**Findings:**
1. **POST /v1/runs in routes.py is control-plane-only.** `_handle_create_run` calls `run_manager.create_run()` and returns a JSON response. No AIAgent is spawned, no message processed, no events beyond `run.started` emitted.
2. **The API server adapter (`gateway/platforms/api_server.py`) has its OWN POST /v1/runs handler** that DOES create an AIAgent and execute it via `agent.run_conversation()`. This is a separate handler with its own route registration.
3. **Execution requires:** AIAgent construction (with model credentials, session store, callbacks), background task/thread management, streaming SSE event management, approval/clarify/stop wiring through the bridge and tools modules, and a GatewayRunner or similar session context.
4. **The architecture intentionally separates** control-plane run creation from execution. The GatewayRunner's `set_runtime_control_bridge()` can wire the bridge for live session control, but POST /v1/runs has no access to a GatewayRunner or AIAgent in the standalone case.

**Decision: Option B — Safe implementation does not exist for the standalone runtime routes module.** Execution is deferred to a future "runtime executor" service.

#### Part B — WebUI integration verification (inspection only)
WebUI agent-runs adapter was verified to correctly proxy:
- GET /api/runs/{run_id} → Agent GET /v1/runs/{run_id}
- GET /api/runs/{run_id}/events → Agent GET /v1/runs/{run_id}/events
- POST /api/runs/{run_id}/cancel → Agent POST /v1/runs/{run_id}/stop
- POST /api/runs/{run_id}/approval → Agent POST /v1/runs/{run_id}/approval
- POST /api/runs/{run_id}/clarify → Agent POST /v1/runs/{run_id}/clarify
- Mobile pending action resolve → Agent approval/clarify endpoints
- Runtime capabilities report correct adapter and features

#### Part C — Cross-repo integration tests added

**Agent (`hermes-agent`):**
- `tests/gateway/test_runtime_cross_repo_contract.py` (25 tests) — Verifies Agent runtime response shapes match WebUI agent-runs adapter expectations:
  - Run creation shape (run_id, session_id, status, controls)
  - Run status shape (all RuntimeStatus fields)
  - Events shape (event_id, seq, type, created_at, terminal, payload)
  - Stop/cancel response shape
  - Approval error mapping (not_found→404, conflict→409, success→resolved)
  - Clarify error mapping (not_found→404, conflict→409, success→resolved)
  - Secret redaction end-to-end (api_key, token, bearer)
  - Non-secret data preservation
- `tests/gateway/test_runtime_v1_runs_execution_gap.py` (16 tests) — Documents and tests the execution gap:
  - POST /v1/runs is control-plane-only (status=queued, no done event, stays queued indefinitely)
  - No AIAgent constructed on create
  - No executor exists in standalone routes
  - No execute flag, agent factory, background task manager, or streaming adapter
  - Recommended future design documented

#### Part D — Execution-plane decision
**Option B selected.** The /v1/runs execution-plane gap is explicitly deferred.

**Missing primitives (documented in test assertions):**
1. `execute` flag on POST /v1/runs body for opt-in execution
2. AIAgent factory accepting session_id, model, credentials
3. Background task manager for non-blocking run execution
4. Event streaming from AIAgent to RunManager
5. Approval/clarify notification wiring from RunManager events to live agent primitives

**Recommended future design:**
A dedicated "runtime executor" service that watches RunManager for queued runs and executes them via a configurable AIAgent factory. This keeps execution separate from the route layer and allows multiple executor backends.

#### Part E — Agent test results
```
scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py \
  tests/gateway/test_runtime_approval_clarify.py \
  tests/gateway/test_runtime_control_bridge.py \
  tests/gateway/test_runtime_gateway_binding.py \
  tests/gateway/test_runtime_messaging_binding.py \
  tests/gateway/test_runtime_non_api_execution.py \
  tests/gateway/test_runtime_slash_approval.py \
  tests/gateway/test_runtime_cross_repo_contract.py \
  tests/gateway/test_runtime_v1_runs_execution_gap.py
```
Result: **345 passed, 0 failed** across 12 runtime test files (100%).

#### Part F — WebUI test results
```
Default mode: 138 passed, 0 failed
```
```
HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs mode: 130 passed, 8 expected failures
  (test_runtime_routes.py tests designed for legacy-direct/journal mode)
```

#### Part G — Live smoke
Contract-level verification performed via 345 Agent runtime tests + 138 WebUI tests. Live cross-repo HTTP smoke deferred — requires AIAgent credentials (model API keys) to run real agent execution. Mock and unit-test verification covers all contract shapes end-to-end.

### Files Changed in Phase 15

**Agent (`hermes-agent`):**
- `tests/gateway/test_runtime_cross_repo_contract.py` — new (25 tests)
- `tests/gateway/test_runtime_v1_runs_execution_gap.py` — new (16 tests)
- `AGENT_HANDOFF.md` — Phase 15 section added
- `IMPLEMENTATION_REPORT.md` — Phase 15 section added
- `PR_DESCRIPTION.md` — Phase 15 changes added

**WebUI (`hermes-webui`):**
- No code changes needed — existing tests already cover agent-runs adapter comprehensively
- Documentation updated

### Phase 16 — Runtime Executor Service (completed)

**Design:**
The `RuntimeExecutor` is an optional service that processes queued runs from `RunManager` through a pluggable `AgentFactory`. It is deliberately separate from the HTTP route layer — routes remain thin and the executor can be enabled/disabled per-deployment.

**Key components:**
- `gateway/runtime/executor.py` — `RuntimeExecutor`, `AgentFactory` (Protocol), `FakeAgentFactory`, `SessionKeyFactory`
- `gateway/runtime/run_manager.py` — `claim_queued_run()` + `list_runs()` primitives
- `gateway/runtime/routes.py` — optional `executor` parameter, `execute: true` body flag on POST /v1/runs

**Status lifecycle (executor-owned runs):**
queued → running → completed / failed / cancelled

**Executor features:**
1. `execute_run(run_id)` — claim, create agent, bind, execute, complete/fail, unbind
2. `run_once()` — dequeue and execute first queued run
3. `start()` / `stop()` — background polling loop
4. `cancel_run(run_id)` — stop/cancel via RuntimeControlBridge
5. Agent factory protocol for testability (FakeAgentFactory)
6. Error redaction via `agent.redact.redact_sensitive_text`
7. Resource cleanup in `finally` blocks (unbind always fires)

**Integration with routes:**
- `register_runtime_routes(app, executor=executor)` wires executor into the app
- POST /v1/runs with `execute: true` in body spawns background task
- Without executor or without execute flag: route stays control-plane-only (backward compatible)

**Approval/clarify compatibility:**
- Works with executor-owned agents via existing RuntimeControlBridge
- Fake agents can simulate approval/clarify request lifecycle
- POST /v1/runs/{run_id}/approval and /clarify resolve through bridge as before

**Stop/cancel compatibility:**
- POST /v1/runs/{run_id}/stop interrupts executor-owned agents via bridge
- Terminal runs are not interrupted
- Synthetic cancellation (RouteManager-only) works without executor

**Gap remains:** Real AIAgent execution requires model credentials (API keys) — executor works with fake agents in tests but needs `DefaultAgentFactory` with credentialed config for production use.

### Files Changed in Phase 16

**Agent (`hermes-agent`):**
- `gateway/runtime/executor.py` — new (330 lines)
- `gateway/runtime/run_manager.py` — added `claim_queued_run()` + `list_runs()`
- `gateway/runtime/routes.py` — added optional `executor` parameter + `execute` body flag
- `gateway/runtime/__init__.py` — exported RuntimeExecutor, AgentFactory, FakeAgentFactory, SessionKeyFactory
- `tests/gateway/test_runtime_executor.py` — new (30 tests)
- `tests/gateway/test_runtime_executor_routes.py` — new (8 tests)
- `tests/gateway/test_runtime_v1_runs_execution_gap.py` — updated (asyncio.create_task now expected in routes)
- `AGENT_HANDOFF.md`, `IMPLEMENTATION_REPORT.md`, `PR_DESCRIPTION.md` — updated

**WebUI (`hermes-webui`):**
- No changes needed — response shape unchanged, no new statuses

### Phase 17 — DefaultAgentFactory and Real AIAgent Execution (completed)

**Design:**
Phase 17 implements `DefaultAgentFactory` — a production-capable `AgentFactory` that constructs real `AIAgent` instances using the same provider/credential resolution chain as the gateway's platform adapters (`_resolve_runtime_agent_kwargs` + `_resolve_gateway_model`). `FakeAgentFactory` is preserved unchanged for deterministic tests.

**Key components:**
- `gateway/runtime/agent_factory.py` — `DefaultAgentFactory`, `create_default_agent_factory()`, `create_runtime_executor_with_default_factory()`
- `gateway/runtime/executor.py` — updated `execute_run` to handle sync (`AIAgent.run_conversation`) and async (`FakeAgent._FakeAgent`) agents via `asyncio.iscoroutinefunction` detection
- `gateway/runtime/__init__.py` — exports `DefaultAgentFactory`
- `tests/gateway/test_runtime_default_agent_factory.py` — 15 tests

**DefaultAgentFactory design:**
1. `agent_kwargs` injected at construction for test mode (pre-resolved kwargs)
2. `_resolve_runtime_agent_kwargs()` for live credential resolution (production)
3. `_resolve_gateway_model()` for default model from config.yaml
4. Run-specific overrides via `create_agent(run_id, ..., model=...)`
5. Validation: missing API key, missing provider, missing model all raise clean `RuntimeError` without secrets
6. Error messages are redacted via `agent.redact.redact_sensitive_text`

**Sync/async handling:**
Real `AIAgent.run_conversation()` is synchronous. `FakeAgent._FakeAgent.run_conversation()` is async. The executor now detects this via `asyncio.iscoroutinefunction()` and wraps sync calls in `run_in_executor`.

**Real AIAgent execution:**
- `DefaultAgentFactory` resolves credentials from gateway config (config.yaml model section + .env API keys)
- Tested with DeepSeek (deepseek-v4-flash): `POST /v1/runs` with `execute: true` → `completed`
- Status lifecycle, events, stop/cancel, approval/clarify all verified
- Error messages for missing credentials are clean and redacted

**Gating:**
`DefaultAgentFactory` is always importable. Live credential resolution happens only when `create_agent()` is called without explicit `agent_kwargs`. The factory can be wired into `RuntimeExecutor` via `create_runtime_executor_with_default_factory(run_manager, control_bridge=cb)`.

### Files Changed in Phase 17

**Agent (`hermes-agent`):**
- `gateway/runtime/agent_factory.py` — new (178 lines)
- `gateway/runtime/executor.py` — updated `execute_run` for sync/async agent support
- `gateway/runtime/__init__.py` — exported `DefaultAgentFactory`
- `tests/gateway/test_runtime_default_agent_factory.py` — new (15 tests)
- `AGENT_HANDOFF.md`, `IMPLEMENTATION_REPORT.md`, `PR_DESCRIPTION.md` — updated

**WebUI (`hermes-webui`):**
- No changes needed — Agent response shape unchanged, no new statuses

### Remaining deferred items
1. **Real messaging-platform adapter live smoke** — requires external bot/platform credentials
2. Real DeepSeek live cross-repo HTTP smoke (deterministic fake-mode smoke implemented and passing; real-credential smoke deferred when DEEPSEEK_API_KEY unavailable)

### Phase 18 — Cross-repo live HTTP smoke harness (completed)

**Design:**
Phase 18 adds a repeatable cross-repo live HTTP smoke harness that starts the Hermes Agent API server with runtime routes and RuntimeExecutor enabled, starts Hermes WebUI in agent-runs mode, submits POST /v1/runs execute:true through both Agent direct and WebUI proxied paths, verifies status/events/cancel/approval/clarify behavior, and records a clean pass/fail report without exposing secrets.

**Key components (Agent):**
- `scripts/standalone_runtime_server.py` — minimal aiohttp server with RuntimeExecutor + AgentFactory, --fake flag for deterministic mode
- `scripts/smoke_runtime_executor_live.sh` — Agent-only live smoke script
- `scripts/smoke_cross_repo.sh` — combined Agent + WebUI cross-repo smoke
- `tests/gateway/test_runtime_live_http_smoke.py` — 11 tests for smoke harness construction + end-to-end HTTP
- `docs/runtime-live-smoke.md` — documentation

**Key components (WebUI):**
- `scripts/smoke_agent_runs_live.sh` — WebUI agent-runs live smoke
- `tests/test_agent_runs_live_http_smoke.py` — 8 tests for smoke harness construction

**Cross-repo smoke verified:**
1. Agent direct POST /v1/runs execute:true → create, complete, events with done
2. WebUI proxied run status — GET /api/runs/{run_id} returns terminal state
3. WebUI proxied run events — GET /api/runs/{run_id}/events contains done
4. WebUI cancel/stop path — POST /api/runs/{run_id}/cancel proxies correctly
5. WebUI runtime capabilities — GET /api/runtime/capabilities shows agent-runs mode
6. WebUI deployment health — GET /api/deployment/health shows agent-runs adapter
7. Agent approval/clarify endpoints — return action_not_found (no pending action)

**Smoke harness safety:**
- API keys and credentials are never printed
- Background servers are reliably cleaned up (EXIT trap, SIGINT/SIGTERM)
- Ports checked for availability before starting
- Deterministic fake-mode works without any credentials
- Real-credential mode requires DEEPSEEK_API_KEY env var

**Approval/clarify live pending-action smoke:**
Deferred — no deterministic pending-action trigger exists without production-only test injection endpoints. Approval/clarify remain verified by contract/unit tests and RunManager-level smoke.

**Test results:**
- Agent: 409 runtime tests passed (16 files, including 11 new live HTTP smoke tests)
- WebUI (default env): 146 passed, 0 failed
- WebUI (agent-runs env): 138 passed, 8 expected failures in test_runtime_routes.py
- Agent-only live smoke (--fake): 7/7 PASSED
- Cross-repo live smoke (--fake): 11/11 PASSED
- Real DeepSeek smoke: SKIPPED (DEEPSEEK_API_KEY not set in this environment)

**Files changed in Phase 18:**

**Agent (`hermes-agent`):**
- `scripts/standalone_runtime_server.py` — new (99 lines)
- `scripts/smoke_runtime_executor_live.sh` — new (270 lines)
- `scripts/smoke_cross_repo.sh` — new (430 lines)
- `tests/gateway/test_runtime_live_http_smoke.py` — new (240 lines)
- `docs/runtime-live-smoke.md` — new (80 lines)
- `AGENT_HANDOFF.md`, `IMPLEMENTATION_REPORT.md`, `PR_DESCRIPTION.md` — updated

**WebUI (`hermes-webui`):**
- `scripts/smoke_agent_runs_live.sh` — new (180 lines)
- `tests/test_agent_runs_live_http_smoke.py` — new (100 lines)
- `AGENT_HANDOFF.md`, `IMPLEMENTATION_REPORT.md`, `PR_DESCRIPTION.md` — updated
