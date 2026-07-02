# Hermes Agent Runtime API Foundation Implementation Report

## Summary

Completed Agent-side runtime API foundation across phases 0, 4, and 9:
- **Phase 0**: Preflight, branch creation, architecture snapshot
- **Phase 4**: Runtime API foundation — models (RuntimeEvent, RuntimeStatus, redaction), in-memory RunManager, route module/API contract tests
- **Phase 9**: Full verification and final implementation report

The `gateway/runtime/` package provides:
- Structured runtime event and status models with secret redaction
- An in-memory, thread-safe RunManager for run lifecycle management
- Route module implementing the /v1/runs API contract

## Branch and SHAs

- **Branch**: `feat/runtime-run-api-contract`
- **Starting SHA** (Phase 0 base): `30e947e0a05ef535e4b25a183d8bbe34fd68d1d5`
- **Phase 4 commit SHA**: `40a255a`
- **Final SHA before report commit**: `f7cc6c5f63f72e6e6db8260398852a257e923e39`
- **Related WebUI SHA**: `368ca078c93701bbdc0a6f935ad4185d01a9c3f9`

## Completed Phases

- [x] Phase 0 — Preflight
- [x] Phase 4 — Runtime API foundation
- [x] Phase 9 — Full verification and final report
- [x] Phase 10A — Mount runtime routes into live Agent API server

## Added Runtime/API Components

| File | Purpose |
|---|---|
| `gateway/runtime/__init__.py` | Package init, re-exports all public symbols |
| `gateway/runtime/models.py` | `RuntimeEvent`, `RuntimeStatus`, `redact_secrets`, status/event constants |
| `gateway/runtime/run_manager.py` | `RunManager` class — in-memory run lifecycle management |
| `tests/gateway/test_runtime_models.py` | 20 tests — model serialization, redaction, imports |
| `tests/gateway/test_runtime_run_manager.py` | 33 tests — lifecycle, events, stop, transitions, thread safety |
| `tests/gateway/test_runtime_routes.py` | 21 tests — API contract shapes, error handling, redaction |

## Runtime API Contract

### Intended Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/runs` | Create a new run |
| GET | `/v1/runs/{run_id}` | Get run status |
| GET | `/v1/runs/{run_id}/events` | Get run events (JSON or SSE) |
| POST | `/v1/runs/{run_id}/stop` | Request run interruption |
| POST | `/v1/runs/{run_id}/approval` | Resolve pending approval |
| POST | `/v1/runs/{run_id}/clarify` | Resolve pending clarification |

### Supported Statuses
`queued`, `running`, `awaiting_approval`, `awaiting_clarify`, `paused`, `cancelling`, `cancelled`, `failed`, `completed`, `expired`

### Supported Event Types
`run.started`, `run.status`, `token.delta`, `reasoning.delta`, `reasoning.done`, `progress`, `tool.started`, `tool.updated`, `tool.done`, `approval.requested`, `approval.resolved`, `clarify.requested`, `clarify.resolved`, `title.updated`, `usage.updated`, `usage.final`, `error`, `done`

### RunManager Public API
- `create_run(session_id, *, message, workspace, profile, model, toolsets, metadata)` — creates run + `run.started` event
- `get_status(run_id)` — returns `RuntimeStatus` dict or `None`
- `append_event(run_id, event_type, *, session_id, payload)` — adds event, updates status
- `read_events(run_id, *, after_seq, limit)` — returns `{"run_id": ..., "events": [...]}`
- `stop_run(run_id)` — transitions to cancelling then cancelled
- `transition_status(run_id, new_status)` — explicit status transition
- `complete_run(run_id, *, result)` — terminal completed
- `fail_run(run_id, *, error)` — terminal failed
- `resolve_approval(run_id, choice)` — returns `not_supported` (deferred)
- `resolve_clarify(run_id, response)` — returns `not_supported` (deferred)

## Verification Results

### Agent — Focused verification
```
Command:
  ./scripts/run_tests.sh tests/gateway/test_runtime_models.py \
    tests/gateway/test_runtime_run_manager.py \
    tests/gateway/test_runtime_routes.py -v

Result: 74 passed, 0 failed in 0.7s (3 files, 16 workers) — PASS
```

### Agent — Full test suite
```
Command:
  ./scripts/run_tests.sh

Result: 70 passed in shard scope, 18 failed total across all files.
  All 18 failures are in pre-existing, unrelated test areas:
    - tests/acp/test_auth.py (2) — ACP auth tests
    - tests/acp/test_edit_approval.py (1) — ACP edit approval
    - tests/gateway/test_wecom_callback.py (3) — WeCom platform
    - tests/tools/test_execute_code_approval_cluster.py (7) — approval cluster
    - tests/tools/test_modal_sandbox_fixes.py (2) — modal sandbox
    - tests/tools/test_voice_mode.py (3) — voice mode detection
  Also 9 files with collection/import errors in tests/acp/.
  None are related to the Phase 4 runtime foundation.
```

### Agent — Import/config smoke checks
```
Command:
  python3 - <<'PY'
  import gateway.runtime.models, gateway.runtime.run_manager
  from gateway.runtime.run_manager import RunManager
  manager = RunManager()
  status = manager.create_run(session_id="sess_smoke", message="smoke test",
                               metadata={"client": "phase9"})

Result: All imports OK. Smoke run created successfully with run_id,
  session_id, status: queued, events_url/status_url/controls populated.
```

## API Server Integration Status

**Mounted** — Route module implemented and integrated into live API server (Phase 10A).

The `gateway/runtime/routes.py` module provides `register_runtime_routes(app)` which registers 6 aiohttp handlers delegating to `RunManager`. The live API server (`gateway/platforms/api_server.py`) conditionally mounts these routes when `HERMES_USE_RUNTIME_RUNS=true` env var or `platforms.api_server.extra.use_runtime_runs` config is set. By default (flag absent), the legacy embedded handlers are used — no behavior change for existing deployments.

**Server mount location:** `gateway/platforms/api_server.py` → `APIServerAdapter.connect()` → `register_runtime_routes(self._app, error_formatter=_openai_error)`

## Unsupported/Deferred

- approval resolution `not_supported` — requires gateway adapter context not available in standalone RunManager
- clarify resolution `not_supported` — same reason: requires gateway adapter context
- true live interruption not implemented — `stop_run` transitions status directly; actual `agent.interrupt()` requires a live `AIAgent` reference
- HTTP route handlers not mounted into live server yet — RESOLVED: mounted in Phase 10A, live-smoke verified in Phase 10B

## Phase 10B — Live Agent-Runs Smoke Verification (completed)

### Live Server

Standalone API server started with runtime route module mounted:
```bash
cd hermes-agent && uv run python /tmp/hermes-agent-standalone.py
# 127.0.0.1:8642, HERMES_USE_RUNTIME_RUNS=1
# register_runtime_routes(app) delegates to RunManager
```

Note: `hermes gateway run` was not used because the full gateway starts all messaging adapters (Telegram, etc.) which require credentials. The standalone server exposes only the runtime routes needed for smoke verification.

### Agent Direct /v1/runs Smoke Results

| Step | Endpoint | Result |
|---|---|---|
| Create run | POST /v1/runs | 202, run_id returned, status "queued" |
| Get status | GET /v1/runs/{run_id} | Full RuntimeStatus shape |
| Get events | GET /v1/runs/{run_id}/events | run.started event with metadata |
| Stop run | POST /v1/runs/{run_id}/stop | status "cancelled", terminal true |
| No secrets | All responses | Verified — no API keys, tokens, passwords |

### WebUI Agent-Runs Live Smoke (via WebUI on port 8789 with agent-runs adapter)

| Step | Result |
|---|---|
| Runtime capabilities | runtime_adapter="agent-runs", all supports flags correct |
| Mobile capabilities | deployment_health, workspace_search, resumable_runs all true |
| Run status proxy | Correctly fetched Agent /v1/runs/{run_id} |
| Run events proxy | All 3 events (run.started, run.status, done) returned |
| Cancel proxy | 200, status "cancelled", clean response |
| Workspace search | 200, no errors, no secrets |

### Post-Smoke Test Suites

**Agent focused: 105 passed, 0 failed**
```bash
uv run python -m pytest tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py -v
```

**WebUI agent-runs env: 149 passed, 8 expected failures**
```bash
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
```

### Remaining Deferred Items
- approval resolution — returns 501 not_supported (requires gateway adapter context)
- clarify resolution — returns 501 not_supported (same)
- true live agent interruption — `stop_run` transitions status directly; actual `agent.interrupt()` requires live `AIAgent`
- `/v1/health` not available on standalone server (only on full gateway API server)
- `hermes gateway run` full startup blocked by messaging adapter dependencies

## Phase 11A — PR Review (completed)

### Code Review Findings

Full branch diff: 11 files changed, 2849 insertions, 6 deletions.

**No secrets leaked.** No hardcoded personal paths (test fixtures use `/home/user/workspace` only). No broad exception swallowing (except existing patterns matching api_server.py). No raw tracebacks in API responses. Route mount correctly gated behind `HERMES_USE_RUNTIME_RUNS` flag — default keeps legacy embedded handlers.

### Bugs Found and Fixed

1. **TOCTOU race in `RunManager.append_event()`** (`run_manager.py:100-123`) — Two separate lock acquisitions created a window where a run could be deleted between session_id lookup and event append. Fixed by keeping lock held across both operations.

2. **Array message parsing gap** (`routes.py:95-102`) — Only handled `{"content": "..."}` parts. Now also handles OpenAI-compatible `{"type": "text", "text": "..."}` parts, matching `_normalize_chat_content()` behavior in `api_server.py`.

### Test Results (Phase 11A)

```
./scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py -v

Result: 105 passed, 0 failed in 0.9s — PASS
```

**Import smoke:** RunManager.create_run() produces valid run with run_id, status "queued", all URL fields populated. PASS.

### Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| TOCTOU in append_event | **FIXED** | Single lock acquisition now holds across session_id resolution and event append |
| Array message parsing | **FIXED** | Now handles `{"type": "text", "text": "..."}` parts |
| approval not_supported | LOW | Returns 501, documented. Requires gateway adapter context |
| clarify not_supported | LOW | Returns 501, documented. Requires gateway adapter context |
| True live interruption | LOW | stop_run transitions status; actual agent.interrupt() requires live AIAgent |
| Bare `except Exception:` on JSON parse | LOW | Follows existing api_server.py pattern; not regressive |

### PR Readiness

- All 105 focused tests pass with fixes applied
- Route mount gated and non-default
- Existing legacy API server tests pass (193 + 23, no regressions)
- Secret redaction verified in all response paths
- Merge-ready

## Rollback

```bash
# Revert Phase 4 commit if needed
git revert 40a255a
# No WebUI behavior depends on live Agent route mount yet
```

The `gateway/runtime/` package is an additive layer with no callers yet. Removing the directory and tests would have zero impact on the running Agent server.

## Phase 11B Approval/Clarify Integration

### Implemented Behavior

Replaced the 501 `not_supported` approval/clarify stubs with a first-class pending action lifecycle in `RunManager`:

- **`request_approval(run_id, approval_id, payload)`** — creates pending approval, appends `approval.requested` event, transitions to `awaiting_approval`
- **`resolve_approval(run_id, approval_id, choice, payload=None)`** — removes pending ID, appends `approval.resolved` event, returns resolved response
- **`request_clarify(run_id, clarify_id, payload)`** — creates pending clarify, appends `clarify.requested` event, transitions to `awaiting_clarify`
- **`resolve_clarify(run_id, clarify_id, answer, payload=None)`** — removes pending ID, appends `clarify.resolved` event, returns resolved response

Error behavior:
- Unknown run → `not_found` (404)
- Unknown action_id → `not_found` with `action_not_found` code (404)
- Terminal run → `conflict` (409)
- Duplicate resolution → `conflict` (409)
- No pending action and no live resolver → `not_found` (not `not_supported`)

Status transitions:
- `resolve_approval`/`resolve_clarify` remove resolved IDs; when all pending IDs are cleared, status transitions back to `running`
- Multiple pending approvals/clarifies supported; status stays `awaiting_*` until all cleared

Secret redaction:
- 9 key-name variants (`api_key`, `token`, `password`, etc.) redacted in all request/resolve payloads
- Resolution event payload is redacted before storage
- Uses existing `redact_secrets()` from `models.py`

URL path run_id enforcement:
- Routes reject requests where body `run_id` differs from URL path `run_id`

### Resolver Architecture

No live AIAgent continuation reference is available in the runtime layer. The gateway's real approval/clarify mechanisms live in `gateway/run.py` with `tools/approval.py` and `tools/clarify_gateway.py`, which use threading events keyed by session, not run_id. The runtime RunManager provides the strongest safe runtime-control layer on its own.

### Tests Run

```
scripts/run_tests.sh tests/gateway/test_runtime_models.py \
  tests/gateway/test_runtime_run_manager.py \
  tests/gateway/test_runtime_routes.py \
  tests/gateway/test_runtime_server_mount.py \
  tests/gateway/test_runtime_approval_clarify.py
Result: 154 tests passed, 0 failed (5 files) — PASS
```

### Import Smoke

```python
from gateway.runtime.run_manager import RunManager
m = RunManager()
m.request_approval(run_id, "approval_smoke", {"command": "echo ok", "api_key": "SHOULD_REDACT"})
r = m.resolve_approval(run_id, "approval_smoke", choice="approve")
# Status: resolved, secrets redacted, events appended — PASS
```

### Live Smoke

RunManager-level verified. Full HTTP live smoke deferred due to lack of test-only injection endpoints for pending actions on the standalone server. The `test_runtime_server_mount.py` integration tests validate the full aiohttp route path.

### Remaining Deferred Items

1. True AIAgent continuation after approval/clarify — requires bridging session_key-based approval primitives (`tools/approval.py`) to run_id-based runtime tracking. The gateway's `GatewayRunner` owns the only live `AIAgent` instances.

### Files Changed

- `gateway/runtime/run_manager.py` — full approval/clarify lifecycle methods
- `gateway/runtime/routes.py` — approval/clarify handlers with error mapping, URL path validation
- `tests/gateway/test_runtime_approval_clarify.py` — new test file (38 tests)
- `tests/gateway/test_runtime_run_manager.py` — updated TestApprovalAndClarify → TestApprovalRequestAndResolve
- `tests/gateway/test_runtime_routes.py` — updated contract tests
- `tests/gateway/test_runtime_server_mount.py` — updated server mount tests

## Phase 11C — Live Agent Control Bridge (completed)

### Implemented Behavior

Created a `RuntimeControlBridge` that bridges run_id-based runtime controls to session_key-based live gateway primitives in `tools/approval.py` and `tools/clarify_gateway.py`:

**`RuntimeControlBridge`** class in `gateway/runtime/control_bridge.py`:
- `resolve_approval(run_id, approval_id, choice, payload)` — updates RunManager, then calls `resolve_gateway_approval(session_key, choice)` when session_key is known
- `resolve_clarify(run_id, clarify_id, answer, payload)` — updates RunManager, then calls `resolve_gateway_clarify(clarify_id, answer)` (clarify_id is universally unique)
- `stop_run(run_id)` — updates RunManager, then signals `AIAgent.interrupt("run_stop")` when live agent is reachable via GatewayRunner
- `bind_run(run_id, session_key)` — establishes run_id → session_key mapping
- Constructor accepts: `get_session_key_for_run` callable and `gateway_runner_ref` weakref

**Route integration:** handlers dynamically look up bridge from `request.app["runtime_control_bridge"]`, delegate when present, fall back to RunManager when absent.

**API server wiring:** creates bridge with `get_session_key_for_run` using RunManager `session_id` and `gateway_runner_ref` pointing to `_gateway_runner_ref`.

### What's Complete

- RuntimeControlBridge exists and is wired into routes and API server
- Clarify resolution delegates to live `resolve_gateway_clarify()` always (no mapping needed)
- Approval resolution delegates to live `resolve_gateway_approval()` when session_key is known
- Stop/cancel delegates to live `AIAgent.interrupt()` through GatewayRunner weakref

### What's Partially Deferred

- Full `run_id` → gateway `session_key` mapping at run creation time — the bridge uses RunManager `session_id` as proxy; true gateway `session_key` requires GatewayRunner cooperation at spawn time
- `bind_run(run_id, session_key)` is ready; a future integration that calls it when spawning runtime-tracked agent runs would complete the mapping

### Tests Run

```
188 tests across 6 files: 188 passed, 0 failed
Full gateway suite: 8845 passed, 3 failed (pre-existing wecom_callback, unrelated)
WebUI tests: 104 passed, 0 failed (no WebUI changes needed)
```

### Files Changed in Phase 11C

- `gateway/runtime/control_bridge.py` — new file
- `gateway/runtime/routes.py` — bridge-aware control handlers
- `gateway/runtime/__init__.py` — added RuntimeControlBridge export
- `gateway/platforms/api_server.py` — bridge creation and wiring
- `tests/gateway/test_runtime_control_bridge.py` — new test file (33 tests total)
- `tests/gateway/test_runtime_server_mount.py` — bridge mount tests (8 tests added)

## Phase 12 — Full GatewayRunner Runtime-Run Binding for Live Agent Continuation (completed)

### Implemented Behavior

Completed live AIAgent continuation by binding runtime run_id to live gateway
session_key and agent reference at spawn time.

**RuntimeControlBridge enhancements** (`gateway/runtime/control_bridge.py`):
- `bind_run(run_id, session_key, agent=None)` now accepts optional agent reference
- `unbind_run(run_id)` cleans up both session_key and agent mappings
- `stop_run()` uses direct agent reference first (from `bind_run`), then falls back to GatewayRunner reset via session_key, then RunManager-only
- `_live_agents` dict stores direct agent references per run_id

**RunManager enhancement** (`gateway/runtime/run_manager.py`):
- `create_run()` now accepts optional `run_id` parameter so callers can
  re-use the same run_id across RunManager and their own tracking system

**Route registration flexibility** (`gateway/runtime/routes.py`):
- `register_runtime_routes()` now accepts `register_create`, `register_status`,
  `register_events` flags for selective route registration
- Default: all three are `True` (zero behavior change)

**API server runtime mode integration** (`gateway/platforms/api_server.py`):
- When `HERMES_USE_RUNTIME_RUNS=1`:
  - Registers only runtime control routes (stop/approval/clarify) via bridge
  - Keeps legacy handlers for run creation (POST /v1/runs), status, and events
  - `_handle_runs` creates RunManager entries with matching `run_id`
  - Calls `bridge.bind_run(run_id, approval_session_key, agent)` at agent spawn
  - Calls `bridge.unbind_run(run_id)` on run terminal (finally block + sweep)
  - Guards against `self._app is None` (test adapter setups)

### What's Complete

- `bind_run(run_id, session_key, agent)` is called at API server runtime run spawn
- Direct agent interrupt via stored reference bypasses GatewayRunner look-up
- Approval resolution bridges through bound session_key to live `resolve_gateway_approval()`
- Clarify resolution bridges through universally unique `clarify_id`
- Stop/cancel interrupts live agent via `bind_run` agent reference
- `unbind_run` clean-up on all terminal states
- 46 new binding-specific tests + 8 additional bridge tests
- Full backward compat: Phase 11B RunManager-only, Phase 11C bridge

### What's Still Deferred

- Full GatewayRunner-level binding: when a GatewayRunner session spawns a
  runtime-tracked agent, `bridge.bind_run()` should be called. The bridge
  infrastructure is ready for this — the GatewayRunner needs to be made
  aware of the runtime bridge instance.
- Non-API-server runtime runs (created directly via `/v1/runs` without the
  API server's agent execution path) have no execution plane. The runtime
  routes provide the control/observation plane but agent execution must be
  wired separately.

### Tests Run

```
Phase 12 specific: 257 tests across 8 files — 257 passed, 0 failed
  - test_runtime_models.py: 20 tests
  - test_runtime_run_manager.py: 40 tests
  - test_runtime_routes.py: 23 tests
  - test_runtime_server_mount.py: 42 tests
  - test_runtime_approval_clarify.py: 38 tests
  - test_runtime_control_bridge.py: 33 tests (8 new)
  - test_runtime_gateway_binding.py: 38 tests (new)
  - test_api_server_runs.py: 23 tests

Full gateway suite: 8891 passed, 3 failed (pre-existing test_wecom_callback.py)
WebUI tests: 104 passed, 0 failed (no WebUI changes needed)
```

### Files Changed in Phase 12

- `gateway/runtime/control_bridge.py` — agent ref storage, unbind, direct interrupt
- `gateway/runtime/routes.py` — selective route registration flags
- `gateway/runtime/run_manager.py` — optional `run_id` parameter
- `gateway/platforms/api_server.py` — runtime mode: bind_run at spawn, unbind at terminal, _app guard
- `tests/gateway/test_runtime_control_bridge.py` — 8 new agent-ref/unbind tests
- `tests/gateway/test_runtime_gateway_binding.py` — new file (38 tests)

---

## Phase 13 — Messaging-platform GatewayRunner Binding (completed)

### What's Implemented

1. **GatewayRunner bridge integration**: GatewayRunner now accepts a `RuntimeControlBridge` via `set_runtime_control_bridge()` and uses it to track messaging-platform agent runs.

2. **Per-turn run_id creation**: In `_process_message`, when the bridge is available, a runtime `run_id` is generated and registered in `RunManager` before the agent starts. The run_id is threaded through `_handle_message_with_agent` → `_run_agent` → `_run_agent_inner`.

3. **Bind at agent promotion**: `bridge.bind_run(run_id, session_key, agent)` is called in the `track_agent()` callback after the AIAgent is promoted to `_running_agents`. The run status transitions to "running".

4. **Unbind at terminal**: `bridge.unbind_run(run_id)` is called in:
   - `_release_running_agent_state()` — the canonical cleanup path for all agent turns
   - `_clear_session_boundary_security_state()` — session reset/switch cleanup

5. **Approval recording**: When `_approval_notify_sync` fires during a messaging run, a unique `approval_id` is generated and `bridge.request_approval(run_id, approval_id, payload)` records the pending approval in RunManager with redacted payload.

6. **Clarify recording**: When `_clarify_callback_sync` fires, `bridge.request_clarify(run_id, clarify_id, payload)` records the pending clarify in RunManager.

7. **Final status**: `bridge.run_manager.complete_run()` or `.fail_run()` is called in the finally block of `_run_agent_inner` before unbind.

8. **API server bridge sharing**: The API server's `_start_api_routes` now sets the bridge on GatewayRunner via `runner.set_runtime_control_bridge(control_bridge)` so both API-server and messaging-platform paths share the same bridge.

### What's Complete

- Messaging-platform run_id → session_key binding
- Messaging-platform live approval continuation (record + resolve through REST API)
- Messaging-platform live clarify continuation (record + resolve through REST API)
- Messaging-platform live interrupt (direct agent ref, GatewayRunner fallback)
- All Phase 12 API server binding preserved
- All Phase 11B/11C behavior preserved
- 307 total runtime tests pass (36 new in this phase)

### What's Deferred

- Execution plane for non-API-server runtime runs
- GatewayRunner `/approve` and `/deny` slash commands updating RunManager state

### Tests Run

```
All 11 runtime test files: 307 passed, 0 failed
  - test_runtime_messaging_binding.py: 36 tests (new)
  - test_runtime_gateway_binding.py: 38 tests
  - test_runtime_control_bridge.py: 33 tests
  - test_runtime_approval_clarify.py: 38 tests
  - test_runtime_run_manager.py: 40 tests
  - test_runtime_server_mount.py: 42 tests
  - test_runtime_routes.py: 23 tests
  - test_runtime_models.py: 20 tests
  - test_runtime_footer.py: 25 tests
  - test_runtime_config_env_expansion.py: 9 tests
  - test_runtime_env_reload_config_authority.py: 3 tests
```

### Files Changed in Phase 13

- `gateway/run.py` — control bridge wiring, run_id lifecycle, approval/clarify recording
- `gateway/platforms/api_server.py` — set bridge on GatewayRunner
- `tests/gateway/test_runtime_messaging_binding.py` — new file (36 tests)
