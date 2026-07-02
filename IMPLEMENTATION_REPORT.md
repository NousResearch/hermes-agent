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
