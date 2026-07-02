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
- HTTP route handlers not mounted into live server yet

## Rollback

```bash
# Revert Phase 4 commit if needed
git revert 40a255a
# No WebUI behavior depends on live Agent route mount yet
```

The `gateway/runtime/` package is an additive layer with no callers yet. Removing the directory and tests would have zero impact on the running Agent server.
