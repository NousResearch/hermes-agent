# PR: Hermes Agent /v1/runs Runtime API Foundation

## Summary

Adds a standalone `gateway/runtime/` package providing structured runtime event/status models, an in-memory thread-safe RunManager for run lifecycle management, and aiohttp route handlers implementing the `/v1/runs` API contract. The route module is conditionally mounted into the live Agent API server behind a feature flag.

## Motivation

The WebUI currently instantiates `AIAgent` directly for chat. The long-term goal is for WebUI to delegate to the Agent runtime API. This PR provides the Agent-side foundation: structured event contract, run state management, and HTTP routes that WebUI's agent-runs adapter can call.

## Major Changes

### New: `gateway/runtime/` package
- `models.py` — `RuntimeEvent`, `RuntimeStatus` dataclasses, secret redaction, event/status constants
- `run_manager.py` — `RunManager` class (thread-safe, in-memory) with create, events, stop, complete, fail, approval/clarify stubs
- `routes.py` — `register_runtime_routes(app)` with 6 aiohttp handlers
- `__init__.py` — public API exports

### Modified: `gateway/platforms/api_server.py`
- Conditional route mount in `APIServerAdapter.connect()` when `HERMES_USE_RUNTIME_RUNS=true` or `platforms.api_server.extra.use_runtime_runs` config is set
- Default: legacy embedded handlers (zero behavior change)

### New tests
- `tests/gateway/test_runtime_models.py` — 20 tests
- `tests/gateway/test_runtime_run_manager.py` — 33 tests
- `tests/gateway/test_runtime_routes.py` — 21 tests
- `tests/gateway/test_runtime_server_mount.py` — 31 tests

## API Changes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/runs` | Create a new run (202) |
| GET | `/v1/runs/{run_id}` | Get run status |
| GET | `/v1/runs/{run_id}/events` | Get run events (JSON or SSE) |
| POST | `/v1/runs/{run_id}/stop` | Request run interruption |
| POST | `/v1/runs/{run_id}/approval` | Resolve pending approval (501 not_supported) |
| POST | `/v1/runs/{run_id}/clarify` | Resolve pending clarification (501 not_supported) |

## Config Flags

- `HERMES_USE_RUNTIME_RUNS=true` (env) — enables runtime route module
- `platforms.api_server.extra.use_runtime_runs: true` (config.yaml) — equivalent config gate
- Default: flag absent → legacy embedded handlers

## Tests Run

```
105 focused tests: 105 passed, 0 failed
193 existing api_server tests: 193 passed, 0 failed
23 existing api_server_runs tests: 23 passed, 0 failed
Import smoke: PASS
Live smoke (Phase 10B): PASS
```

## Compatibility Notes

- Backward compatible by default — existing `/v1/runs` API behavior unchanged
- Legacy embedded handlers remain active until feature flag is set
- No change to `/v1/chat/completions`, tool orchestration, or messaging platforms

## Known Limitations

- Approval/clarify resolution returns 501 `not_supported` — requires gateway adapter context
- True live interruption not implemented — `stop_run` transitions status; `agent.interrupt()` needs live `AIAgent`
- `hermes gateway run` full startup blocked by messaging adapter dependencies in test environments

## Rollback Plan

```bash
unset HERMES_USE_RUNTIME_RUNS
# Or: revert commit range
git revert <commit-range>
```

The `gateway/runtime/` package is additive — removing it has zero impact on running Agent server.
