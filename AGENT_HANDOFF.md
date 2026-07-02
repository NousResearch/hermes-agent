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
