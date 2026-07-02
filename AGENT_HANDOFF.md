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
**Phase 1 — Audit existing /v1/runs contract:**
- Trace the full lifecycle from `POST /v1/runs` through agent execution to `run.completed`/`run.failed`
- Document the SSE event schema and pollable status schema
- Identify gaps between current contract and desired runtime API contract
- Identify what (if anything) needs extraction into a standalone RunManager class
- Audit secret redaction in event payloads and response bodies
