# A2A (Agent2Agent) Protocol Server for Hermes Agent

## Motivation

[A2A](https://a2a-protocol.org) is the open standard for **agent-to-agent**
interoperability: it lets one agent discover another, delegate a task, and
stream back status/artifacts over plain HTTP — regardless of framework, vendor,
or language. Where MCP connects an agent to _tools_, A2A connects an agent to
_other agents_. Exposing Hermes over A2A makes it a first-class participant in
multi-agent systems: any A2A-speaking orchestrator (LangGraph, CrewAI, Google
ADK, custom routers, the `a2a-inspector`) can call Hermes as a remote worker.

Hermes already ships sibling protocol adapters — `acp_adapter/` (editor
integration over stdio) and `mcp_serve.py` (tools over MCP). A2A is the missing
third edge, implemented through the bundled platform plugin surface:
**Hermes as a callable agent for other agents.**

## What It Enables

```
┌────────────────────┐                                   ┌──────────────────────┐
│  A2A client / peer  │   GET /.well-known/agent-card.json │  hermes-a2a          │
│  • LangGraph        │ ────────────────────────────────► │  (A2A plugin)        │
│  • CrewAI           │                                    │                      │
│  • Google ADK       │   POST /  message/send             │  ┌────────────────┐  │
│  • a2a-inspector    │ ────────────────────────────────► │  │ HermesAgent    │  │
│  • another Hermes   │                                    │  │ Executor       │  │
│                     │   POST /  message/stream (SSE)     │  └───────┬────────┘  │
│                     │ ◄────────────────────────────────  │          │           │
│                     │   TaskStatusUpdate / Artifact       │   run_conversation() │
└────────────────────┘                                     │      AIAgent         │
                                                            └──────────────────────┘
```

A user would:

1. `pip install hermes-agent[a2a]`
2. `hermes-a2a --host 0.0.0.0 --port 9100` (or `python -m plugins.platforms.a2a`)
3. Point any A2A client at `http://localhost:9100` — it fetches the Agent Card,
   then sends messages and receives streamed task updates.

## Scope (this cut: working vertical slice)

**In:**

- Agent Card served at `/.well-known/agent-card.json` (A2A v0.3, JSON-RPC transport).
- `message/send` — synchronous request/response (returns a completed `Task`).
- `message/stream` — SSE streaming of `TaskStatusUpdateEvent` + `TaskArtifactUpdateEvent`.
- `tasks/get` / `tasks/cancel` — provided by the SDK's `DefaultRequestHandler` +
  a bounded in-memory store; `cancel` wired into Hermes interruption with
  monotonic terminal-state persistence.
- Conversation continuity: A2A `contextId` ↔ a persistent Hermes session
  (one `AIAgent` + history per context).
- Live agent progress: Hermes tool-calls, reasoning, and streamed text mapped
  to A2A working-status updates; the final answer delivered as an artifact.

**Out (deferred, not designed away):**

- Push-notification webhooks (`tasks/pushNotificationConfig/*`).
- Persistent (DB-backed) task store and `tasks/resubscribe`.
- Auth schemes on the card (served unauthenticated; document `0.0.0.0` risk).
- gRPC / HTTP+JSON transports (JSON-RPC only for the slice).
- Multimodal input parts (text-only in; the seam accepts more later).

## Protocol ↔ Hermes mapping

| A2A concept           | Hermes equivalent                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| Agent Card            | Built from `hermes_cli.__version__` + a curated skill list (mirrors `acp_registry/agent.json`) |
| `contextId`           | A Hermes session: one `AIAgent` instance + `conversation_history`                              |
| `taskId`              | One `run_conversation()` turn within a context                                                 |
| `message/send` (text) | `agent.run_conversation(user_message=..., conversation_history=..., task_id=...)`              |
| streamed text delta   | `agent.stream_delta_callback` → `TaskUpdater.update_status(working, msg)`                      |
| tool start            | `agent.tool_progress_callback` (`tool.started`) → working status + tool metadata               |
| tool result / step    | `agent.step_callback` → working status with result metadata                                    |
| model reasoning       | `agent.reasoning_callback` → working status (metadata `kind=reasoning`)                        |
| final response        | `result["final_response"]` → `TaskUpdater.add_artifact(...)` + `complete()`                    |
| `tasks/cancel`        | `session.cancel_event.set()` + `agent.interrupt()`                                             |

This reuses the **exact** callback seam that `acp_adapter/events.py` uses; the
only difference is the translation target (A2A `TaskUpdater` events instead of
ACP `session_update`s).

## Architecture

`AIAgent.run_conversation()` is **synchronous and blocking**, while the a2a-sdk
`AgentExecutor.execute()` is **async** and owns the request's event loop. So,
mirroring the ACP adapter, the agent turn runs in a dedicated bounded worker
pool and its callbacks marshal A2A events back onto the loop
with `asyncio.run_coroutine_threadsafe`. The SDK's `EventQueue` +
`DefaultRequestHandler` turn those events into the JSON-RPC response (or SSE
stream); `A2AStarletteApplication` serves the card and RPC endpoint over uvicorn.

### Module layout (`plugins/platforms/a2a/`)

| File          | Responsibility                                                                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adapter.py`  | `BasePlatformAdapter` lifecycle + `ctx.register_platform()` integration                                                                                                               |
| `card.py`     | `build_agent_card(url)` → `AgentCard` (version, skills, capabilities)                                                                                                                 |
| `sessions.py` | `ContextSessionStore`: `contextId → HermesSession(agent, history, cancel_event)`; `agent_factory` injection for tests; real `AIAgent` build mirrors `acp_adapter.session._make_agent` |
| `events.py`   | Callback factories: AIAgent callbacks → `TaskUpdater` events via a thread-safe scheduler (no heavy Hermes imports, so the adapter is unit-testable standalone)                        |
| `executor.py` | `HermesAgentExecutor(AgentExecutor)`: `execute()` (resolve session → new task → wire callbacks → run turn in thread → stream → artifact + complete) and `cancel()`                    |
| `entry.py`    | CLI: load `~/.hermes/.env`, logging, args (`--host/--port/--check/--version`), build card+handler+app, `uvicorn.run`                                                                  |
| `__main__.py` | `python -m plugins.platforms.a2a`                                                                                                                                                     |

### Data flow (`message/stream`)

```
client ──POST message/stream──► DefaultRequestHandler ──► HermesAgentExecutor.execute()
   │                                                          │ new_task() → enqueue Task
   │                                                          │ TaskUpdater.start_work()
   │                                                          │ to_thread(agent.run_conversation)
   │                                                          │    ├─ stream_delta_cb → update_status(working, text)
   │ ◄────────── SSE: TaskStatusUpdateEvent (working) ────────┤    ├─ tool_progress_cb → update_status(working, tool meta)
   │                                                          │    └─ step_cb        → update_status(working, result meta)
   │                                                          │ add_artifact(final_response)
   │ ◄────────── SSE: TaskArtifactUpdateEvent ────────────────┤ complete()
   │ ◄────────── SSE: TaskStatusUpdateEvent (completed) ──────┘
```

## Packaging

Mirrors the ACP adapter exactly:

- `[project.optional-dependencies]`: `a2a = ["a2a-sdk[http-server]==0.3.26"]`
  (pydantic-based 0.3.x line — matches the broad A2A client ecosystem and
  Hermes' pydantic idioms; pinned exact per repo policy; published 2026-04-09,
  clears the 7-day cooldown).
- `[project.scripts]`: `hermes-a2a = "plugins.platforms.a2a.entry:main"`.
- Bundled discovery: `plugins/platforms/a2a/plugin.yaml` +
  `ctx.register_platform(name="a2a", ...)`.
- `[all]`: add `hermes-agent[a2a]` (parity with `acp`; not lazy-installable).

## Testing

Unit/integration tests under `tests/a2a/`, runnable without model credentials by
injecting a `FakeAgent` (same pattern as `tests/acp_adapter`):

- `test_card.py` — card has required fields and is served at the well-known URL
  (Starlette `TestClient`).
- `test_executor.py` — a fake agent drives `execute()`; assert the emitted event
  sequence is `Task → working → artifact(final_response) → completed`.
- `test_sessions.py` — same `contextId` reuses one agent/history; cancel sets the
  event and calls `interrupt()`.
- `test_end_to_end_echo.py` — build the real Starlette app around an echo
  executor and drive it in-process via `httpx.ASGITransport` with the A2A client,
  proving the full JSON-RPC + SSE path with no network/LLM.

## Why a2a-sdk 0.3.26 (not 1.1.0)

The 1.x line is protobuf-first (`AgentCard`/`Message`/`Part` are proto messages,
verbose to construct, and the ASGI app builder moved). 0.3.26 is the pydantic
line the entire A2A tutorial/client/inspector ecosystem targets today, it reads
naturally alongside Hermes' pydantic code, and it still exposes
`A2AStarletteApplication` + helper builders. For a clean, interoperable slice
it's the better engineering choice; revisit 1.x when the ecosystem's clients move.

## Non-goals / known gaps

- Served unauthenticated by default — bind to `127.0.0.1` unless fronted by a
  reverse proxy / auth layer. Documented in `entry.py --help` and the card.
- Bounded in-memory task store: tasks are lost on restart (acceptable for the slice).
