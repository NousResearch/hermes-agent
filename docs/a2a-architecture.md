# A2A Protocol — Architecture & Developer Guide

This document describes the internal design, data flow, and extension points of the A2A (Agent-to-Agent) protocol implementation in Hermes Agent.

---

## Overview

The A2A implementation consists of two independent components:

| Component | Location | Purpose |
|---|---|---|
| **A2A Server** | `a2a_adapter/` | Expose Hermes as an A2A agent (server side) |
| **A2A Client Tools** | `tools/a2a_tool.py` | Call and discover remote A2A agents from within Hermes (client side) |

These are deliberately decoupled. The server runs as a separate process. The client tool runs inside the agent loop with no dependency on the server.

---

## Repository Structure

```
hermes-agent/
├── a2a_adapter/
│   ├── __init__.py       # Package marker
│   ├── __main__.py       # python -m a2a_adapter entry point
│   ├── entry.py          # CLI startup: load env, configure logging, run uvicorn
│   └── server.py         # SDK integration: AgentExecutor, app factory, helpers
│
├── tools/
│   └── a2a_tool.py       # a2a_discover + a2a_call + a2a_local_scan tools + registry
│
└── docs/
    ├── a2a-user-guide.md      # End-user guide (this sibling doc)
    └── a2a-architecture.md    # This file
```

---

## Design Decisions

### Why `a2a-sdk` for the server?

The official `a2a-sdk` (`pip install 'a2a-sdk[http-server]'`) provides:
- A standards-compliant JSON-RPC 2.0 router
- Correct SSE framing for `tasks/sendSubscribe`
- Task lifecycle management (state machine: `working → completed/failed/canceled`)
- An in-memory (or pluggable persistent) task store
- Future-proof: SDK updates pick up protocol changes automatically

The alternative — raw aiohttp routes — was implemented first and proved in production (Ronny agent, Akela Pack integration). The SDK version supersedes it for correctness and long-term maintainability.

### Why raw `httpx` for the client tool?

The `a2a-sdk` client is pre-1.0 and its API surface changes frequently. The A2A client protocol is simple: `POST /` with JSON-RPC 2.0. Raw `httpx` is stable, has no version coupling, and keeps the client tool dependency-free — it works even when `a2a-sdk` is not installed.

### Why a separate process, not a gateway platform adapter?

The existing gateway platform adapters (`gateway/platforms/api_server.py`, `telegram.py`, etc.) all share a single process and port. Adding A2A routes to the gateway was considered but rejected because:

1. A2A's `POST /` root conflicts with potential future gateway root routes
2. The gateway uses aiohttp; the SDK expects ASGI (Starlette/FastAPI)
3. A separate process gives independent lifecycle, port, and scaling

The pattern mirrors `acp_adapter/` — a standalone `hermes acp` process — which the Hermes project already uses for the Agent Communication Protocol.

---

## A2A Server — Data Flow

```
External Orchestrator (Vertex AI / LangGraph / Akela)
        │
        │  GET /.well-known/agent.json
        ▼
  A2AFastAPIApplication (a2a-sdk)
        │
        │  returns AgentCard (built from env vars)
        │
        │  POST / (JSON-RPC 2.0)
        │  method: tasks/send or tasks/sendSubscribe
        ▼
  DefaultRequestHandler (a2a-sdk)
        │
        │  routes to HermesAgentExecutor.execute()
        ▼
  HermesAgentExecutor
        │
        │  extracts user text from Message.parts
        │  creates Hermes AIAgent via _make_agent()
        │  runs agent.run_conversation() in ThreadPoolExecutor
        │
        │  (streaming path)
        │  stream_delta_callback → delta_queue
        │  drains queue → EventQueue.enqueue_event(TaskArtifactUpdateEvent)
        │
        ▼
  EventQueue (a2a-sdk)
        │
        │  serialises events as SSE frames
        ▼
External Orchestrator receives streaming response
```

### Non-streaming path (`tasks/send`)

1. Orchestrator sends `tasks/send` with a `Message`
2. `DefaultRequestHandler` deserialises and calls `HermesAgentExecutor.execute()`
3. Executor emits `TaskStatus(working)` immediately
4. Runs `agent.run_conversation()` in thread executor (blocks until complete)
5. Emits `TaskArtifactUpdateEvent` with full response text
6. Emits `TaskStatus(completed)` — final
7. Handler serialises result as a JSON-RPC response and returns HTTP 200

### Streaming path (`tasks/sendSubscribe`)

1. Same entry as above up to executor
2. Executor starts agent in thread executor with `stream_delta_callback`
3. Callback pushes tokens to a `queue.Queue`
4. Main async loop drains queue, calling `EventQueue.enqueue_event()` per token
5. SDK serialises each event as an SSE frame: `data: {...}\n\n`
6. Connection stays open until `TaskStatus(completed, final=True)` is emitted

---

## A2A Server — Key Classes and Functions

### `HermesAgentExecutor` (`a2a_adapter/server.py`)

Implements `a2a.server.agent_execution.AgentExecutor`. The two required methods:

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None
async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None
```

`execute()` is the core integration point. It:
- Extracts text from `context.message` (via `_extract_text()`)
- Creates a Hermes `AIAgent` via `_make_agent()`
- Bridges the synchronous `run_conversation()` to async via `loop.run_in_executor()`
- Bridges the streaming callback to the async `EventQueue` via `queue.Queue`

### `_make_agent()` (`a2a_adapter/server.py`)

Mirrors `acp_adapter/session.py:SessionManager._make_agent()`. Creates an `AIAgent` using:
- `hermes_cli.config.load_config()` — reads model from `~/.hermes/config.yaml`
- `hermes_cli.runtime_provider.resolve_runtime_provider()` — resolves API key, base_url, provider

Key difference from the gateway's `_create_agent()`: does not call `_resolve_runtime_agent_kwargs()` or `_resolve_gateway_model()` (gateway-internal functions). Uses the same config path as the ACP adapter for consistency.

### `build_agent_card()` (`a2a_adapter/server.py`)

Builds the `AgentCard` object from environment variables. Called once at startup. The Agent Card is served at `GET /.well-known/agent.json` by the SDK automatically.

### `build_app()` (`a2a_adapter/server.py`)

Factory function used by `entry.py`:
```python
app = build_app(port=9000)
uvicorn.run(app, host="0.0.0.0", port=9000)
```

Wires together:
- `HermesAgentExecutor` — business logic
- `InMemoryTaskStore` — task lifecycle (SDK-managed)
- `DefaultRequestHandler` — JSON-RPC routing
- `A2AStarletteApplication` — ASGI app builder

### `entry.py`

Mirrors `acp_adapter/entry.py` exactly:
1. `_setup_logging()` — route all logs to stderr
2. `_load_env()` — load `~/.hermes/.env` via `hermes_cli.env_loader`
3. `build_app()` + `uvicorn.run()` — start the HTTP server

---

## A2A Client Tool — Data Flow

```
Hermes agent loop
    │
    │  calls tool: a2a_call(url=..., message=...)
    ▼
_tool_a2a_call()
    │
    │  resolves URL (direct or from config a2a_agents)
    ▼
a2a_call()
    │
    │  POST {url} with tasks/send JSON-RPC payload
    │  (httpx sync client, 120s timeout)
    ▼
Remote A2A agent
    │
    │  returns JSON-RPC result with artifacts
    ▼
_tool_a2a_call()
    │
    │  extracts text from result.artifacts[].parts[].text
    │  returns plain string to agent loop
    ▼
Hermes agent loop continues with remote agent's response
```

### `a2a_discover(url)` (`tools/a2a_tool.py`)

1. `GET {url}/.well-known/agent.json`
2. Returns clean JSON summary: name, description, skills, model, streaming support

### `a2a_call(url, message, session_id, bearer_token, stream)` (`tools/a2a_tool.py`)

1. Auto-detects streaming capability from the cached Agent Card (`capabilities.streaming`)
2. **Non-streaming path** (`tasks/send`): single POST, waits for JSON-RPC response
3. **Streaming path** (`tasks/sendSubscribe`): opens an SSE stream, reads `data:` lines, accumulates artifact text, stops on terminal state (`completed`/`failed`/`canceled`)
4. Extracts text from `result.artifacts[].parts[].text`
5. Returns plain text string — same as any other tool output

`stream` parameter: `None` (auto-detect), `True` (force SSE), `False` (force non-streaming).

### `a2a_local_scan(host, port_start, port_end)` (`tools/a2a_tool.py`)

Closes the discovery gap: when no agents are pre-configured and no URL is provided in the prompt, the model can call this tool to find what is running locally.

1. Iterates `port_start` to `port_end` (default `9000–9010`, max range 100 ports)
2. `GET http://{host}:{port}/.well-known/agent.json` with a 2s timeout per port
3. Silently skips closed ports and non-A2A services
4. Returns JSON list of discovered agents: `endpoint`, `name`, `description`, `skills`, `streaming`

Uses only `httpx` — no additional dependencies. Registered in the `a2a` toolset alongside `a2a_discover` and `a2a_call`.

### Named agent resolution

If `agent_name` is provided instead of `url`, `_load_a2a_agents()` reads `~/.hermes/config.yaml`:

```yaml
a2a_agents:
  researcher:
    url: http://192.168.1.100:9000
    bearer_token: "optional"
```

This lets the LLM refer to agents by name without needing to know their URLs.

### Registry (`tools/a2a_tool.py`)

All three tools are registered via `tools.registry.registry.register()` into the `a2a` toolset. The toolset is inactive unless `a2a` is listed in `enabled_toolsets` in `config.yaml` or passed via `--toolsets`. This prevents unintended outbound calls.

---

## Relationship to `acp_adapter/`

| Aspect | ACP (`acp_adapter/`) | A2A (`a2a_adapter/`) |
|---|---|---|
| Transport | stdio (JSON-RPC over stdin/stdout) | HTTP + SSE |
| Use case | IDE integrations (Cursor, VS Code) | Agent orchestrators (Vertex AI, LangGraph) |
| SDK | `agent-client-protocol` | `a2a-sdk` |
| Session model | Long-lived sessions persisted to DB | Stateless per task (session ID for context) |
| Streaming | Via ACP session updates | Via SSE event stream |
| CLI command | `hermes-acp` / `hermes acp` | `hermes-a2a` / `hermes a2a` |
| Agent creation | `SessionManager._make_agent()` | `_make_agent()` (same pattern) |

Both adapters call `AIAgent.run_conversation()` via `loop.run_in_executor()` — the synchronous agent is the same in both cases.

---

## Extension Points

### Adding authentication to the A2A server

The current implementation supports a single Bearer token via `A2A_KEY`. To add per-client API keys or OAuth:

1. Subclass `DefaultRequestHandler` and override `handle()` to inspect request headers
2. Or add a Starlette middleware in `build_app()` before `A2AStarletteApplication`

### Persistent task storage

Replace `InMemoryTaskStore` in `build_app()`:

```python
# SQLite-backed store (available in a2a-sdk[sqlite])
from a2a.server.tasks import SQLiteTaskStore
handler = DefaultRequestHandler(
    agent_executor=HermesAgentExecutor(),
    task_store=SQLiteTaskStore("~/.hermes/a2a_tasks.db"),
)
```

### Maintaining conversation history across tasks

Currently each task creates a fresh agent with empty `conversation_history`. To persist history between tasks from the same `sessionId`:

1. Add a session store (dict keyed by `session_id`)
2. In `execute()`, look up existing history by `context.context_id`
3. Pass history to `_run_sync()` and update store after completion

This mirrors the `acp_adapter/session.py:SessionManager` pattern and could be extracted as a shared utility.

### Adding tool results to A2A artifacts

The current implementation returns only the `final_response` text. To include tool call traces (file diffs, terminal output) as additional artifact parts, extend `execute()` to inspect `result.get("messages")` and extract tool results before emitting the final `TaskArtifactUpdateEvent`.

---

## Testing

### Manual end-to-end test

```bash
# Start the server
AGENT_NAME=test A2A_PORT=9001 hermes-a2a &

# Discover
curl http://localhost:9001/.well-known/agent.json | jq

# Non-streaming call
curl -s -X POST http://localhost:9001 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"what is 2+2?"}]}}}' \
  | jq .result.artifacts[0].parts[0].text
```

### Unit testing `HermesAgentExecutor`

Mock `_make_agent` to avoid real LLM calls:

```python
from unittest.mock import patch, MagicMock
from a2a_adapter.server import HermesAgentExecutor

async def test_execute():
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "4"}
    
    with patch("a2a_adapter.server._make_agent", return_value=mock_agent):
        executor = HermesAgentExecutor()
        # build mock context and event_queue, then await executor.execute(...)
```

### Unit testing `a2a_tool.py`

```python
from unittest.mock import patch
import httpx
from tools.a2a_tool import a2a_call

def test_a2a_call(respx_mock):
    respx_mock.post("http://localhost:9000").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0", "id": "1",
            "result": {
                "id": "t1",
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": "hello"}]}]
            }
        })
    )
    result = a2a_call("http://localhost:9000", "hi")
    assert result == "hello"
```

---

## pyproject.toml Changes

```toml
[project.optional-dependencies]
a2a = ["a2a-sdk[http-server]>=0.2.0,<1", "uvicorn[standard]>=0.24.0,<1"]

[project.scripts]
hermes-a2a = "a2a_adapter.entry:main"

[tool.setuptools.packages.find]
include = [..., "a2a_adapter"]
```

Install command:

```bash
pip install 'hermes-agent[a2a]'
```

---

## Protocol Reference

The implementation targets the A2A specification at `https://a2a-protocol.org`.

Key spec documents:
- [Agent Card spec](https://a2a-protocol.org/latest/spec/agent-card/)
- [Task lifecycle](https://a2a-protocol.org/latest/spec/task-lifecycle/)
- [JSON-RPC methods](https://a2a-protocol.org/latest/spec/json-rpc/)
- [SSE streaming](https://a2a-protocol.org/latest/spec/streaming/)

Related issues in this repo:
- #514 — Feature request: A2A protocol support (this implementation closes it)
- #344 — Multi-Agent Architecture
- #342 — Hermes as MCP Server
- #413 — Cross-CLI Orchestration
