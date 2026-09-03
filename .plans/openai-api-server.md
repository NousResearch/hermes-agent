# OpenAI-Compatible API Server for Hermes Agent

## Motivation

Every major chat frontend (Open WebUI 126k★, LobeChat 73k★, LibreChat 34k★,
AnythingLLM 56k★, NextChat 87k★, ChatBox 39k★, Jan 26k★, HF Chat-UI 8k★,
big-AGI 7k★) connects to backends via the OpenAI-compatible REST API with
SSE streaming. By exposing this endpoint, hermes-agent becomes instantly
usable as a backend for all of them — no custom adapters needed.

## What It Enables

```
┌──────────────────┐
│  Open WebUI      │──┐
│  LobeChat        │  │    POST /v1/chat/completions
│  LibreChat       │  ├──► Authorization: Bearer <key>     ┌─────────────────┐
│  AnythingLLM     │  │    {"messages": [...]}             │  hermes-agent   │
│  NextChat        │  │                                    │  gateway        │
│  Any OAI client  │──┘    ◄── SSE streaming response      │  (API server)   │
└──────────────────┘                                        └─────────────────┘
```

A user would:
1. Set `API_SERVER_ENABLED=true` in `~/.hermes/.env`
2. Run `hermes gateway` (API server starts alongside Telegram/Discord/etc.)
3. Point Open WebUI (or any frontend) at `http://localhost:8642/v1`
4. Chat with hermes-agent through any OpenAI-compatible UI

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | Chat with the agent (streaming + non-streaming) |
| GET | `/v1/models` | List available "models" (returns hermes-agent as a model) |
| GET | `/health` | Health check |

## Architecture

### Option A: Gateway Platform Adapter (recommended)

Create `gateway/platforms/api_server.py` as a new platform adapter that
extends `BasePlatformAdapter`. This is the cleanest approach because:

- Reuses all gateway infrastructure (session management, auth, context building)
- Runs in the same async loop as other adapters
- Gets message handling, interrupt support, and session persistence for free
- Follows the established pattern (like Telegram, Discord, etc.)
- Uses `aiohttp.web` (already a dependency) for the HTTP server

The adapter would start an `aiohttp.web.Application` server in `connect()`
and route incoming HTTP requests through the standard `handle_message()` pipeline.

### Option B: Standalone Component

A separate HTTP server class in `gateway/api_server.py` that creates its own
AIAgent instances directly. Simpler but duplicates session/auth logic.

**Recommendation: Option A** — fits the existing architecture, less code to
maintain, gets all gateway features for free.

## Request/Response Format

### Chat Completions (non-streaming)

```
POST /v1/chat/completions
Authorization: Bearer hermes-api-key-here
Content-Type: application/json

{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What files are in the current directory?"}
  ],
  "stream": false,
  "temperature": 0.7
}
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Here are the files in the current directory:\n..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250
  }
}
```

### Chat Completions (streaming)

Same request with `"stream": true`. Response is SSE:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Here "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"are "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Models List

```
GET /v1/models
Authorization: Bearer hermes-api-key-here
```

Response:
```json
{
  "object": "list",
  "data": [{
    "id": "hermes-agent",
    "object": "model",
    "created": 1710000000,
    "owned_by": "hermes-agent"
  }]
}
```

## Key Design Decisions

### 1. Session Management

The OpenAI API is stateless — each request includes the full conversation.
But hermes-agent sessions have persistent state (memory, skills, tool context).

**Approach: Hybrid**
- Default: Stateless. Each request is independent. The `messages` array IS
  the conversation. No session persistence between requests.
- Opt-in persistent sessions via `X-Session-ID` header. When provided, the
  server maintains session state across requests (conversation history,
  memory context, tool state). This enables richer agent behavior.
- The session ID also enables interrupt support — a subsequent request with
  the same session ID while one is running triggers an interrupt.

### 2. Streaming

The agent's `run_conversation()` is synchronous and returns the full response.
For real SSE streaming, we need to emit chunks as they're generated.

**Phase 1 (MVP):** Run agent in a thread, return the complete response as
a single SSE chunk + `[DONE]`. This works with all frontends — they just see
a fast single-chunk response. Not true streaming but functional.

**Phase 2:** Add a response callback to AIAgent that emits text chunks as the
LLM generates them. The API server captures these via a queue and streams them
as SSE events. This gives real token-by-token streaming.

**Phase 3:** Stream tool execution progress too — emit tool call/result events
as the agent works, giving frontends visibility into what the agent is doing.

### 3. Tool Transparency

Two modes:
- **Opaque (default):** Frontends see only the final response. Tool calls
  happen server-side and are invisible. Best for general-purpose UIs.
- **Transparent (opt-in via header):** Tool calls are emitted as OpenAI-format
  tool_call/tool_result messages in the stream. Useful for agent-aware frontends.

### 4. Authentication

- Bearer token via `Authorization: Bearer <key>` header
- Token configured via `API_SERVER_KEY` env var
- Optional: allow unauthenticated local-only access (127.0.0.1 bind)
- Follows the same pattern as other platform adapters

### 5. Model Mapping

Frontends send `"model": "hermes-agent"` (or whatever). The actual LLM model
used is configured server-side in config.yaml. The API server maps any
requested model name to the configured hermes-agent model.

Optionally, allow model passthrough: if the frontend sends
`"model": "anthropic/claude-sonnet-4"`, the agent uses that model. Controlled
by a config flag.

## Configuration

```yaml
# In config.yaml
api_server:
  enabled: true
  port: 8642
  host: "127.0.0.1"        # localhost only by default
  key: "your-secret-key"   # or via API_SERVER_KEY env var
  allow_model_override: false  # let clients choose the model
  max_concurrent: 5         # max simultaneous requests
```

Environment variables:
```bash
API_SERVER_ENABLED=true
API_SERVER_PORT=8642
API_SERVER_HOST=127.0.0.1
API_SERVER_KEY=your-secret-key
```

## Implementation Plan

### Phase 1: MVP (non-streaming) — PR

1. `gateway/platforms/api_server.py` — new adapter
   - aiohttp.web server with endpoints:
     - `POST /v1/chat/completions` — Chat Completions API (universal compat)
     - `POST /v1/responses` — Responses API (server-side state, tool preservation)
     - `GET /v1/models` — list available models
     - `GET /health` — health check
   - Bearer token auth middleware
   - Non-streaming responses (run agent, return full result)
   - Chat Completions: stateless, messages array is the conversation
   - Responses API: server-side conversation storage via previous_response_id
     - Store full internal conversation (including tool calls) keyed by response ID
     - On subsequent requests, reconstruct full context from stored chain
   - Frontend system prompt layered on top of hermes-agent's core prompt

2. `gateway/config.py` — add `Platform.API_SERVER` enum + config

3. `gateway/run.py` — register adapter in `_create_adapter()`

4. Tests in `tests/gateway/test_api_server.py`

### Phase 2: SSE Streaming

1. Add response streaming to both endpoints
   - Chat Completions: `choices[0].delta.content` SSE format
   - Responses API: semantic events (response.output_text.delta, etc.)
   - Run agent in thread, collect output via callback queue
   - Handle client disconnect (cancel agent)

2. Add `stream_callback` parameter to `AIAgent.run_conversation()`

### Phase 3: Enhanced Features

1. Tool call transparency mode (opt-in)
2. Model passthrough/override
3. Concurrent request limiting
4. Usage tracking / rate limiting
5. CORS headers for browser-based frontends
6. GET /v1/responses/{id} — retrieve stored response
7. DELETE /v1/responses/{id} — delete stored response

## Files Changed

| File | Change |
|------|--------|
| `gateway/platforms/api_server.py` | NEW — main adapter (~300 lines) |
| `gateway/config.py` | Add Platform.API_SERVER + config (~20 lines) |
| `gateway/run.py` | Register adapter in _create_adapter() (~10 lines) |
| `tests/gateway/test_api_server.py` | NEW — tests (~200 lines) |
| `cli-config.yaml.example` | Add api_server section |
| `README.md` | Mention API server in platform list |

## Compatibility Matrix

Once implemented, hermes-agent works as a drop-in backend for:

| Frontend | Stars | How to Connect |
|----------|-------|---------------|
| Open WebUI | 126k | Settings → Connections → Add OpenAI API, URL: `http://localhost:8642/v1` |
| NextChat | 87k | BASE_URL env var |
| LobeChat | 73k | Custom provider endpoint |
| AnythingLLM | 56k | LLM Provider → Generic OpenAI |
| Oobabooga | 42k | Already a backend, not a frontend |
| ChatBox | 39k | API Host setting |
| LibreChat | 34k | librechat.yaml custom endpoint |
| Chatbot UI | 29k | Custom API endpoint |
| Jan | 26k | Remote model config |
| AionUI | 18k | Custom API endpoint |
| HF Chat-UI | 8k | OPENAI_BASE_URL env var |
| big-AGI | 7k | Custom endpoint |
