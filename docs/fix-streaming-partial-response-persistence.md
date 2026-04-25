# Fix: Streaming Partial Response Lost on Page Refresh

## Problem

When the user refreshes the chat page while the AI is streaming a response, the partial (in-progress) assistant response was lost. Only messages already fully persisted to the database survived the refresh.

## Root Cause

The streaming text existed only in two places during generation:

1. **Frontend React state** (`ChatPage.tsx`) — lost on page refresh
2. **Agent in-memory variable** (`run_agent.py` `_current_streamed_assistant_text`) — never persisted until the full response completes

Additionally, the page had no mechanism to remember which session was active — on refresh, if the URL didn't contain `?resume=<id>`, the page would start a new empty session.

## Fix (Applied)

Three layers of defense ensure partial responses survive interruptions:

### Layer 1: Periodic Mid-Stream DB Flush

**File: `hermes_cli/web_server.py`** — `_flush_stream_partial()`, `_on_delta()` hook

The `_on_delta` WebSocket callback now accumulates streamed text into `stream_partial_text` and periodically flushes it to the database:

- Every ~80 characters of new content
- Every 0.5 seconds minimum interval
- Uses `_persist_interrupted_streaming_text()` to save with `finish_reason="interrupted"`
- `replace_interrupted_assistant_message()` handles dedup by UPDATE-ing existing partial rows

This means even if the browser crashes (not just refreshes), the partial text is already in the database.

### Layer 2: Disconnect-Time Save

**File: `hermes_cli/web_server.py`** — `_persist_interrupted_streaming_response()`

On WebSocket disconnect (page refresh, browser close, server shutdown), extracts the agent's `_current_streamed_assistant_text` and persists it before calling `interrupt()`. Handles three paths:

- `chat.interrupt` RPC (user clicks stop button)
- `WebSocketDisconnect` (page refresh / browser close)
- `asyncio.CancelledError` (server shutdown)

### Layer 3: Frontend Session Memory

**File: `web/src/pages/ChatPage.tsx`** — `rememberedChatSession()` / `rememberChatSession()` / `forgetChatSession()`

The session ID is persisted to `localStorage` (`hermes.chat.activeSessionId`). On page load:

1. Try URL `?resume=<id>` param first
2. Fall back to `localStorage` if no URL param
3. Load messages from DB for the resumed session
4. Browser URL is updated via `replaceState` so subsequent refreshes also work
5. "New session" button clears `localStorage` and starts fresh

### Agent Flush Dedup

**Files: `hermes_state.py` + `run_agent.py`**

When the agent thread eventually finishes and `_flush_messages_to_session_db()` runs, it calls `replace_interrupted_assistant_message()` instead of inserting a new row. This replaces the partial text saved by Layer 1/2 with the final complete response, preventing duplicate messages.

## Data Flow After Fix

```
Streaming starts:
  _on_delta("Hello") → stream_partial_text = "Hello"
  _on_delta(" world") → stream_partial_text = "Hello world"
  ... 80 chars accumulated ...
  _flush_stream_partial() → DB: assistant "Hello world..." (finish_reason=interrupted)

User refreshes page:
  → WebSocket disconnects
  → _persist_interrupted_streaming_response() saves full partial text to DB
  → agent.interrupt() called
  → New page loads
  → rememberedChatSession() reads session ID from localStorage
  → Fetches messages from DB → partial response IS there
  → Agent thread finishes, _flush_messages_to_session_db() runs
  → replace_interrupted_assistant_message() replaces partial with final → no duplicate
```

## Changed Files

### `hermes_cli/web_server.py`

- **`_persist_interrupted_streaming_text()`** — low-level helper: saves partial text to DB with dedup/replace logic
- **`_persist_interrupted_streaming_response()`** — extracts `_current_streamed_assistant_text` from running agent, delegates to above
- **`_on_delta()` hook** — accumulates `stream_partial_text`, calls `_flush_stream_partial()` every ~80 chars / 0.5s
- **Disconnect handlers** — all three exit paths (`WebSocketDisconnect`, `CancelledError`, `chat.interrupt`) call `_persist_interrupted_streaming_response()` before `agent.interrupt()`

### `hermes_state.py`

- **`replace_interrupted_assistant_message()`** — atomic UPDATE of an existing `finish_reason="interrupted"` row. Checks content prefix overlap, keeps the longer version. Full parameter support (tool_calls, reasoning, etc.).

### `run_agent.py`

- **`_flush_messages_to_session_db()`** — before INSERT, tries `replace_interrupted_assistant_message()` for assistant messages. If replaced, skips INSERT (`continue`). This prevents duplicates when the agent flushes after the WebSocket handler already saved the partial text.

### `web/src/pages/ChatPage.tsx`

- **`rememberedChatSession()`** / **`rememberChatSession()`** / **`forgetChatSession()`** — localStorage helpers for session ID persistence
- **`initSession()`** — tries URL `resume` param, then localStorage, to restore the previous session
- **`session.created` handler** — saves session ID to localStorage and updates browser URL
- **`startNewSession()`** — clears localStorage before creating a new session
