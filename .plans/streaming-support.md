# Streaming LLM Response Support for Hermes Agent

## Overview

Add token-by-token streaming of LLM responses across all platforms. When enabled,
users see the response typing out live instead of waiting for the full generation.
Streaming is opt-in via config, defaults to off, and all existing non-streaming
code paths remain intact as the default.

## Design Principles

1. **Feature-flagged**: `streaming.enabled: true` in config.yaml. Off by default.
   When off, all existing code paths are unchanged — zero risk to current behavior.
2. **Callback-based**: A simple `stream_callback(text_delta: str)` function injected
   into AIAgent. The agent doesn't know or care what the consumer does with tokens.
3. **Graceful degradation**: If the provider doesn't support streaming, or streaming
   fails for any reason, silently fall back to the non-streaming path.
4. **Platform-agnostic core**: The streaming mechanism in AIAgent works the same
   regardless of whether the consumer is CLI, Telegram, Discord, or the API server.

---

## Architecture

```
                              stream_callback(delta)
                                    │
  ┌─────────────┐    ┌─────────────▼──────────────┐
  │  LLM API    │    │      queue.Queue()          │
  │  (stream)   │───►│  thread-safe bridge between │
  │             │    │  agent thread & consumer    │
  └─────────────┘    └─────────────┬──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │    CLI     │ │  Gateway  │ │ API Server│
              │ print to   │ │ edit msg  │ │ SSE event │
              │ terminal   │ │ on Tg/Dc  │ │ to client │
              └───────────┘ └───────────┘ └───────────┘
```

The agent runs in a thread. The callback puts tokens into a thread-safe queue.
Each consumer reads the queue in its own context (async task, main thread, etc.).

---

## Configuration

### config.yaml

```yaml
streaming:
  enabled: false          # Master switch. Default off.
  # Per-platform overrides (optional):
  # cli: true             # Override for CLI only
  # telegram: true        # Override for Telegram only
  # discord: false        # Keep Discord non-streaming
  # api_server: true      # Override for API server
```

### Environment variables

```
HERMES_STREAMING_ENABLED=true    # Master switch via env
```

### How the flag is read

- **CLI**: `load_cli_config()` reads `streaming.enabled`, sets env var. AIAgent
  checks at init time.
- **Gateway**: `_run_agent()` reads config, decides whether to pass
  `stream_callback` to the AIAgent constructor.
- **API server**: For Chat Completions `stream=true` requests, always uses streaming
  regardless of config (the client is explicitly requesting it). For non-stream
  requests, uses config.

### Precedence

1. API server: client's `stream` field overrides everything
2. Per-platform config override (e.g., `streaming.telegram: true`)
3. Master `streaming.enabled` flag
4. Default: off

---

## Implementation Plan

### Phase 1: Core streaming infrastructure in AIAgent

**File: run_agent.py**

#### 1a. Add stream_callback parameter to __init__ (~5 lines)

```python
def __init__(self, ..., stream_callback: callable = None, ...):
    self.stream_callback = stream_callback
```

No other init changes. The callback is optional — when None, everything
works exactly as before.

#### 1b. Add _run_streaming_chat_completion() method (~65 lines)

New method for Chat Completions API streaming:

```python
def _run_streaming_chat_completion(self, api_kwargs: dict):
    """Stream a chat completion, emitting text tokens via stream_callback.
    
    Returns a fake response object compatible with the non-streaming code path.
    Falls back to non-streaming on any error.
    """
    stream_kwargs = dict(api_kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    
    accumulated_content = []
    accumulated_tool_calls = {}  # index -> {id, name, arguments}
    final_usage = None
    
    try:
        stream = self.client.chat.completions.create(**stream_kwargs)
        
        for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk (final)
                if chunk.usage:
                    final_usage = chunk.usage
                continue
            
            delta = chunk.choices[0].delta
            
            # Text content — emit via callback
            if delta.content:
                accumulated_content.append(delta.content)
                if self.stream_callback:
                    try:
                        self.stream_callback(delta.content)
                    except Exception:
                        pass
            
            # Tool call deltas — accumulate silently
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "", "arguments": ""
                        }
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments
        
        # Build fake response compatible with existing code
        tool_calls = []
        for idx in sorted(accumulated_tool_calls):
            tc = accumulated_tool_calls[idx]
            if tc["name"]:
                tool_calls.append(SimpleNamespace(
                    id=tc["id"], type="function",
                    function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
                ))
        
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="".join(accumulated_content) or "",
                    tool_calls=tool_calls or None,
                    role="assistant",
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )],
            usage=final_usage,
            model=self.model,
        )
    
    except Exception as e:
        logger.debug("Streaming failed, falling back to non-streaming: %s", e)
        return self.client.chat.completions.create(**api_kwargs)
```

#### 1c. Modify _run_codex_stream() for Responses API (~10 lines)

The method already iterates the stream. Add callback emission:

```python
def _run_codex_stream(self, api_kwargs: dict):
    with self.client.responses.stream(**api_kwargs) as stream:
        for event in stream:
            # Emit text deltas if streaming callback is set
            if self.stream_callback and hasattr(event, 'type'):
                if event.type == 'response.output_text.delta':
                    try:
                        self.stream_callback(event.delta)
                    except Exception:
                        pass
        return stream.get_final_response()
```

#### 1d. Modify _interruptible_api_call() (~5 lines)

Add the streaming branch:

```python
def _call():
    try:
        if self.api_mode == "codex_responses":
            result["response"] = self._run_codex_stream(api_kwargs)
        elif self.stream_callback is not None:
            result["response"] = self._run_streaming_chat_completion(api_kwargs)
        else:
            result["response"] = self.client.chat.completions.create(**api_kwargs)
    except Exception as e:
        result["error"] = e
```

#### 1e. Signal end-of-stream to consumers (~5 lines)

After the API call returns, signal the callback that streaming is done
so consumers can finalize (remove cursor, close SSE, etc.):

```python
# In run_conversation(), after _interruptible_api_call returns:
if self.stream_callback:
    try:
        self.stream_callback(None)  # None = end of stream signal
    except Exception:
        pass
```

Consumers check: `if delta is None: finalize()`

**Tests for Phase 1:** (~150 lines)
- Test _run_streaming_chat_completion with mocked stream
- Test fallback to non-streaming on error
- Test tool_call accumulation during streaming
- Test stream_callback receives correct deltas
- Test None signal at end of stream
- Test streaming disabled when callback is None

---

### Phase 2: Gateway consumers (Telegram, Discord, etc.)

**File: gateway/run.py**

#### 2a. Read streaming config (~15 lines)

In `_run_agent()`, before creating the AIAgent:

```python
# Read streaming config
_streaming_enabled = False
try:
    # Check per-platform override first
    platform_key = source.platform.value if source.platform else ""
    _stream_cfg = {}  # loaded from config.yaml streaming section
    if _stream_cfg.get(platform_key) is not None:
        _streaming_enabled = bool(_stream_cfg[platform_key])
    else:
        _streaming_enabled = bool(_stream_cfg.get("enabled", False))
except Exception:
    pass
# Env var override
if os.getenv("HERMES_STREAMING_ENABLED", "").lower() in ("true", "1", "yes"):
    _streaming_enabled = True
```

#### 2b. Set up queue + callback (~15 lines)

```python
_stream_q = None
_stream_done = None
_stream_msg_id = [None]  # mutable ref for the async task

if _streaming_enabled:
    import queue as _q
    _stream_q = _q.Queue()
    _stream_done = threading.Event()
    
    def _on_token(delta):
        if delta is None:
            _stream_done.set()
        else:
            _stream_q.put(delta)
```

Pass `stream_callback=_on_token` to the AIAgent constructor.

#### 2c. Telegram/Discord stream preview task (~50 lines)

```python
async def stream_preview():
    """Progressively edit a message with streaming tokens."""
    if not _stream_q:
        return
    adapter = self.adapters.get(source.platform)
    if not adapter:
        return
    
    accumulated = []
    token_count = 0
    last_edit = 0.0
    MIN_TOKENS = 20          # Don't show until enough context
    EDIT_INTERVAL = 1.5      # Respect Telegram rate limits
    
    try:
        while not _stream_done.is_set():
            try:
                chunk = _stream_q.get(timeout=0.1)
                accumulated.append(chunk)
                token_count += 1
            except queue.Empty:
                continue
            
            now = time.monotonic()
            if token_count >= MIN_TOKENS and (now - last_edit) >= EDIT_INTERVAL:
                preview = "".join(accumulated) + " ▌"
                if _stream_msg_id[0] is None:
                    r = await adapter.send(
                        chat_id=source.chat_id,
                        content=preview,
                        metadata=_thread_metadata,
                    )
                    if r.success and r.message_id:
                        _stream_msg_id[0] = r.message_id
                else:
                    await adapter.edit_message(
                        chat_id=source.chat_id,
                        message_id=_stream_msg_id[0],
                        content=preview,
                    )
                last_edit = now
        
        # Drain remaining tokens
        while not _stream_q.empty():
            accumulated.append(_stream_q.get_nowait())
        
        # Final edit — remove cursor, show complete text
        if _stream_msg_id[0] and accumulated:
            await adapter.edit_message(
                chat_id=source.chat_id,
                message_id=_stream_msg_id[0],
                content="".join(accumulated),
            )
    
    except asyncio.CancelledError:
        # Clean up on cancel
        if _stream_msg_id[0] and accumulated:
            try:
                await adapter.edit_message(
                    chat_id=source.chat_id,
                    message_id=_stream_msg_id[0],
                    content="".join(accumulated),
                )
            except Exception:
                pass
    except Exception as e:
        logger.debug("stream_preview error: %s", e)
```

#### 2d. Skip final send if already streamed (~10 lines)

In `_process_message_background()` (base.py), after getting the response,
if streaming was active and `_stream_msg_id[0]` is set, the final response
was already delivered via progressive edits. Skip the normal `self.send()`
call to avoid duplicating the message.

This is the most delicate integration point — we need to communicate from
the gateway's `_run_agent` back to the base adapter's response sender that
the response was already delivered. Options:

- **Option A**: Return a special marker in the result dict:
  `result["_streamed_msg_id"] = _stream_msg_id[0]`
  The base adapter checks this and skips `send()`.
  
- **Option B**: Edit the already-sent message with the final response
  (which may differ slightly from accumulated tokens due to think-block
  stripping, etc.) and don't send a new one.

- **Option C**: The stream preview task handles the FULL final response
  (including any post-processing), and the handler returns None to skip
  the normal send path.

Recommended: **Option A** — cleanest separation. The result dict already
carries metadata; adding one more field is low-risk.

**Platform-specific considerations:**

| Platform | Edit support | Rate limits | Streaming approach |
|----------|-------------|-------------|-------------------|
| Telegram | ✅ edit_message_text | ~20 edits/min | Edit every 1.5s |
| Discord | ✅ message.edit | 5 edits/5s per message | Edit every 1.2s |
| Slack | ✅ chat.update | Tier 3 (~50/min) | Edit every 1.5s |
| WhatsApp | ❌ no edit support | N/A | Skip streaming, use normal path |
| HomeAssistant | ❌ no edit | N/A | Skip streaming |
| API Server | ✅ SSE native | No limit | Real SSE events |

WhatsApp and HomeAssistant fall back to non-streaming automatically because
they don't support message editing.

**Tests for Phase 2:** (~100 lines)
- Test stream_preview sends/edits correctly
- Test skip-final-send when streaming delivered
- Test WhatsApp/HA graceful fallback
- Test streaming disabled per-platform config
- Test thread_id metadata forwarded in stream messages

---

### Phase 3: CLI streaming

**File: cli.py**

#### 3a. Set up callback in the CLI chat loop (~20 lines)

In `_chat_once()` or wherever the agent is invoked:

```python
if streaming_enabled:
    _stream_q = queue.Queue()
    _stream_done = threading.Event()
    
    def _cli_stream_callback(delta):
        if delta is None:
            _stream_done.set()
        else:
            _stream_q.put(delta)
    
    agent.stream_callback = _cli_stream_callback
```

#### 3b. Token display thread/task (~30 lines)

Start a thread that reads the queue and prints tokens:

```python
def _stream_display():
    """Print tokens to terminal as they arrive."""
    first_token = True
    while not _stream_done.is_set():
        try:
            delta = _stream_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if first_token:
            # Print response box top border
            _cprint(f"\n{top}")
            first_token = False
        sys.stdout.write(delta)
        sys.stdout.flush()
    # Drain remaining
    while not _stream_q.empty():
        sys.stdout.write(_stream_q.get_nowait())
    sys.stdout.flush()
    # Print bottom border
    _cprint(f"\n\n{bot}")
```

**Integration challenge: prompt_toolkit**

The CLI uses prompt_toolkit which controls the terminal. Writing directly
to stdout while prompt_toolkit is active can cause display corruption.
The existing KawaiiSpinner already solves this by using prompt_toolkit's
`patch_stdout` context. The streaming display would need to do the same.

Alternative: use `_cprint()` for each token chunk (routes through
prompt_toolkit's renderer). But this might be slow for individual tokens.

Recommended approach: accumulate tokens in small batches (e.g., every 50ms)
and `_cprint()` the batch. This balances display responsiveness with
prompt_toolkit compatibility.

**Tests for Phase 3:** (~50 lines)
- Test CLI streaming callback setup
- Test response box borders with streaming
- Test fallback when streaming disabled

---

### Phase 4: API Server real streaming

**File: gateway/platforms/api_server.py**

Replace the pseudo-streaming `_write_sse_chat_completion()` with real
token-by-token SSE when the agent supports it.

#### 4a. Wire streaming callback for stream=true requests (~20 lines)

```python
if stream:
    _stream_q = queue.Queue()
    
    def _api_stream_callback(delta):
        _stream_q.put(delta)  # None = done
    
    # Pass callback to _run_agent
    result, usage = await self._run_agent(
        ..., stream_callback=_api_stream_callback,
    )
```

#### 4b. Real SSE writer (~40 lines)

```python
async def _write_real_sse(self, request, completion_id, model, stream_q):
    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    
    # Role chunk
    await response.write(...)
    
    # Stream content chunks as they arrive
    while True:
        try:
            delta = await asyncio.get_event_loop().run_in_executor(
                None, lambda: stream_q.get(timeout=0.1)
            )
        except queue.Empty:
            continue
        
        if delta is None:  # End of stream
            break
        
        chunk = {"id": completion_id, "object": "chat.completion.chunk", ...
                 "choices": [{"delta": {"content": delta}, ...}]}
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    
    # Finish + [DONE]
    await response.write(...)
    await response.write(b"data: [DONE]\n\n")
    return response
```

**Challenge: concurrent execution**

The agent runs in a thread executor. SSE writing happens in the async event
loop. The queue bridges them. But `_run_agent()` currently awaits the full
result before returning. For real streaming, we need to start the agent in
the background and stream tokens while it runs:

```python
# Start agent in background
agent_task = asyncio.create_task(self._run_agent_async(...))

# Stream tokens while agent runs
await self._write_real_sse(request, ..., stream_q)

# Agent is done by now (stream_q received None)
result, usage = await agent_task
```

This requires splitting `_run_agent` into an async version that doesn't
block waiting for the result, or running it in a separate task.

**Responses API SSE format:**

For `/v1/responses` with `stream=true`, the SSE events are different:

```
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}

event: response.completed  
data: {"type":"response.completed","response":{...}}
```

This needs a separate SSE writer that emits Responses API format events.

**Tests for Phase 4:** (~80 lines)
- Test real SSE streaming with mocked agent
- Test SSE event format (Chat Completions vs Responses)
- Test client disconnect during streaming
- Test fallback to pseudo-streaming when callback not available

---

## Integration Issues & Edge Cases

### 1. Tool calls during streaming

When the model returns tool calls instead of text, no text tokens are emitted.
The stream_callback is simply never called with text. After tools execute, the
next API call may produce the final text response — streaming picks up again.

The stream preview task needs to handle this: if no tokens arrive during a
tool-call round, don't send/edit any message. The tool progress messages
continue working as before.

### 2. Duplicate messages

The biggest risk: the agent sends the final response normally (via the
existing send path) AND the stream preview already showed it. The user
sees the response twice.

Prevention: when streaming is active and tokens were delivered, the final
response send must be suppressed. The `result["_streamed_msg_id"]` marker
tells the base adapter to skip its normal send.

### 3. Response post-processing

The final response may differ from the accumulated streamed tokens:
- Think block stripping (`<think>...</think>` removed)
- Trailing whitespace cleanup
- Tool result media tag appending

The stream preview shows raw tokens. The final edit should use the
post-processed version. This means the final edit (removing the cursor)
should use the post-processed `final_response`, not just the accumulated
stream text.

### 4. Context compression during streaming

If the agent triggers context compression mid-conversation, the streaming
tokens from BEFORE compression are from a different context than those
after. This isn't a problem in practice — compression happens between
API calls, not during streaming.

### 5. Interrupt during streaming

User sends a new message while streaming → interrupt. The stream is killed
(HTTP connection closed), accumulated tokens are shown as-is (no cursor),
and the interrupt message is processed normally. This is already handled by
`_interruptible_api_call` closing the client.

### 6. Multi-model / fallback

If the primary model fails and the agent falls back to a different model,
streaming state resets. The fallback call may or may not support streaming.
The graceful fallback in `_run_streaming_chat_completion` handles this.

### 7. Rate limiting on edits

Telegram: ~20 edits/minute (~1 every 3 seconds to be safe)
Discord: 5 edits per 5 seconds per message
Slack: ~50 API calls/minute

The 1.5s edit interval is conservative enough for all platforms. If we get
429 rate limit errors on edits, just skip that edit cycle and try next time.

---

## Files Changed Summary

| File | Phase | Changes |
|------|-------|---------|
| `run_agent.py` | 1 | +stream_callback param, +_run_streaming_chat_completion(), modify _run_codex_stream(), modify _interruptible_api_call() |
| `gateway/run.py` | 2 | +streaming config reader, +queue/callback setup, +stream_preview task, +skip-final-send logic |
| `gateway/platforms/base.py` | 2 | +check for _streamed_msg_id in response handler |
| `cli.py` | 3 | +streaming setup, +token display, +response box integration |
| `gateway/platforms/api_server.py` | 4 | +real SSE writer, +streaming callback wiring |
| `hermes_cli/config.py` | 1 | +streaming config defaults |
| `cli-config.yaml.example` | 1 | +streaming section |
| `tests/test_streaming.py` | 1-4 | NEW — ~380 lines of tests |

**Total new code**: ~500 lines across all phases
**Total test code**: ~380 lines

---

## Rollout Plan

1. **Phase 1** (core): Merge to main. Streaming disabled by default.
   Zero impact on existing behavior. Can be tested with env var.

2. **Phase 2** (gateway): Merge to main. Test on Telegram manually.
   Enable per-platform: `streaming.telegram: true` in config.

3. **Phase 3** (CLI): Merge to main. Test in terminal.
   Enable: `streaming.cli: true` or `streaming.enabled: true`.

4. **Phase 4** (API server): Merge to main. Test with Open WebUI.
   Auto-enabled when client sends `stream: true`.

Each phase is independently mergeable and testable. Streaming stays
off by default throughout. Once all phases are stable, consider
changing the default to enabled.

---

## Config Reference (final state)

```yaml
# config.yaml
streaming:
  enabled: false          # Master switch (default: off)
  cli: true               # Per-platform override
  telegram: true
  discord: true
  slack: true
  api_server: true        # API server always streams when client requests it
  edit_interval: 1.5      # Seconds between message edits (default: 1.5)
  min_tokens: 20          # Tokens before first display (default: 20)
```

```bash
# Environment variable override
HERMES_STREAMING_ENABLED=true
```
