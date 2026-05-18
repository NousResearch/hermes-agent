# NATS Gateway Channel — Design Doc

**Status:** Draft (Phase 0 deliverable, pending review before any code changes)
**Scope:** Add NATS as a new gateway channel in Hermes Agent so callers can prompt the agent over NATS — send text, send attachments, and receive token-streamed responses — using the **NATS Agent Protocol v0.2**.
**Wire spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.2.0-draft).
**Agent-side SDK:** `natsagent` at `../synadia-agents/client-sdk/python` (the SDK now lives inside the `synadia-ai/synadia-agents` monorepo; `client-sdk/python` is its subtree).

Cross-references to the protocol spec are by section number (e.g. §5.6). Cross-references to the Hermes codebase use `file:line`.

---

## 1. Summary

Add `gateway/platforms/nats.py` as a new `BasePlatformAdapter` subclass. It registers one `natsagent.Agent` with the identity `agents.hermes.<owner>.<name>` at gateway startup; each inbound `prompt` is translated into a Hermes `MessageEvent`, routed through the normal gateway handler, streamed back chunk-by-chunk over NATS, and terminated by the SDK's empty-body terminator.

Session routing uses the caller-supplied envelope field `session` (protocol §5.1, optional string). Mid-stream approvals round-trip via `PromptStream.ask()`. Attachments round-trip base64 ↔ Hermes media cache. The adapter owns its own `AIAgent` construction and streaming pipeline (api_server-style), bypassing the gateway's `GatewayStreamConsumer` — see §6 for why.

Explicit non-goals: the future `attachments` endpoint (§5.5), JetStream at-least-once, E2E encryption, cross-platform adapter-level approval refactor. All are carried forward in §13.

---

## 2. Protocol ↔ Adapter mapping

| Direction | Protocol (v0.2)                                                   | Adapter surface                                                                                     |
|-----------|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Inbound   | `Envelope.prompt`                                                 | `MessageEvent.text`                                                                                  |
| Inbound   | `Envelope.attachments[i]` (base64)                                 | Decoded via `att.to_bytes()`, routed through `cache_{image,audio,document,video}_from_bytes` → `MessageEvent.media_urls` / `media_types` |
| Inbound   | `Envelope.session` (§5.1 optional field)                           | `SessionSource.chat_id` — opaque session string, default `"default"`                                 |
| Inbound   | `agents.hermes.<owner>.<name>` subject (prompt endpoint)           | Implicit — SDK dispatches to `Agent.on_prompt()`                                                     |
| Outbound  | `{type: response, data: "<text>"}` (§6.3 bare-string form)         | `stream.send(ResponseChunk(text=delta))` via adapter-local `stream_delta_callback`                   |
| Outbound  | `{type: response, data: {text, attachments: [...]}}`               | `send_image_file()` / `send_document()` / `send_voice()` / `send_video()` → `ResponseChunk(text=caption, attachments=[Attachment.from_path(...)])` |
| Outbound  | `{type: status, data: "ack"}` (§6.4)                               | Keep-alive emitted every ~20 s while the handler is silent (see §6.2)                                |
| Outbound  | `{type: query, data: {...}}` (§7.1)                                | `adapter.request_interaction()` → `stream.ask(prompt, timeout=…)`                                    |
| Outbound  | Empty-body terminator (§6.5)                                       | Emitted automatically by the SDK when `_on_prompt()` returns (see `agent.py:278`)                    |
| Outbound  | Error-headered frame + terminator (§9.3)                           | Raise from `_on_prompt()`; SDK calls `respond_error(...)` + terminator                               |
| Liveness  | `agents.hermes.<owner>.<name>.heartbeat` (§8)                      | Automatic, SDK-owned — we pass `heartbeat_interval_s` only                                           |
| Discovery | `$SRV.PING.agents`, `$SRV.INFO.agents[.{id}]` (§4) | Automatic — SDK registers as a NATS micro service                                                    |
| Cancel    | None (§6.7 — interest-based, no wire signal)                       | MVP: no detection. Agent runs to completion. Revisit post-MVP with a periodic `stream.ask("alive?")` |

### Key points

- **SDK owns**: subject construction (§2), service registration (§3), envelope parsing (§5), chunk wrapping (§6.2/6.3), terminator (§6.5), error frames (§9), heartbeat emission (§8).
- **SDK does NOT own**: `status:ack` keep-alive cadence (§6.4) — we must emit. Per the spec's recommended caller inactivity timeout (§6.6: 60 s), we keep silence below that by a comfortable margin.

---

## 3. Session model

**Design decision:** caller-supplied `session` envelope field, default `"default"`.

### Why a caller-supplied field

- Protocol §5.1 makes `session` an optional top-level envelope field. The `natsagent` SDK models it as `Envelope.session: str | None`.
- The protocol's `metadata.session` (§3.2) identifies the *agent instance*, not the *conversation*. Using a single agent instance with per-caller session scoping is cheaper than spawning one NATS registration per Hermes session (§3.3 allows it but it's heavyweight).
- Keeps the NATS wire shape compatible with `nats pub` plain-text testing (default falls through when no envelope field is set).

### Inbound translation

```text
envelope.session  (string or None)
     │
     ▼
chat_id = (envelope.session or "").strip() or session_default
     │
     ▼
SessionSource(
    platform=Platform.NATS,
    chat_id=chat_id,
    chat_type="dm",
    user_id=chat_id,          # opaque; same as chat_id for NATS DM semantics
    user_name=chat_id,        # no richer context on the wire
)
     │
     ▼
build_session_key(source, group_sessions_per_user=True)
  → "agent:main:nats:dm:<chat_id>"
```

This lands in exactly the same shape the rest of the gateway already handles — `handle_message()` on the base class does its usual routing, pending-message draining, and session bookkeeping without special-casing NATS.

### Reading `session`

The SDK exposes `session` as a first-class field on the parsed `Envelope`, so the adapter reads it directly:

```python
# gateway/platforms/nats.py::_on_prompt
chat_id = (envelope.session or "").strip() or self._session_default()
```

No raw-bytes re-parse, no private attribute access. Earlier drafts of this doc and pre-0.1.1 SDK versions required the adapter to peek `stream._request.data` because the SDK's pydantic model dropped unknown fields via `extra="ignore"`; that workaround shipped in Phases 4–8 and was removed once the SDK landed first-class session support.

### Interaction with `group_sessions_per_user`

NATS DMs have no concept of groups — every message is a DM by construction (`chat_type="dm"`). The `group_sessions_per_user` path in `build_session_key()` (`gateway/session.py:491`) only fires for `chat_type in ("group", "channel", "thread")`, so the setting is a no-op for NATS. We still forward it in `extra` to match the contract `_create_adapter()` enforces (`gateway/run.py:2717`).

---

## 4. Config surface

```yaml
# config.yaml
platforms:
  nats:
    enabled: true
    extra:
      # One of `servers` or `context` is required.
      servers: ["nats://127.0.0.1:4222"]
      # OR
      context: "local-nats"               # $NATS_CONFIG_HOME/context/<name>.json

      # Identity on the wire — produces subject agents.hermes.<owner>.<name>
      agent: hermes                       # §2 token; default "hermes"
      owner: rene                         # §2 token; required
      name: gateway                       # §2 token; required

      # Behavior tuning (all optional)
      session_default: "default"          # envelope.session fallback
      heartbeat_interval_s: 30            # §8.2 default
      max_payload: "1MB"                  # §2.1 endpoint metadata
      attachments_ok: true                # §2.1 endpoint metadata
      ack_keepalive_interval_s: 20        # adapter-local, not on the wire
```

### Validation (fails `_set_fatal_error(..., retryable=False)` in `__init__`)

- Exactly one of `servers` (non-empty list of strings) or `context` (non-empty string) is set.
- `owner` and `name` are non-empty strings conforming to §2.2 naming rules (the SDK's `AgentSubject.new()` enforces and sanitizes — but we fail fast with a readable error before construction).
- `max_payload` parses via `natsagent._bytes.parse_human_bytes()` (the SDK will crash on construction otherwise).
- `ack_keepalive_interval_s < 60` (leave headroom under §6.6's recommended 60 s caller inactivity timeout).

### Env var overrides (`_apply_env_overrides()` in `gateway/config.py`)

| Env var               | Overrides                        | Notes                                                      |
|-----------------------|----------------------------------|------------------------------------------------------------|
| `NATS_URL`            | `extra.servers` (single-URL list) | Canonical env name in the NATS ecosystem                   |
| `NATS_CONTEXT`        | `extra.context`                   | Matches `natsagent.connect(context=…)` semantics            |
| `HERMES_NATS_AGENT`   | `extra.agent`                     | Optional; rarely overridden                                 |
| `HERMES_NATS_OWNER`   | `extra.owner`                     | Common in multi-tenant deployments                          |
| `HERMES_NATS_NAME`    | `extra.name`                      | Common when running multiple Hermes profiles                |
| `HERMES_NATS_SESSION` | `extra.session_default`           | Rarely used; mostly for tests                               |

Pattern mirrors Signal (`gateway/config.py:926-943`): if the env var is set, ensure the platform entry exists, set `enabled=True`, and `update()` the `extra` dict.

### `get_connected_platforms()` rule

Add an arm:

```python
elif platform == Platform.NATS and (
    config.extra.get("servers") or config.extra.get("context")
):
    connected.append(platform)
```

Matches the existing "enabled AND has creds" pattern.

---

## 5. Profile isolation (scoped lock)

Two Hermes profiles trying to register the **same** `(agent, owner, name)` would land on the same NATS inbox subject and both receive load-balanced prompts — silently wrong.

**Lock scope:** `"nats"`
**Lock identity:** `f"{agent}:{owner}:{name}"`

```python
# in connect(), before natsagent.connect(...)
if not self._acquire_platform_lock("nats", f"{agent}:{owner}:{name}", "NATS agent identity"):
    return False
```

`_acquire_platform_lock` (base.py:986) wraps `gateway.status.acquire_scoped_lock()`. Released in `disconnect()` via `_release_platform_lock()`. Telegram (`telegram.py:667`) is the canonical reference.

The lock is **machine-local**. Cross-machine collisions (two hosts, same identity, same NATS cluster) are not prevented — the protocol explicitly permits multiple instances per identity (§3.3), so we don't treat this as an error. The local lock only prevents the "two profiles on one laptop" footgun.

---

## 6. Streaming architecture

### 6.1 Pattern choice: api_server-style, not telegram-style

The gateway's default streaming path runs every delta through `GatewayStreamConsumer` (`gateway/stream_consumer.py`) which buffers, rate-limits, and progressively **edits a single message** via `adapter.edit_message()`. That's the right model for Telegram/Slack/Discord (one chat message, keep editing).

NATS is request/reply: every chunk is a separate publish to the caller's reply subject. Progressive edits are meaningless. Using the default consumer path would serialize every delta through a needless edit buffer *and* require us to fake out `edit_message` semantics.

`gateway/platforms/api_server.py` already solves this exact shape for SSE — the adapter owns its own `AIAgent` construction (`api_server.py:704-755`) and the stream callback is a closure over a local queue (`api_server.py:918-929`). We mirror that pattern.

### 6.2 Per-prompt lifecycle

```text
NATS msg on agents.hermes.<owner>.<name>
         │
         ▼
 SDK decodes envelope → calls adapter._on_prompt(envelope, stream)
         │
         ├─── read envelope.session → chat_id
         ├─── decode attachments → cache_* → media_urls/media_types
         ├─── build MessageEvent
         │
         ├─── self._active_streams[chat_id] = stream     ────┐
         ├─── start keep-alive task (emits {type:status,   │
         │         data:"ack"} every 20 s if queue silent) │
         │                                                 │
         ├─── delta_queue: asyncio.Queue[str|None]         │
         ├─── pump_task: drains delta_queue, calls         │
         │         stream.send(ResponseChunk(text=delta))  │
         │                                                 │
         ├─── agent = build_nats_agent(session_id=chat_id, │
         │         stream_delta_callback=_queue_delta,     │
         │         approval_callback=_nats_approval)       │
         │                                                 │
         ├─── run agent in executor; await completion     │
         │                                                 │
         ├─── delta_queue.put(None)  # stop pump          │
         ├─── await pump_task                             │
         ├─── stop keep-alive task                        │
         └─── self._active_streams.pop(chat_id)   ────────┘
         │
         ▼
 SDK returns from _on_prompt_request → emits empty terminator (§6.5)
```

### 6.3 Delta ordering

Deltas arrive on the agent's worker thread; `stream.send` must be awaited from the event-loop thread. We use `asyncio.Queue` with `loop.call_soon_threadsafe(queue.put_nowait, delta)` inside the sync callback:

```python
def _queue_delta(delta: str | None) -> None:
    loop.call_soon_threadsafe(delta_queue.put_nowait, delta if delta else _SENTINEL_NEWLINE)
```

The pump task preserves publication order (FIFO queue, single consumer), matching §6.6 ("Chunks are delivered in publication order").

### 6.4 `status:ack` keep-alive

MVP: fixed 20 s tick independent of activity. Reasoning:

- Protocol §6.6 recommends callers default to 60 s inactivity timeout. 20 s is 3× headroom.
- Simpler than activity-aware (no race between "just sent a delta" and "tick timer fired"); worst case is occasional redundant acks.

Revisit if caller-side logs get noisy. The 20 s default is a config knob (`ack_keepalive_interval_s`) for exactly that reason.

### 6.5 Error propagation

Exceptions inside `_on_prompt()` propagate up to the SDK's `_on_prompt_request()` (`agent.py:256`), which calls `request.respond_error("500", ...)` and then the terminator. We don't need to do anything special — we DO need to make sure our own error paths (e.g., attachment-decode failures) either raise or handle-and-return cleanly without leaking a half-streamed response.

Specific cases handled explicitly:

| Failure                          | Handling                                                                    |
|----------------------------------|-----------------------------------------------------------------------------|
| `envelope.session` is empty/blank | Fall back to `session_default` — matches spec §5.1 "optional" semantics     |
| Attachment base64 invalid        | Raise `ProtocolError` → SDK responds 400                                    |
| `attachments_ok=false` + atts    | Raise — SDK responds 400 per §5.4                                           |
| Envelope > `max_payload`         | SDK caller-side enforcement (§5.4). Agent-side enforcement deferred (§13)   |
| Handler raises                   | SDK responds 500 + terminator                                                |
| NATS disconnect mid-stream       | Pump task's `stream.send` raises; we log and return. SDK emits error frame. |
| Caller drops subscription (§6.7) | No detection; agent runs to completion (MVP)                                 |

---

## 7. Mid-stream queries (NATS-local approval wiring)

### 7.1 Current state

The gateway has a `_pending_approvals` dict and an `_approval_notify_sync()` callback (`gateway/run.py:9857-9922`). For most adapters, the notify path calls `adapter.send()` with "/approve yes|no" instructions and waits for a follow-up text message. This path is effectively non-functional today — none of the messaging platforms consistently round-trip approval replies.

`_approval_notify_sync()` is a bridge built on `tools/approval.py`'s `_gateway_queues[session_key]`: when the agent thread hits a dangerous-command prompt, it parks on an `entry.event.wait()`. The notify callback must eventually cause `resolve_gateway_approval(session_key, choice)` to be called, which sets the event and unblocks the agent.

### 7.2 NATS decision: new capability-gated adapter hook

Scope explicitly excluded from this MVP: a cross-platform adapter refactor. We add exactly one new *opt-in* method on `BasePlatformAdapter`:

```python
# gateway/platforms/base.py (addition)
async def request_interaction(
    self,
    chat_id: str,
    prompt: str,
    *,
    kind: str,           # "approval" | "clarification" | future kinds
    timeout: float,
) -> str | None:
    """Prompt the caller mid-conversation and return their reply.

    Default implementation raises NotImplementedError; the gateway
    detects capability via getattr(type(adapter), "request_interaction",
    BasePlatformAdapter.request_interaction) is not the base default.
    """
    raise NotImplementedError("request_interaction not supported on this adapter")
```

NATS implementation:

```python
# gateway/platforms/nats.py
async def request_interaction(self, chat_id, prompt, *, kind, timeout):
    stream = self._active_streams.get(chat_id)
    if stream is None:
        raise RuntimeError(f"no active NATS stream for chat_id={chat_id}")
    try:
        reply = await stream.ask(prompt, timeout=timeout)
    except natsagent.QueryTimeout:
        return None
    return reply.prompt
```

### 7.3 Gateway wiring (T6.3)

Inside `_approval_notify_sync()` (or its async sibling), check whether the adapter defines `request_interaction`:

```python
# in gateway/run.py
has_capability = type(adapter).request_interaction is not BasePlatformAdapter.request_interaction
if has_capability:
    # run the adapter hook on the gateway loop; on reply, call resolve_gateway_approval()
    ...
else:
    # existing behavior preserved (the /approve text-reply dance)
    ...
```

**Critically:** existing adapters inherit the base's `NotImplementedError` and therefore fall into the else branch — behavior is preserved bit-for-bit.

### 7.4 Query chunk attachments

Out of scope for MVP. `request_interaction(prompt: str)` is string-only. If a future approval wants to attach a diff, we extend the signature.

---

## 8. Attachments round-trip

### 8.1 Inbound (caller → agent)

```text
envelope.attachments[i]   # {"filename": "...", "content": "<base64>"}
         │
         ├── filename:    "report.pdf"
         └── content:     base64-decoded → bytes
                              │
                              ▼
         route by extension (re-use existing helpers in base.py):
           .png .jpg .jpeg .gif .webp      → cache_image_from_bytes(...)
           .wav .mp3 .m4a .ogg .flac       → cache_audio_from_bytes(...)
           .mp4 .mov .webm                  → cache_video_from_bytes(...)
           *                                → cache_document_from_bytes(...)
                              │
                              ▼
         MessageEvent.media_urls = [cached_path, ...]
         MessageEvent.media_types = [MessageType.PHOTO | AUDIO | VIDEO | DOCUMENT, ...]
         MessageEvent.message_type = <first media type, or TEXT if no media>
```

Extension routing is deliberately simple for MVP — no content sniffing. If the filename is ambiguous (e.g. `.bin`) we treat it as DOCUMENT. Protocol §5.2 says "Agents interpret the bytes by extension or content sniff" so extension-only is compliant.

### 8.2 Outbound (agent → caller)

Hermes tools express attached outputs via `MEDIA:/path` markup in response text (or by calling `adapter.send_image_file()` etc. directly). We override each of those:

```python
# send_image_file, send_document, send_voice, send_video all follow this shape
async def send_image_file(self, chat_id, image_path, caption=None, **_):
    stream = self._active_streams.get(chat_id)
    if stream is None:
        return SendResult(success=False, error="no active NATS stream")
    attachment = Attachment.from_path(image_path)
    await stream.send(ResponseChunk(text=caption or "", attachments=[attachment]))
    return SendResult(success=True, message_id=str(uuid.uuid4()))
```

The `send()` method itself is purely for text chunks, looking up the stream the same way.

### 8.3 `message_id` semantics

NATS has no per-message identifier on the wire. We return a fresh UUID for each `SendResult.message_id` to satisfy callers that thread reply-ids. `edit_message()` is **not supported** on NATS (protocol has no edit semantics); we inherit the base's `SendResult(success=False, error="Not supported")`.

---

## 9. Lifecycle diagram

### Gateway startup

```
gateway/run.py:_create_adapter(Platform.NATS)
        │
        ▼
NatsAdapter.__init__(config)
        │
        ├── parse config.extra into NatsAdapterSettings
        ├── validate (fatal if bad)
        └── store settings; do NOT connect yet
        │
        ▼
GatewayRunner.connect_all() → await adapter.connect()
        │
        ├── _acquire_platform_lock("nats", f"{agent}:{owner}:{name}", ...)
        ├── self._nc = await natsagent.connect(servers=..., context=...)
        ├── self._agent = natsagent.Agent(agent=..., owner=..., name=..., nc=self._nc, ...)
        ├── self._agent.on_prompt(self._on_prompt)
        ├── await self._agent.start()    # registers micro service + starts heartbeat
        └── self._mark_connected()
```

### Per-prompt

```
natsagent SDK receives NATS msg
        │
        ▼
Agent._on_prompt_request → decode envelope → PromptStream → handler
        │
        ▼
NatsAdapter._on_prompt(envelope, stream)
        │
        ├── chat_id = envelope.session or settings.session_default
        ├── media = decode_attachments(envelope.attachments)
        ├── event  = MessageEvent(text=envelope.prompt, source=..., media_urls=..., ...)
        │
        ├── register stream in self._active_streams[chat_id]
        ├── start keep-alive + pump tasks
        │
        ├── agent = self._build_nats_agent(
        │         session_id=build_session_key(source),
        │         stream_delta_callback=self._queue_delta_for(chat_id),
        │         # approval_callback hook defaults to base's _pending_approvals
        │         # flow; the new request_interaction path is triggered on the
        │         # gateway side in run.py (T6.3).
        │     )
        ├── await self.handle_message(event)
        │     └── which eventually drives agent.run_conversation in an executor
        │
        ├── await pump_task (drains remaining deltas)
        ├── stop keep-alive
        └── pop stream from self._active_streams
        │
        ▼
SDK emits empty terminator (§6.5). Caller's async-iterator exits.
```

### Shutdown

```
GatewayRunner.disconnect_all() → await adapter.disconnect()
        │
        ├── signal cancellation to in-flight _on_prompt handlers
        │     (via asyncio.Event or task cancellation)
        ├── await all outstanding pump / keep-alive / _on_prompt tasks
        ├── await self._agent.stop()     # stops heartbeat + deregisters micro service
        ├── await self._nc.close()
        ├── _release_platform_lock()
        └── _mark_disconnected()
```

---

## 10. Slash commands

Slash commands arrive to the gateway as `MessageEvent(message_type=COMMAND)` — `handle_message()` routes them through the existing `COMMAND_REGISTRY` / `command.dispatch()` path. No adapter-side code changes required for the dispatch itself.

Commands available over NATS: all `cli_only=False` entries in `hermes_cli/commands.py::COMMAND_REGISTRY`, same as every other gateway channel — `/new`, `/reset`, `/model`, `/status`, `/stop`, `/help`, `/compress`, `/resume`, etc. (T7.1 verifies this list.)

Output of commands lands in `stream.send(ResponseChunk(text=...))` the same way ordinary agent output does. `/help` is plain text, which the NATS wire is indifferent to.

---

## 11. Failure modes

| Scenario                                  | Detection                                 | Response                                                                |
|-------------------------------------------|-------------------------------------------|-------------------------------------------------------------------------|
| NATS server unreachable at connect        | `natsagent.connect` raises                | `_set_fatal_error("nats_connect_error", ..., retryable=True)`; gateway may retry |
| Identity already locked on this host      | `_acquire_platform_lock` returns False    | `_set_fatal_error("nats_lock", ..., retryable=False)`; do not retry      |
| NATS reconnect mid-stream                 | `stream.send` raises inside pump          | Log; agent continues to completion; SDK emits error frame                |
| Caller drops reply subscription (§6.7)    | No detection in MVP                        | Agent runs to completion; published chunks dropped by NATS server        |
| Oversize inbound envelope                 | `natsagent`'s caller-side §5.4 + our check | Reject before cache_*; raise → SDK `respond_error(400)`                  |
| Handler raises unexpectedly               | SDK wraps exception                        | `respond_error(500, <sanitized desc>)` then terminator                   |
| `envelope.session` empty/blank/None       | SDK's pydantic field (`str | None`)        | Fall back to `session_default`                                           |
| `max_payload` format bad at init          | `natsagent._bytes.parse_human_bytes` raise | `_set_fatal_error(...)` during `__init__`                                |
| `owner`/`name` violate §2.2               | `AgentSubject.new()` raise                  | `_set_fatal_error(...)` during `connect()`                               |
| Two Hermes profiles, same identity        | Lock collision                             | Second fails fast with actionable message (`telegram.py` precedent)      |

---

## 12. Testing strategy

Tests use `scripts/run_tests.sh` (hermetic wrapper). Mirror the Telegram collection-time mock so the suite runs without `natsagent` installed:

```python
# tests/gateway/conftest.py (addition)
def _ensure_natsagent_mock() -> None:
    if "natsagent" in sys.modules and hasattr(sys.modules["natsagent"], "__file__"):
        return
    mod = MagicMock()

    # Connection factory
    mod.connect = AsyncMock()

    # Agent / PromptStream — support await usage in tests
    mod.Agent = MagicMock()
    mod.Agent.return_value.start = AsyncMock()
    mod.Agent.return_value.stop = AsyncMock()
    mod.PromptStream = MagicMock()

    # Exception types (real classes so `except` works)
    mod.QueryTimeout = type("QueryTimeout", (Exception,), {})
    mod.ProtocolError = type("ProtocolError", (Exception,), {})

    # Envelope / Attachment / chunks — pydantic-ish stand-ins
    class _FakeAttachment:
        def __init__(self, filename: str, content: str): ...
        def to_bytes(self) -> bytes: ...
        @classmethod
        def from_path(cls, p): ...
        @classmethod
        def from_bytes(cls, filename, data): ...
    mod.Attachment = _FakeAttachment

    sys.modules["natsagent"] = mod

_ensure_natsagent_mock()
```

Test files (one file per table row):

| File                             | Covers                                                     |
|----------------------------------|-----------------------------------------------------------|
| `test_nats_config.py`            | T2.2 — parse config.extra, env overrides, bad config      |
| `test_nats_connect.py`           | T3.4 — connect/disconnect, lock, handler registration     |
| `test_nats_inbound.py`           | T4.4 — envelope → MessageEvent, deltas out, keep-alive    |
| `test_nats_outbound.py`          | T5.5 — image/doc/voice → ResponseChunk(attachments=...)   |
| `test_nats_query.py`             | T6.4 — approval → stream.ask round-trip → agent unblocks  |

Integration smoke (T8.*) uses the real SDK against a local `nats-server` — these are manual, not in `scripts/run_tests.sh`.

---

## 13. Explicit non-goals (for this MVP)

1. **Future `attachments` endpoint (§5.5)** — v0.1 inline base64 only. Attachment handling is factored so the future chunked-upload endpoint is additive.
2. **JetStream at-least-once** — core NATS only; per-chunk loss is permitted by §6.6.
3. **E2E encryption / agent identity** — delegated to NATS server auth (§10.1).
4. **Cross-platform adapter-level approval refactor** — NATS-local via new `request_interaction` hook (§7.2). Other platforms inherit the base's `NotImplementedError` and keep their current behavior.
5. **Cancellation detection (§6.7)** — MVP runs to completion on caller drop. A periodic `stream.ask("alive?")` liveness probe is a candidate follow-up.
6. **Server-side `max_payload` enforcement** — trust the SDK's caller-side check; revisit if abuse is observed.
7. **Activity-aware `status:ack` cadence** — fixed 20 s tick. Revisit if caller logs get noisy.
8. **`envelope.session` rate-limiting / validation** — trust NATS account-level auth to gate publishers.
9. **Per-session `natsagent.Agent` registration** — one registration per Hermes instance; `envelope.session` in the envelope distinguishes conversations (§3.3 would allow the alternative but it's heavyweight).

---

## 14. Local smoke-test recipe (feeds T8.*)

Installation (one-time):

```bash
source venv/bin/activate
pip install -e ../synadia-agents/client-sdk/python   # while natsagent is not yet on PyPI
```

Local broker:

```bash
nats-server -DV        # or use a Synadia Cloud context
```

Hermes config:

```yaml
platforms:
  nats:
    enabled: true
    extra:
      servers: ["nats://127.0.0.1:4222"]
      agent: hermes
      owner: rene
      name: gateway
      heartbeat_interval_s: 30
      max_payload: "1MB"
      attachments_ok: true
```

Start the gateway:

```bash
hermes gateway
```

Verify (from `../synadia-agents/client-sdk/python`):

```bash
uv run python examples/01-discover.py --url nats://127.0.0.1:4222
# expect: agents.hermes.rene.gateway in the output

uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 "what is 2+2"
# expect: streamed response, terminator

uv run python examples/03-prompt-attachment.py --url nats://127.0.0.1:4222 path/to/test.pdf "summarize"
# expect: hermes ingests the PDF and streams a summary

uv run python examples/04-query-reply.py --url nats://127.0.0.1:4222
# expect: Query chunk, reply "yes", stream resumes

uv run python examples/05-liveness.py --url nats://127.0.0.1:4222
# expect: heartbeats visible; kill hermes; liveness flips False after 3× interval

nats req '$SRV.INFO.agents' '' --replies=0 --timeout=2s
nats sub 'agents.hermes.*.*.heartbeat'
# expect: protocol Appendix C interop confirmed
```

---

## 15. Critical cross-references

| Area                           | Hermes file                                          | Notes                                                |
|--------------------------------|------------------------------------------------------|------------------------------------------------------|
| Abstract adapter contract      | `gateway/platforms/base.py`                          | Add optional `request_interaction` (T6.2)            |
| Reference adapter (locks)      | `gateway/platforms/telegram.py:667, 916`             | Canonical `_acquire_platform_lock` + disconnect flow |
| Reference adapter (streaming)  | `gateway/platforms/api_server.py:704, 920, 2594`     | `_on_delta` + adapter-owned `AIAgent` construction   |
| Adapter factory                | `gateway/run.py:2717`                                | Add `elif Platform.NATS:` branch                     |
| Stream delta callback wiring   | `gateway/run.py:9614, 9715`                          | (Default path — NATS bypasses this)                  |
| Approval callback surface      | `gateway/run.py:9857-9922, 9979`                     | T6.3 adds capability check here                      |
| Platform enum + env overrides  | `gateway/config.py`                                  | Add `Platform.NATS` + env block                      |
| Connection gate                | `gateway/config.py::get_connected_platforms`         | Add NATS arm                                         |
| Scoped lock primitive          | `gateway/status.py::acquire_scoped_lock`             | Signature: `(scope, identity, metadata=None)`        |
| Session source + key           | `gateway/session.py`                                 | `SessionSource`, `build_session_key`                 |
| Test mock pattern              | `tests/gateway/conftest.py::_ensure_telegram_mock`   | Mirror for `_ensure_natsagent_mock`                  |
| CLI approval callback          | `hermes_cli/callbacks.py` (reference only)           | CLI-only; gateway uses its own notify bridge         |
| SDK source                     | `../synadia-agents/client-sdk/python/src/natsagent/` | `agent.py`, `envelope.py`, `messages.py`, `connect.py` |
| SDK examples                   | `../synadia-agents/client-sdk/python/examples/01..05-*.py` | Smoke-test inputs (§14)                        |
| Protocol spec                  | `../nats-agent-sdk-docs/core-protocol.md`            | v0.2.0-draft                                         |

---

## 16. Decision log (for reviewer)

- **Session identity is `envelope.session` in the envelope, not `metadata.session`** on the registration. Keeps one registration per Hermes instance (§3.3 would allow per-session, but the overhead isn't justified).
- **Default `envelope.session` is `"default"`** — matches Appendix C guidance for session-less harnesses (`openclaw`) and preserves `nats pub` plain-text testability.
- **NATS bypasses `GatewayStreamConsumer`** — it's designed for edit-based transports; NATS semantics (each chunk is a separate publish) don't match. Follow `api_server.py` pattern instead.
- **New `request_interaction` hook is capability-gated**, not a cross-platform refactor. Non-NATS adapters keep their current (mostly non-functional) approval flow unchanged.
- **`status:ack` is a fixed 20 s tick** for MVP simplicity; knob exists for later activity-aware tuning.
- **Parse `envelope.session` from raw bytes** (option (b) in §3) as a local workaround; upstream the raw-handle to the SDK as follow-up.
- **Lock scope is `"nats"` + `"{agent}:{owner}:{name}"`** identity — matches the resource being guarded (the wire subject), not the server URL.

Open for reviewer input:

1. Should the SDK change (adding raw-bytes access to the handler) be upstreamed before we ship, or after? (Recommendation: after; use option (b) now.)
2. Should the adapter register a single `Agent` or one per active `envelope.session`? (Recommendation: single; rationale in §13(9).)
3. Do we expose `ack_keepalive_interval_s` in config, or hard-code? (Recommendation: expose, with 20 s default.)
4. Do we want a metric/log for "prompt received but session mismatched expectations" (e.g. envelope-parse failures with JSON-looking body)? (Recommendation: log at warning; no metric in MVP.)

---

## 17. Lessons learned (written post-MVP)

This section is intentionally written in past tense, as a retrospective. Each lesson maps back to a decision-log entry in `docs/nats-gateway-progress.md` if you want the moment-of-landing context. Listed roughly in order of broadest applicability → most NATS-specific.

### 17.1 Contextvars do not cross `run_coroutine_threadsafe` boundaries

`asyncio.run_coroutine_threadsafe(coro, loop)` schedules `coro` into a **fresh** context on the target loop. Every contextvar set on the calling thread is invisible inside `coro`. Three separate Phase 6 bugs were the same bug under different names:

- The stream-resolution contextvar (`_current_stream`) set in `_on_prompt` was unreadable from inside the coroutine that the approval-notify callback scheduled on the gateway loop.
- The session-key contextvar (`set_current_session_key`) set on the gateway's async side was unreadable from the executor thread running `run_conversation`.
- The approval-entry-id contextvar (`_current_approval_entry_id`) set by `check_all_command_guards` on the agent thread was unreadable inside the coroutine scheduled by the adapter's notify callback.

Fixes in all three cases were identical in shape: **capture the contextvar value synchronously on the calling side, before scheduling; pass the captured value explicitly through the closure**. `gateway/run.py::_run_in_executor_with_context` uses `copy_context()` to propagate context **into** executor threads — that direction works. There is no `copy_context()` analog for `run_coroutine_threadsafe` and upstream has no plans to add one.

**Generalizable rule.** Any sync→async or async→sync handoff that isn't `asyncio.create_task` on the same loop is a contextvar boundary. Treat it as the default assumption when planning approval, streaming, or tool-dispatch flows that cross threads.

### 17.2 Prefer structural elimination of races over reconciling them

Phase 5's T5.0 shipped a contextvar-primary + compound-key-dict hybrid for concurrent same-session stream lookup. It was correct under most scenarios but fragile — e.g. sends scheduled across `run_coroutine_threadsafe` boundaries fell through to the dict-lookup fallback, which was order-dependent when multiple streams shared a `chat_id`. Phase 6 initially stacked a second reconciliation layer on top (closure-captured streams inside notify callbacks) and a third (`register_gateway_notify` overwrite semantics).

The third attempt replaced all of it with a per-`chat_id` `asyncio.Lock`: at most one handler per session at any instant, so stream-ambiguity and callback-overwrite both became structurally impossible. The Phase 5 T5.0 regression test — which manually gated two handlers to overlap and asserted each send reached its own stream — became **deadlock-by-design under serialization** and was rewritten as a timeline-ordering assertion (A enter → A leave → B enter → B leave).

**Generalizable rule.** When a correctness bug has the shape "multiple concurrent things of type X share state Y," serialize the entry to X before reconciling Y correctly. Losing a test that exercises "race-safe" machinery in favor of a test that exercises "no race possible" is a win. The opposite direction — retrofitting serialization after building reconciliation — is much more expensive because the reconciliation machinery spreads across the code.

### 17.3 Adapter-owned `AIAgent` bypasses more of the gateway than it looks

The api_server pattern (own the agent, feed a stream_delta_callback into a queue) was the right architectural call for NATS (§6.1). But every time the adapter sidesteps `GatewayRunner._handle_message()`, something that runs inside the default path silently stops running. In Phase 8, two separate gaps surfaced after the adapter was already shipping:

- Media enrichment (`_enrich_message_with_vision` + document/audio context notes) ran in `_handle_message`, not in the adapter. Without it, uploaded images were cached but never mentioned to the agent — the agent literally replied "I don't see an image attached" on a PNG round-trip. Fix: `NatsAdapter._enrich_event_with_media` replicates the canonical template byte-for-byte.
- Approval-notify registration (`register_gateway_notify` inside the async side of `_handle_message`) didn't run for the adapter-owned path. Without it, dangerous-command approvals hung the agent thread for the 300 s framework timeout. Fix: adapter-local `register_gateway_notify` inside `_run_agent_sync`.

**Generalizable rule.** When designing an adapter that bypasses `handle_message`, audit every side effect that `_handle_message` triggers before the `AIAgent.run_conversation` call — media enrichment, approval registration, system-prompt hint injection, session-source threading, whatever. Each is a latent gap unless replicated on the adapter hot path. The symptom is always the same: "works in unit tests, caller observes feature silently absent in integration."

### 17.4 Keep cross-adapter user-message shape identical

Phase 8's first-pass fix for dropped media_urls used a bracketed note pointing the agent at `vision_analyze` for the user to call itself. It worked for the MVP §8.1 contract but diverged from every other platform's behavior. The refactor that followed adopted the canonical template verbatim (inline `vision_analyze` → description injected into user_message with the exact same wording Telegram / Discord / Slack use).

Reason: any downstream behavior that assumes a consistent user-message shape — skill prompts, conversation-history replay, session migration across platforms, memory formation — silently miscomputes if one platform's messages look different from the others. "Works on Telegram, broken on NATS" is a whole class of future bugs; matching byte-for-byte removes it.

**Generalizable rule.** Adapter hot-path transformations of the user-facing message are the right place for platform-specific behavior (formatting, attachment handling, emoji policy). Adapter-side construction of the agent-facing message should remain indistinguishable across adapters unless the platform genuinely exposes richer structure.

### 17.5 Cross-module registration surfaces need consistency tests in the phase-close gate

The `hermes-nats` toolset existed in `toolsets.py`'s `TOOLSETS["hermes-nats"]` and in `hermes_cli/platforms.py`'s `PLATFORMS` map from Phase 4 onward, but was missing from the `TOOLSETS["hermes-gateway"]["includes"]` aggregator list. The gap survived three subsequent phases because each phase's file-targeted test subset (`scripts/run_tests.sh tests/gateway/test_nats_*.py`) didn't exercise the consistency test that lives in a different subtree (`tests/hermes_cli/test_tools_config.py`).

**Generalizable rule.** Running the full suite every phase is overkill at 14k+ tests and 4 min wall time per cycle. But whenever a phase touches a cross-module **registration point** — a `Platform` enum value, a toolset name, an env var the factory reads, a platform label in the CLI registry — run the consistency test for that surface before closing the phase. For platforms specifically, `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` is <1 s and would have caught the miss.

### 17.6 `notify_cb` is sync and can fire multiple times per session

`tools/approval.py::register_gateway_notify` wires `cb(approval_data: dict) -> None`. The registered callback runs on the agent's worker thread, synchronously, and is expected to return quickly so the agent thread can proceed to `entry.event.wait()`. Three implications caused bugs during Phase 6:

1. Any async work from the callback has to cross threads via `run_coroutine_threadsafe` — which re-enters §17.1's contextvar problem.
2. A single callback registration can be invoked multiple times per session (parallel subagents fan out from one `delegate_tool` call). Per-invocation state must live in `approval_data`, not a closure over the registration.
3. Every path through the callback must eventually unblock the waiting entry. Either via the scheduled async work that calls `resolve_gateway_approval`, or by calling `resolve_gateway_approval(session_key, "deny")` directly on failure. Missing this made one dispatch-failure scenario hang the agent for the full 300 s `gateway_timeout` before the framework surfaced it as "deny" anyway — same outcome, 300 s → ~0 ms after the fix.

### 17.7 Attachment enrichment was a canonical-template match, not a design trade-off

The first-pass media-enrichment fix (adapter-local bracketed notes pointing the agent at `vision_analyze`) was cheaper to ship than inline vision pre-analysis — one round-trip saved per image if the agent chose not to call the tool. It was rejected on the review pass because *every other adapter* inline-pre-analyzes images via `_enrich_message_with_vision`, and the consistency of the user-message shape matters more than the per-request latency.

The lesson compounds §17.4: if there's a canonical template for a side effect that every platform shares, adopting it verbatim is almost always the right call, even when a locally-cheaper alternative looks viable. The cost of "this adapter behaves subtly differently from the others" accumulates in every downstream feature.

### 17.8 Per-session `asyncio.Lock` for a "request/reply" transport that supports concurrent sessions

NATS's protocol is "request/reply" at a message level, but two callers can target the same `envelope.session` string simultaneously. Our MVP's single-registration-per-Hermes-instance choice (§13(9)) compresses that into one adapter handling both concurrent prompts. Per-session serialization (§17.2) turned out to be the minimum-viable correctness story here — the same approach would apply to any transport where "session" is a caller-supplied field rather than a transport-enforced identifier.

Design choices that ended up load-bearing:

- Keep-alive starts *before* the lock: a queued handler still emits `status:ack` chunks so the caller doesn't hit §6.6's inactivity timeout.
- `_unpack_envelope` runs *before* the lock: attachment decode failures fail fast with an SDK 500, not blocked behind a busy-session queue.
- `_session_locks` is cleared in `_teardown_handles()`: reconnects don't inherit locks held by cancelled tasks.
- Distinct `chat_id`s still run in parallel: the lock is per-session, not global.

### 17.9 Private-attribute peek (stream._request.data) was an acceptable MVP crutch — now retired

**Status: resolved post-Phase 9.** The SDK now exposes `session` as a first-class `Envelope` field, and the adapter reads `envelope.session` directly. The paragraphs below are preserved as the original rationale for the workaround that shipped in Phases 4–8.

Design doc §3 flagged option (b) — peeking the SDK's `stream._request.data` to extract the session value before the SDK's `Envelope` decoder drops the field per `extra="ignore"`. It was shipped and stayed private-attribute-dependent through Phase 8. The failure mode is loud (AttributeError at handler entry) and confined (falls back to session default), which makes it a defensible MVP crutch rather than a landmine.

The cleaner long-term fix was an upstream change to the `natsagent` SDK exposing the field on `Envelope`. That shipped; the crutch (`_extract_session` + `_extract_x_session`) was removed in the same pass that updated this doc. Filing as follow-up rather than a blocker turned out to be the right call.

### 17.10 `entry_id` threading for parallel subagents fits under the "structural" umbrella

Same shape as §17.2 but scoped to within a single adapter handler. Parallel subagents (`delegate_tool`) can each produce a dangerous-command approval that fires the shared `notify_cb`. FIFO-pop semantics in `resolve_gateway_approval` made replies racy — reply-for-B could land on entry-A if A was still pending.

Per-session serialization can't eliminate this race because the subagents are inside one handler (same `chat_id`, same lock already held). Instead, a uuid `id` on each `_ApprovalEntry` + a `_current_approval_entry_id` contextvar lets the adapter capture the id synchronously in the notify callback and pass it through `resolve_gateway_approval(entry_id=…)` to resolve the *correct* entry by match rather than order. Fully backwards-compatible — other adapters resolve without an id and fall through to FIFO.

**Generalizable rule.** When "structural elimination" isn't available because the concurrency is a feature rather than a leak, the next best thing is a precise identity on the thing being resolved and a way to capture it at the correct moment in the flow. Both are cheap; the tricky part is noticing you need them.

### 17.11 "Verify against the full suite" is a concrete phase-close gate, not a formality

Restating §17.5 as policy. Phases 1–7's phase-close ritual used `scripts/run_tests.sh tests/gateway/test_nats_*.py` and declared phase done on green. Phase 4's `hermes-nats` → `hermes-gateway` aggregator miss survived three phases that way. Phase 8's T8.8 caught it because T8.8 explicitly requires the full suite.

The fix isn't "run the full suite every phase" — it's "when a phase touches a cross-module registration point, run the consistency test for that surface." Specific to hermes-agent: `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` for any toolset / `Platform` enum / `PLATFORMS` map change. Cheap insurance, <1 s.

### 17.12 Surprise that did not surface: cache-friendly streaming over a chunked transport

The design doc's §12 testing strategy and the prompt-caching concerns in `CLAUDE.md` warn against mutating past context mid-conversation. The adapter-owned `AIAgent` path was built assuming it would interact with prompt caching the same way CLI and api_server do — and it did. No surprises here despite the concern at Phase 0. Worth mentioning explicitly: the adapter's streaming loop doesn't touch `AIAgent` message history, so cache invariants hold transitively.
