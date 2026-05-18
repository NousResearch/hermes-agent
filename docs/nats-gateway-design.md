# NATS Gateway Channel — Design Doc

**Status:** Living doc — Phase 10 migration to protocol v0.3 is the current shipped state.
**Scope:** NATS as a gateway channel in Hermes Agent so callers can prompt the agent over NATS — send text, send attachments, and receive token-streamed responses — using the **NATS Agent Protocol v0.3**.
**Wire spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.3).
**SDKs (split as of client v0.5 / agent v0.1, 2026-04-30; published to PyPI 2026-05-01; current floors `synadia-ai-agents>=0.7` / `synadia-ai-agent-service>=0.3` since 2026-05-04):** Two distributions sourced from the `synadia-ai/synadia-agents` monorepo, both pulled in by the `hermes-agent[nats]` extra.
- **Client SDK** — [`synadia-ai-agents`](https://pypi.org/project/synadia-ai-agents/) (source: `../synadia-agents/client-sdk/python`, import root `synadia_ai.agents`). Wire types only: `Envelope`, `Attachment`, `ResponseChunk`, `StatusChunk`, `QueryChunk`, `HeartbeatPayload`, `AgentSubject`, errors (`QueryTimeout`, `ProtocolError`, `StreamStalledError`, `StreamMaxWaitExceededError`, `AgentsClosedError`, …), discovery (`Agents`, `DiscoverFilter`), helpers (`load_context_options`, `parse_nats_url`). v0.6 added the per-prompt `max_wait_s` ceiling (default 600 s) and per-NC mux reply-inbox; v0.7 restored optional `HeartbeatPayload.session` (§8.3) — present iff `metadata.session` is set, omitted on re-encode when absent. Wire-shape stays at protocol `"0.3"`.
- **Agent SDK** — [`synadia-ai-agent-service`](https://pypi.org/project/synadia-ai-agent-service/) (source: `../synadia-agents/agent-sdk/python`, import root `synadia_ai.agent_service`). Host-side only: `AgentService`, `PromptStream`, `PromptHandler`, host defaults, heartbeat publisher loop. v0.2 was a dependency-floor bump (no code change); v0.3 restored §3.2 `metadata.session` advertisement + §8.3 / §8.7 `session` payload field — the SDK populates them from the constructor `session_name=`, so no adapter-side change. Depends on `synadia-ai-agents>=0.7`.

Cross-references to the protocol spec are by section number (e.g. §5.6). Cross-references to the Hermes codebase use `file:line`.

---

## 1. Summary

`gateway/platforms/nats.py` is a `BasePlatformAdapter` subclass. It registers one `synadia_ai.agent_service.AgentService` with the identity `agents.prompt.hermes.<owner>.<session_name>` at gateway startup; each inbound `prompt` is translated into a Hermes `MessageEvent`, routed through the normal gateway handler, streamed back chunk-by-chunk over NATS, and terminated by the SDK's empty-body terminator.

Session routing uses the **5th subject token** (`session_name`) — v0.3 collapsed `Envelope.session` into the subject itself. Each `AgentService` serves exactly one session; multi-session deployments use Hermes profile isolation (one profile = one service). Mid-stream approvals round-trip via `PromptStream.ask()`. Attachments round-trip base64 ↔ Hermes media cache. The adapter owns its own `AIAgent` construction and streaming pipeline (api_server-style), bypassing the gateway's `GatewayStreamConsumer` — see §6 for why.

Explicit non-goals: the future `attachments` endpoint (§5.5), JetStream at-least-once, E2E encryption, cross-platform adapter-level approval refactor, multi-session multiplexing within one process. All are carried forward in §13.

---

## 2. Protocol ↔ Adapter mapping

| Direction | Protocol (v0.3)                                                            | Adapter surface                                                                                                                                    |
|-----------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Inbound   | `Envelope.prompt`                                                          | `MessageEvent.text`                                                                                                                                 |
| Inbound   | `Envelope.attachments[i]` (base64)                                         | Decoded via `att.to_bytes()`, routed through `cache_{image,audio,document,video}_from_bytes` → `MessageEvent.media_urls` / `media_types`            |
| Inbound   | 5th subject token (`session_name`)                                         | `SessionSource.chat_id` — fixed per service, sourced from `settings.session_name`                                                                   |
| Inbound   | `agents.prompt.hermes.<owner>.<session_name>` (prompt endpoint, v0.3)      | Implicit — SDK dispatches to `AgentService.on_prompt()`                                                                                              |
| Inbound   | `agents.status.hermes.<owner>.<session_name>` (status endpoint, v0.3 PR#24)| SDK-owned — replies with the current heartbeat-shaped JSON on demand                                                                                |
| Outbound  | `{type: response, data: "<text>"}` (§6.3 bare-string form)                 | `stream.send(ResponseChunk(text=delta))` via adapter-local `stream_delta_callback`                                                                   |
| Outbound  | `{type: response, data: {text, attachments: [...]}}`                       | `send_image_file()` / `send_document()` / `send_voice()` / `send_video()` → `ResponseChunk(text=caption, attachments=[Attachment.from_path(...)])`   |
| Outbound  | `{type: status, data: "ack"}` (§6.4)                                       | Keep-alive emitted every ~20 s while the handler is silent (see §6.2)                                                                                |
| Outbound  | `{type: query, data: {...}}` (§7.1)                                        | `adapter.request_interaction()` → `stream.ask(prompt, timeout=…)`                                                                                    |
| Outbound  | Empty-body terminator (§6.5)                                               | Emitted automatically by the SDK when `_on_prompt()` returns                                                                                         |
| Outbound  | Error-headered frame + terminator (§9.3)                                   | Raise from `_on_prompt()`; SDK calls `respond_error(...)` + terminator                                                                               |
| Liveness  | `agents.hb.hermes.<owner>.<session_name>` (v0.3 verb-first heartbeat)      | Automatic, SDK-owned — we pass `heartbeat_interval_s` only                                                                                            |
| Liveness  | Reply inbox prefix `_INBOX.agents` (v0.3 PR#25, caller-side)               | Informational — service-side. Document in NATS account permissions (`_INBOX.agents.>` for caller principals)                                          |
| Discovery | `$SRV.PING.agents`, `$SRV.INFO.agents[.{id}]`                              | Automatic — SDK registers as a NATS micro service                                                                                                    |
| Cancel    | None (§6.7 — interest-based, no wire signal)                               | MVP: no detection. Agent runs to completion. Revisit post-MVP with a periodic `stream.ask("alive?")`                                                 |

### Key points

- **v0.3 verb-first subjects:** every endpoint gets a verb token before the identity tuple — `agents.prompt.{a}.{o}.{s}`, `agents.hb.{a}.{o}.{s}`, `agents.status.{a}.{o}.{s}`. `metadata.protocol_version` is `"0.3"`.
- **SDK owns**: subject construction (§2), service registration (§3), envelope parsing (§5), chunk wrapping (§6.2/6.3), terminator (§6.5), error frames (§9), heartbeat emission (§8), the on-demand `agents.status` request endpoint, the `_INBOX.agents` caller reply prefix.
- **SDK does NOT own**: `status:ack` keep-alive cadence (§6.4) — we must emit. Per the spec's recommended caller inactivity timeout (§6.6: 60 s), we keep silence below that by a comfortable margin.

---

## 3. Session model

**Design decision (v0.3):** session is the **5th subject token** (`session_name`), fixed per `AgentService`. One service = one session_name; multi-session deployments use Hermes profile isolation.

### Why a fixed-per-service session

- Protocol v0.3 (PR #26) collapsed `name` and `session` into a single `session_name`. `metadata.session`, `Envelope.session`, `HeartbeatPayload.session`, `AgentService(session=...)` and `Agent.prompt(session=...)` were all removed. "A worker that wants N sessions registers N services."
- Hermes profile isolation already separates `HERMES_HOME`, config, sessions, memory, and per-platform locks per profile. One profile = one `AgentService` = one `session_name` is the natural fit and requires no additional plumbing.
- Removes the v0.2 envelope.session demux + per-chat_id Lock pool entirely — the lock collapses to a single `_session_lock` and stream resolution becomes unambiguous.

### Inbound translation

```text
5th subject token (session_name, fixed at service start)
     │
     ▼
chat_id = self._settings.session_name
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

### Multi-session deployments

To run N sessions on one host, run N profiles:

```bash
hermes -p alice profile create
hermes -p alice setup            # HERMES_NATS_SESSION_NAME=alice
hermes -p bob profile create
hermes -p bob setup              # HERMES_NATS_SESSION_NAME=bob
```

Each profile registers its own `AgentService` at a distinct subject (`agents.prompt.hermes.<owner>.alice` vs `agents.prompt.hermes.<owner>.bob`) and acquires its own scoped lock. The platform-lock contract (§5) is unchanged — its identity simply uses `session_name` in place of v0.2's `name` token.

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

      # Identity on the wire — produces subject
      # agents.prompt.hermes.<owner>.<session_name>
      agent: hermes                       # §2 token; default "hermes"
      owner: rene                         # §2 token; required
      session_name: default               # 5th subject token; required

      # Behavior tuning (all optional)
      heartbeat_interval_s: 30            # §8.2 default
      max_payload: "1MB"                  # §2.1 endpoint metadata
      attachments_ok: true                # §2.1 endpoint metadata
      ack_keepalive_interval_s: 20        # adapter-local, not on the wire
```

### Validation (fails `_set_fatal_error(..., retryable=False)` in `__init__`)

- Exactly one of `servers` (non-empty list of strings) or `context` (non-empty string) is set.
- `owner` and `session_name` are non-empty strings conforming to §2.2 naming rules (the SDK's `AgentSubject.new()` enforces and sanitizes — but we fail fast with a readable error before construction).
- `max_payload` parses against the SDK's size grammar (the SDK will crash on construction otherwise).
- `ack_keepalive_interval_s < 60` (leave headroom under §6.6's recommended 60 s caller inactivity timeout).

### Env var overrides (`_apply_env_overrides()` in `gateway/config.py`)

| Env var                    | Overrides                        | Notes                                                      |
|----------------------------|----------------------------------|------------------------------------------------------------|
| `NATS_URL`                 | `extra.servers` (single-URL list) | Canonical env name in the NATS ecosystem                   |
| `NATS_CONTEXT`             | `extra.context`                   | Splatted via `sdk.load_context_options(name)` into `nats.connect` |
| `HERMES_NATS_AGENT`        | `extra.agent`                     | Optional; rarely overridden                                 |
| `HERMES_NATS_OWNER`        | `extra.owner`                     | Common in multi-tenant deployments                          |
| `HERMES_NATS_SESSION_NAME` | `extra.session_name`              | The 5th subject token; required                             |

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

Two Hermes profiles trying to register the **same** `(agent, owner, session_name)` would land on the same NATS prompt subject and both receive load-balanced prompts — silently wrong.

**Lock scope:** `"nats"`
**Lock identity:** `f"{agent}:{owner}:{session_name}"`

```python
# in connect(), before nats.connect(...)
if not self._acquire_platform_lock("nats", f"{agent}:{owner}:{session_name}", "NATS agent identity"):
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
NATS msg on agents.prompt.hermes.<owner>.<session_name>
         │
         ▼
 SDK decodes envelope → calls adapter._on_prompt(envelope, stream)
         │
         ├─── chat_id = settings.session_name (5th subject token)
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
    except sdk.QueryTimeout:
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
        ├── _acquire_platform_lock("nats", f"{agent}:{owner}:{session_name}", ...)
        ├── self._nc = await nats.connect(servers=...)              # or **sdk.load_context_options(context)
        ├── self._service = sdk.AgentService(agent=..., owner=..., session_name=..., nc=self._nc, ...)
        ├── self._service.on_prompt(self._on_prompt)
        ├── await self._service.start()  # registers micro service (prompt + status) + heartbeat
        └── self._mark_connected()
```

### Per-prompt

```
synadia_ai.agents SDK receives NATS msg
        │
        ▼
AgentService prompt endpoint → decode envelope → PromptStream → handler
        │
        ▼
NatsAdapter._on_prompt(envelope, stream)
        │
        ├── chat_id = settings.session_name (5th subject token)
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
        ├── await self._service.stop()   # stops heartbeat + deregisters micro service
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
| NATS server unreachable at connect        | `nats.connect` raises                     | `_set_fatal_error("nats_connect_error", ..., retryable=True)`; gateway may retry |
| Identity already locked on this host      | `_acquire_platform_lock` returns False    | `_set_fatal_error("nats_lock", ..., retryable=False)`; do not retry      |
| NATS reconnect mid-stream                 | `stream.send` raises inside pump          | Log; agent continues to completion; SDK emits error frame                |
| Caller drops reply subscription (§6.7)    | No detection in MVP                        | Agent runs to completion; published chunks dropped by NATS server        |
| Oversize inbound envelope                 | SDK's caller-side §5.4 + our check         | Reject before cache_*; raise → SDK `respond_error(400)`                  |
| Handler raises unexpectedly               | SDK wraps exception                        | `respond_error(500, <sanitized desc>)` then terminator                   |
| `max_payload` format bad at init          | SDK size-grammar parser raises             | `_set_fatal_error(...)` during `__init__`                                |
| `owner`/`session_name` violate §2.2       | `AgentSubject.new()` raise                 | `_set_fatal_error(...)` during `connect()`                               |
| Two Hermes profiles, same identity        | Lock collision                             | Second fails fast with actionable message (`telegram.py` precedent)      |

---

## 12. Testing strategy

Tests use `scripts/run_tests.sh` (hermetic wrapper). Mirror the Telegram collection-time mock so the suite runs without `synadia-ai-agents` / `synadia-ai-agent-service` installed. After the v0.5 client / v0.1 agent split, the host-side surface (`AgentService`, `PromptStream`, `PromptHandler`) lives on a separate `synadia_ai.agent_service` module that must be registered alongside the wire-types mock:

```python
# tests/gateway/conftest.py (addition)
def _ensure_synadia_agents_mock() -> None:
    if (
        "synadia_ai.agents" in sys.modules
        and hasattr(sys.modules["synadia_ai.agents"], "__file__")
        and "synadia_ai.agent_service" in sys.modules
        and hasattr(sys.modules["synadia_ai.agent_service"], "__file__")
    ):
        return

    # Wire-types module — synadia_ai.agents
    mod = MagicMock()
    mod.load_context_options = MagicMock(return_value={"servers": ["nats://stub:4222"]})
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

    # Host-side module — synadia_ai.agent_service (split out in v0.5/0.1)
    agent_service_mod = MagicMock()
    agent_service_mod.AgentService = MagicMock()
    agent_service_mod.AgentService.return_value.start = AsyncMock()
    agent_service_mod.AgentService.return_value.stop = AsyncMock()
    agent_service_mod.PromptStream = MagicMock()
    agent_service_mod.PromptHandler = MagicMock

    parent = MagicMock()
    parent.agents = mod
    parent.agent_service = agent_service_mod
    sys.modules["synadia_ai"] = parent
    sys.modules["synadia_ai.agents"] = mod
    sys.modules["synadia_ai.agent_service"] = agent_service_mod

_ensure_synadia_agents_mock()
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
8. **Caller rate-limiting / validation** — trust NATS account-level auth to gate publishers.
9. **Multi-session multiplexing within one process** — v0.3 fixed: one `AgentService` = one `session_name` (PR #26). For N sessions, run N profiles. The v0.2 envelope.session demux is gone.

---

## 14. Local smoke-test recipe (feeds T8.*)

Installation (one-time):

```bash
source venv/bin/activate
# Both SDKs are pulled from PyPI by the [nats] extra:
uv sync --all-extras --locked
# or, without uv:  pip install 'hermes-agent[nats]'
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

Verify (from a checkout of the SDK monorepo — `git clone https://github.com/synadia-ai/synadia-agents && cd synadia-agents/client-sdk/python`; the wheel doesn't include `examples/`):

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
| Test mock pattern              | `tests/gateway/conftest.py::_ensure_telegram_mock`   | Mirror for `_ensure_synadia_agents_mock`             |
| CLI approval callback          | `hermes_cli/callbacks.py` (reference only)           | CLI-only; gateway uses its own notify bridge         |
| Client SDK source (wire types) | `synadia-agents/client-sdk/python/src/synadia_ai/agents/` (monorepo) | `envelope.py`, `messages.py`, `connect.py`, errors, discovery |
| Agent SDK source (host)        | `synadia-agents/agent-sdk/python/src/synadia_ai/agent_service/` (monorepo) | `service.py`, `prompt_stream.py`, heartbeat publisher |
| SDK examples                   | `synadia-agents/client-sdk/python/examples/01..05-*.py` (monorepo only — not in wheel) | Smoke-test inputs (§14)                        |
| Protocol spec                  | `../nats-agent-sdk-docs/core-protocol.md`            | v0.3                                                 |

---

## 16. Decision log (for reviewer)

- **Session identity is the 5th subject token (`session_name`)** — v0.3 PR #26 collapsed `name` and `session` into one. Hermes profile isolation provides multi-session deployments (one profile = one service = one `session_name`).
- **NATS bypasses `GatewayStreamConsumer`** — it's designed for edit-based transports; NATS semantics (each chunk is a separate publish) don't match. Follow `api_server.py` pattern instead.
- **New `request_interaction` hook is capability-gated**, not a cross-platform refactor. Non-NATS adapters keep their current (mostly non-functional) approval flow unchanged.
- **`status:ack` is a fixed 20 s tick** for MVP simplicity; knob exists for later activity-aware tuning.
- **Lock scope is `"nats"` + `"{agent}:{owner}:{session_name}"`** identity — matches the resource being guarded (the wire subject), not the server URL.

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

**Phase 10 addendum (v0.3):** the per-`chat_id` Lock pool collapsed to a single `_session_lock` once the SDK pinned one `session_name` per service. The serialization invariant survived verbatim — what changed is that there's now exactly one Lock to hold, and `chat_id` is a constant of the process rather than a per-prompt input.

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

### 17.8 `asyncio.Lock` for a "request/reply" transport that supports concurrent prompts

NATS's protocol is "request/reply" at a message level, but two callers can hit the same prompt subject simultaneously. v0.2 had this aggravated by envelope.session demux on top of one adapter; v0.3 pinned one `session_name` per service, so the lock collapses from a per-`chat_id` pool to a single `_session_lock`. The serialization invariant remains: at most one handler in flight at a time. Same approach applies to any transport where the service hosts a single conversational session.

Design choices that ended up load-bearing:

- Keep-alive starts *before* the lock: a queued handler still emits `status:ack` chunks so the caller doesn't hit §6.6's inactivity timeout.
- `_unpack_envelope` runs *before* the lock: attachment decode failures fail fast with an SDK 500, not blocked behind a busy queue.
- `_session_lock` is rebuilt in `connect()`: reconnects don't inherit a Lock held by a cancelled task.

### 17.9 Private-attribute peek (stream._request.data) was an acceptable MVP crutch — now retired

**Status: resolved post-Phase 9, then made moot in Phase 10.** Phase 9 saw the SDK expose `session` as a first-class `Envelope` field; Phase 10 removed `Envelope.session` entirely (v0.3 PR #26) and moved the session into the 5th subject token. The adapter now reads `settings.session_name` and never inspects the envelope for session identity.

Design doc §3 flagged option (b) — peeking the SDK's `stream._request.data` to extract the session value before the SDK's `Envelope` decoder drops the field per `extra="ignore"`. It was shipped and stayed private-attribute-dependent through Phase 8. The failure mode was loud (AttributeError at handler entry) and confined (falls back to session default), which made it a defensible MVP crutch rather than a landmine.

The cleaner long-term fix arrived in two stages: first the SDK exposed `session` on `Envelope` (Phase 9), then v0.3 collapsed it into the subject token (Phase 10). Filing as follow-up rather than a blocker turned out to be the right call.

### 17.10 `entry_id` threading for parallel subagents fits under the "structural" umbrella

Same shape as §17.2 but scoped to within a single adapter handler. Parallel subagents (`delegate_tool`) can each produce a dangerous-command approval that fires the shared `notify_cb`. FIFO-pop semantics in `resolve_gateway_approval` made replies racy — reply-for-B could land on entry-A if A was still pending.

Per-session serialization can't eliminate this race because the subagents are inside one handler (same `chat_id`, same lock already held). Instead, a uuid `id` on each `_ApprovalEntry` + a `_current_approval_entry_id` contextvar lets the adapter capture the id synchronously in the notify callback and pass it through `resolve_gateway_approval(entry_id=…)` to resolve the *correct* entry by match rather than order. Fully backwards-compatible — other adapters resolve without an id and fall through to FIFO.

**Generalizable rule.** When "structural elimination" isn't available because the concurrency is a feature rather than a leak, the next best thing is a precise identity on the thing being resolved and a way to capture it at the correct moment in the flow. Both are cheap; the tricky part is noticing you need them.

### 17.11 "Verify against the full suite" is a concrete phase-close gate, not a formality

Restating §17.5 as policy. Phases 1–7's phase-close ritual used `scripts/run_tests.sh tests/gateway/test_nats_*.py` and declared phase done on green. Phase 4's `hermes-nats` → `hermes-gateway` aggregator miss survived three phases that way. Phase 8's T8.8 caught it because T8.8 explicitly requires the full suite.

The fix isn't "run the full suite every phase" — it's "when a phase touches a cross-module registration point, run the consistency test for that surface." Specific to hermes-agent: `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` for any toolset / `Platform` enum / `PLATFORMS` map change. Cheap insurance, <1 s.

### 17.12 Surprise that did not surface: cache-friendly streaming over a chunked transport

The design doc's §12 testing strategy and the prompt-caching concerns in `CLAUDE.md` warn against mutating past context mid-conversation. The adapter-owned `AIAgent` path was built assuming it would interact with prompt caching the same way CLI and api_server do — and it did. No surprises here despite the concern at Phase 0. Worth mentioning explicitly: the adapter's streaming loop doesn't touch `AIAgent` message history, so cache invariants hold transitively.

### 17.13 SDK split (v0.5 client / v0.1 agent, 2026-04-30)

The single `synadia-ai-agents` distribution was split upstream into a wire-only client SDK (`synadia-ai-agents` v0.5, `synadia_ai.agents`) and a separate host-side agent SDK (`synadia-ai-agent-service` v0.1, `synadia_ai.agent_service`). Hermes-agent imports a mix of host-side (`AgentService`, `PromptStream`) and wire-side (`Envelope`, `Attachment`, chunk classes, `load_context_options`) symbols, so the split required retargeting the host-side imports to the new module while leaving wire-type imports on the old root.

Sympathetic change: PR #41 made the SDK clamp a constructor-supplied `max_payload` down to `nc.max_payload` at `start()`. Hermes hardcoded `DEFAULT_MAX_PAYLOAD = "1MB"` and unconditionally passed it to `AgentService`, which capped every host at 1 MB regardless of negotiated capacity. The fix was to make `NatsConfig.max_payload` `Optional[str]` and derive from `nc.max_payload` in `_on_connect` when unset — the SDK's clamp-down logic still runs unchanged. The connected log line now shows `(server-negotiated)` vs `(configured)` so operators can tell at a glance which path resolved.

**Generalizable rule.** When an upstream package splits along a layering boundary, treat host imports and wire imports as two move sets — even when a function happens to keep the same name. Static `from synadia_ai.agents import AgentService` would have silently kept resolving from the old root, masking the split until the wheel was rebuilt; an `import synadia_ai.agents as sdk` aliasing pattern made the renamed call site (`sdk_svc.AgentService`) impossible to miss in code review.
