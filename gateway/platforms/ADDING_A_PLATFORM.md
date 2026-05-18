# Adding a New Messaging Platform

There are two ways to add a platform to the Hermes gateway:

## Plugin Path (Recommended for Community/Third-Party)

Create a plugin directory in `~/.hermes/plugins/` (or under `plugins/platforms/`
for bundled plugins) with a `plugin.yaml` and `adapter.py`.  The adapter
inherits from `BasePlatformAdapter` and registers via
`ctx.register_platform()` in the `register(ctx)` entry point.  This requires
**zero changes to core Hermes code**.

The plugin system automatically handles: adapter creation, config parsing,
user authorization, cron delivery, send_message routing, system prompt hints,
status display, gateway setup, and more.

**Optional hooks cover the edges most adapters need:**

- `env_enablement_fn: () -> Optional[dict]` — seeds `PlatformConfig.extra`
  (and an optional `home_channel` dict) from env vars BEFORE the adapter is
  constructed.  Without this, env-only setups don't surface in
  `hermes gateway status` or `get_connected_platforms()` until the SDK
  instantiates.
- `apply_yaml_config_fn: (yaml_cfg, platform_cfg) -> Optional[dict]` —
  translate this platform's `config.yaml` keys into env vars and/or seed
  `PlatformConfig.extra` directly.  Lets a plugin own its YAML schema
  instead of growing core `gateway/config.py` boilerplate per platform.
  Mutating `os.environ` is allowed (use `not os.getenv(...)` guards to
  preserve env > YAML precedence); the returned dict is merged into
  `PlatformConfig.extra`.  Called during `load_gateway_config()` after
  the generic shared-key loop and before `_apply_env_overrides()`.
- `cron_deliver_env_var: str` — name of the `*_HOME_CHANNEL` env var.  When
  set, `deliver=<name>` cron jobs route to this var without editing
  `cron/scheduler.py`'s hardcoded sets.
- `standalone_sender_fn: async (...) -> dict`: out-of-process delivery
  for cron jobs that run separately from the gateway.  Without this, a
  `deliver=<name>` job fires correctly but the actual send returns
  `No live adapter for platform '<name>'`.  Pair with `cron_deliver_env_var`
  for end-to-end cron support.  See the docsite for the signature.
- `plugin.yaml` `requires_env` / `optional_env` rich-dict entries —
  auto-populate `OPTIONAL_ENV_VARS` in `hermes_cli/config.py` so the setup
  wizard surfaces proper descriptions, prompts, password flags, and URLs.

**Subclassing for platform-specific UX.** When a platform has a hard
time-window constraint that the base adapter can't anticipate (LINE's
60s single-use reply token, WhatsApp's 24h session window, etc.), an
adapter can override `_keep_typing` to layer a mid-flight bubble at a
threshold without expanding the kwarg surface. Always
`await super()._keep_typing(...)` so the typing heartbeat keeps running,
and tear down your side task in `finally`. See `plugins/platforms/line/`
for the full pattern (Template Buttons postback at 45s, `RequestCache`
state machine, `interrupt_session_activity` override for `/stop`
orphans) and the developer-guide page for the prose walkthrough.

See `plugins/platforms/irc/`, `plugins/platforms/teams/`, and
`plugins/platforms/google_chat/` for complete working examples, and
`website/docs/developer-guide/adding-platform-adapters.md` for the full
plugin guide with code examples and hook documentation.

---

## Built-in Path (Core Contributors Only)

Checklist for integrating a platform directly into the Hermes core.
Use this as a reference when building a built-in adapter — every item here
is a real integration point. Missing any of them will cause broken
functionality, missing features, or inconsistent behavior.

---

## 1. Core Adapter (`gateway/platforms/<platform>.py`)

The adapter is a subclass of `BasePlatformAdapter` from `gateway/platforms/base.py`.

### Required methods

| Method | Purpose |
|--------|---------|
| `__init__(self, config)` | Parse config, init state. Call `super().__init__(config, Platform.YOUR_PLATFORM)` |
| `connect() -> bool` | Connect to the platform, start listeners. Return True on success |
| `disconnect()` | Stop listeners, close connections, cancel tasks |
| `send(chat_id, text, ...) -> SendResult` | Send a text message |
| `send_typing(chat_id)` | Send typing indicator |
| `send_image(chat_id, image_url, caption) -> SendResult` | Send an image |
| `get_chat_info(chat_id) -> dict` | Return `{name, type, chat_id}` for a chat |

### Optional methods (have default stubs in base)

| Method | Purpose |
|--------|---------|
| `send_document(chat_id, path, caption)` | Send a file attachment |
| `send_voice(chat_id, path)` | Send a voice message |
| `send_video(chat_id, path, caption)` | Send a video |
| `send_animation(chat_id, path, caption)` | Send a GIF/animation |
| `send_image_file(chat_id, path, caption)` | Send image from local file |
| `request_interaction(chat_id, prompt, *, kind, timeout)` | Ask the caller mid-stream and return their reply. Used for dangerous-command approval over transports that have a native request/reply query channel (e.g. NATS `stream.ask`). Default raises `NotImplementedError`; the gateway's `_approval_notify_sync` detects capability via `type(adapter).request_interaction is not BasePlatformAdapter.request_interaction` and falls back to the legacy text-reply flow for adapters that don't override. Canonical implementation: `gateway/platforms/nats.py::NatsAdapter.request_interaction`. |

### Required function

```python
def check_<platform>_requirements() -> bool:
    """Check if this platform's dependencies are available."""
```

### Key patterns to follow

- Use `self.build_source(...)` to construct `SessionSource` objects
- Call `self.handle_message(event)` to dispatch inbound messages to the gateway
- Use `MessageEvent`, `MessageType`, `SendResult` from base
- Use `cache_image_from_bytes`, `cache_audio_from_bytes`, `cache_document_from_bytes` for attachments
- Filter self-messages (prevent reply loops)
- Filter sync/echo messages if the platform has them
- Redact sensitive identifiers (phone numbers, tokens) in all log output
- Implement reconnection with exponential backoff + jitter for streaming connections
- Set `MAX_MESSAGE_LENGTH` if the platform has message size limits

### Streaming model: edit-based vs. adapter-owned `AIAgent`

The default path runs text prompts through `GatewayStreamConsumer`, which buffers deltas and calls `adapter.edit_message()` to progressively replace one chat message. That fits Telegram / Slack / Discord / Matrix. Transports where every delta is a separate publish (NATS, SSE in `api_server`) don't want edits — they want **each delta emitted as its own wire frame**.

If your transport is in the second category:

- Set `SUPPORTS_MESSAGE_EDITING = False` as a class attribute — `gateway/run.py` short-circuits `GatewayStreamConsumer` construction when this is False.
- Own `AIAgent` construction inside the adapter. `gateway/platforms/api_server.py` is the reference; `gateway/platforms/nats.py::_run_text_prompt` + `_run_agent_sync` are the most recent example.
- Register a per-handler `stream_delta_callback` that feeds an `asyncio.Queue`, and drain the queue with a pump task that publishes each delta.
- Attachment enrichment runs in `_handle_message` on the default path. If you're bypassing `handle_message()` for text prompts, replicate the enrichment inline on the adapter hot path — see `NatsAdapter._enrich_event_with_media` (inline `vision_analyze` + bracketed document/audio notes; matches the `GatewayRunner._enrich_message_with_vision` template byte-for-byte). If you skip this, the agent will never see uploaded media — images/docs attached to the envelope silently vanish after caching.

### Approval / mid-stream interaction wiring (adapter-owned agent path)

`_approval_notify_sync` registers its hook via `register_gateway_notify(session_key, cb)` inside the **default** `handle_message()` path (`gateway/run.py:9993`). Adapters that own their own `AIAgent` and bypass `handle_message` for text prompts **also bypass that registration** — approval callbacks from the agent thread will find no registered notify and hang on `entry.event.wait()` until the framework timeout (default 300 s).

Fix if you're using the adapter-owned pattern:

1. Register your own `register_gateway_notify(session_key, cb)` at `_run_agent_sync` entry, and unregister it in the `finally` block.
2. Call `set_current_session_key(session_key)` **on the executor thread** at entry, and `reset_current_session_key(token)` in `finally`. `asyncio.loop.run_in_executor()` does NOT propagate contextvars by default, so the session-key contextvar that tools see (e.g. `tools/approval.py::get_current_session_key()`) must be set explicitly inside the worker thread.
3. If you implement `request_interaction`, use `gateway/platforms/base.py::dispatch_approval_via_request_interaction` to route the callback — it handles the prompt formatting, scheduled-coroutine lifecycle, reply parsing, and `resolve_gateway_approval` call in lockstep with the default path, including the entry-id path for parallel subagents.
4. Synchronously capture `get_current_approval_entry_id()` in the notify callback **before** scheduling async work. The contextvar is set on the agent's worker thread; the coroutine scheduled via `asyncio.run_coroutine_threadsafe` starts with a fresh context and can't read it. Pass the captured id through to `resolve_gateway_approval(entry_id=…)` so parallel subagents route their replies to the right entry rather than FIFO-popping.

### Contextvar propagation across threads

`asyncio.run_coroutine_threadsafe(coro, loop)` creates a fresh context on the target loop — **no contextvars set on the calling thread are visible inside the scheduled coroutine**. This is the default behavior and won't change upstream. Three places this surfaces in the gateway:

- Session-key contextvar (`tools/approval.py`) → required explicitly on both sides when crossing threads.
- Approval entry-id contextvar (`tools/approval.py`) → capture sync **before** scheduling, close over the captured value.
- Any adapter-local contextvar you add (e.g. `NatsAdapter._current_stream`) → same rule.

Conversely, `gateway/run.py::_run_in_executor_with_context` uses `copy_context()` to propagate context **into** executor threads. That direction works; the `run_coroutine_threadsafe` direction does not. Mixing sync worker threads with async loops without respecting this will surface as "works in unit tests, hangs in integration."

### Session serialization (structural race elimination)

If your transport supports multiple concurrent prompts on the same prompt subject (NATS, gRPC streaming, anything programmatic where multiple callers can race the same endpoint), a session-scoped `asyncio.Lock` inside the adapter eliminates entire classes of races structurally — concurrent-handler stream-registration overwrites, notify-callback overwrites, ambiguous contextvar fallbacks — by making the concurrent state impossible rather than reconciling it correctly.

Canonical pattern (`NatsAdapter`, post-v0.3):

- `_session_lock: asyncio.Lock` — single Lock per service, since v0.3 pins one `session_name` per `AgentService` (multi-session = multi-profile). For a transport that genuinely hosts multiple sessions in one process, scale this to `dict[str, asyncio.Lock]` keyed on the session id.
- Acquire the lock **after** keep-alive emission starts (a queued handler should still signal liveness to the caller) and **after** envelope/attachment decoding (malformed inputs should fail fast, not queue).
- Rebuild the Lock in `connect()` so reconnects don't inherit a Lock held by a cancelled task.

Prefer this over per-send disambiguation (contextvar priority, compound-key registries, closure-captured streams) when the concurrency itself isn't load-bearing. A test that asserts "two overlapping handlers each route their send to the right stream" becomes **structurally impossible under serialization** — the interleaved sequence deadlocks — and the replacement test asserts the stronger timeline invariant (A enter → A leave → B enter → B leave).

---

## 2. Platform Enum (`gateway/config.py`)

Add the platform to the `Platform` enum:

```python
class Platform(Enum):
    ...
    YOUR_PLATFORM = "your_platform"
```

Add env var loading in `_apply_env_overrides()`:

```python
# Your Platform
your_token = os.getenv("YOUR_PLATFORM_TOKEN")
if your_token:
    if Platform.YOUR_PLATFORM not in config.platforms:
        config.platforms[Platform.YOUR_PLATFORM] = PlatformConfig()
    config.platforms[Platform.YOUR_PLATFORM].enabled = True
    config.platforms[Platform.YOUR_PLATFORM].token = your_token
```

Update `get_connected_platforms()` if your platform doesn't use token/api_key
(e.g., WhatsApp uses `enabled` flag, Signal uses `extra` dict, NATS requires
`enabled AND (servers OR context)`).

---

## 3. Adapter Factory (`gateway/run.py`)

Add to `_create_adapter()`:

```python
elif platform == Platform.YOUR_PLATFORM:
    from gateway.platforms.your_platform import YourAdapter, check_your_requirements
    if not check_your_requirements():
        logger.warning("Your Platform: dependencies not met")
        return None
    return YourAdapter(config)
```

---

## 4. Authorization Maps (`gateway/run.py`)

Add to BOTH dicts in `_is_user_authorized()`:

```python
platform_env_map = {
    ...
    Platform.YOUR_PLATFORM: "YOUR_PLATFORM_ALLOWED_USERS",
}
platform_allow_all_map = {
    ...
    Platform.YOUR_PLATFORM: "YOUR_PLATFORM_ALLOW_ALL_USERS",
}
```

### Transport-layer auth: early-return instead of per-user allowlist

Some transports delegate authorization to the transport layer itself (HMAC
signatures for webhooks, HASS_TOKEN for Home Assistant, NATS accounts / NKey /
JWT / TLS for NATS). These platforms **skip the allowlist check** entirely —
add them to the `(Platform.HOMEASSISTANT, Platform.WEBHOOK, Platform.NATS)`
tuple in the early-return branch of `_is_user_authorized` instead of wiring
them into `platform_env_map` / `platform_allow_all_map`.

Failing to add a transport-authed platform to the early-return tuple causes a
subtle bug: inbound messages are treated as anonymous / un-paired users and
the bot replies with a pairing code instead of executing the command.
Regression test pattern: `tests/gateway/test_unauthorized_dm_behavior.py::test_nats_is_authorized_without_user_allowlist`.

---

## 5. Session Source (`gateway/session.py`)

If your platform needs extra identity fields (e.g., Signal's UUID alongside
phone number), add them to the `SessionSource` dataclass with `Optional` defaults,
and update `to_dict()`, `from_dict()`, and `build_source()` in base.py.

---

## 6. System Prompt Hints (`agent/prompt_builder.py`)

Add a `PLATFORM_HINTS` entry so the agent knows what platform it's on:

```python
PLATFORM_HINTS = {
    ...
    "your_platform": (
        "You are on Your Platform. "
        "Describe formatting capabilities, media support, etc."
    ),
}
```

Without this, the agent won't know it's on your platform and may use
inappropriate formatting (e.g., markdown on platforms that don't render it).
The `PLATFORM_HINTS` entry is only reached when the adapter routes text
prompts through `handle_message()` → `_build_system_prompt()`. Adapters that
own their own `AIAgent` construction (§1 "Streaming model: adapter-owned
AIAgent") also bypass the hint injection unless they explicitly thread
`platform` into their `AIAgent()` call — add the hint anyway for the default
path, and wire `platform="your_platform"` into the adapter's agent
construction so the hint is applied regardless of path.

---

## 7. Toolset (`toolsets.py`)

Add a named toolset for your platform:

```python
"hermes-your-platform": {
    "description": "Your Platform bot toolset",
    "tools": _HERMES_CORE_TOOLS,
    "includes": []
},
```

And add it to the `hermes-gateway` composite:

```python
"hermes-gateway": {
    "includes": [..., "hermes-your-platform"]
}
```

**Both steps are required.** Forgetting the `hermes-gateway` includes line is
easy to miss because the platform-specific tests still pass, but
`tests/hermes_cli/test_tools_config.py::TestPlatformToolsetConsistency::test_gateway_toolset_includes_all_messaging_platforms`
will fail. This test is cheap (<1 s) — run it as part of your phase close,
not just your adapter-file subset:

```bash
scripts/run_tests.sh tests/hermes_cli/test_tools_config.py
```

The NATS adapter shipped three phases with this line missing because the
NATS-only subset never exercised it; Phase 8's first full-suite run caught it.
Registering any toolset name (platform or otherwise) is a cross-module surface
— run the consistency test whenever you touch `toolsets.py`, `PLATFORMS` in
`hermes_cli/platforms.py`, or the `Platform` enum.

---

## 8. Cron Delivery (`cron/scheduler.py`)

Add to `platform_map` in `_deliver_result()`:

```python
platform_map = {
    ...
    "your_platform": Platform.YOUR_PLATFORM,
}
```

Without this, `cronjob(action="create", deliver="your_platform", ...)` silently fails.

**Skip this step** if your transport is request/reply only (no server-initiated
delivery). Cron delivery requires the adapter to publish to a persistent
address that exists between prompts — a chat ID, room, email address, etc. For
pure request/reply transports like NATS (where the "reply subject" exists only
for the duration of one prompt and vanishes when the caller's iterator exits),
cron delivery has no meaningful destination and is correctly omitted. The
NATS adapter is absent from `cron/scheduler.py::_deliver_result` by design.

---

## 9. Send Message Tool (`tools/send_message_tool.py`)

Add to `platform_map` in `send_message_tool()`:

```python
platform_map = {
    ...
    "your_platform": Platform.YOUR_PLATFORM,
}
```

Add routing in `_send_to_platform()`:

```python
elif platform == Platform.YOUR_PLATFORM:
    return await _send_your_platform(pconfig, chat_id, message)
```

Implement `_send_your_platform()` — a standalone async function that sends
a single message without requiring the full adapter (for use by cron jobs
and the send_message tool outside the gateway process).

Update the tool schema `target` description to include your platform example.

**Skip this step** if your transport is request/reply only (same reasoning as
§8). The `send_message` tool ships a fresh message to a persistent address;
request/reply transports have no such address. The NATS adapter omits
`send_message_tool` routing by design.

---

## 10. Cronjob Tool Schema (`tools/cronjob_tools.py`)

Update the `deliver` parameter description and docstring to mention your
platform as a delivery option.

---

## 11. Channel Directory (`gateway/channel_directory.py`)

If your platform can't enumerate chats (most can't), add it to the
session-based discovery list:

```python
for plat_name in ("telegram", "whatsapp", "signal", "your_platform"):
```

**Skip this step** for pure request/reply transports where "channels" are
transient (one reply subject per prompt, no persistence). NATS is absent from
the channel directory by design.

---

## 12. Status Display (`hermes_cli/status.py`)

Add to the `platforms` dict in the Messaging Platforms section:

```python
platforms = {
    ...
    "Your Platform": ("YOUR_PLATFORM_TOKEN", "YOUR_PLATFORM_HOME_CHANNEL"),
}
```

---

## 13. Gateway Setup Wizard (`hermes_cli/gateway.py`)

Add to the `_PLATFORMS` list:

```python
{
    "key": "your_platform",
    "label": "Your Platform",
    "emoji": "📱",
    "token_var": "YOUR_PLATFORM_TOKEN",
    "setup_instructions": [...],
    "vars": [...],
}
```

If your platform needs custom setup logic (connectivity testing, QR codes,
policy choices), add a `_setup_your_platform()` function and route to it
in the platform selection switch.

Update `_platform_status()` if your platform's "configured" check differs
from the standard `bool(get_env_value(token_var))`.

---

## 14. Phone/ID Redaction (`agent/redact.py`)

If your platform uses sensitive identifiers (phone numbers, etc.), add a
regex pattern and redaction function to `agent/redact.py`. This ensures
identifiers are masked in ALL log output, not just your adapter's logs.

---

## 15. Documentation

| File | What to update |
|------|---------------|
| `README.md` | Platform list in feature table + documentation table |
| `AGENTS.md` | Gateway description + env var config section |
| `website/docs/user-guide/messaging/<platform>.md` | **NEW** — Full setup guide (see existing platform docs for template) |
| `website/docs/user-guide/messaging/index.md` | Architecture diagram, toolset table, security examples, Next Steps links |
| `website/docs/reference/environment-variables.md` | All env vars for the platform |

---

## 16. Tests (`tests/gateway/test_<platform>.py`)

Recommended test coverage:

- Platform enum exists with correct value
- Config loading from env vars via `_apply_env_overrides`
- Adapter init (config parsing, allowlist handling, default values)
- Helper functions (redaction, parsing, file type detection)
- Session source round-trip (to_dict → from_dict)
- Authorization integration (platform in allowlist maps)
- Send message tool routing (platform in platform_map)

Optional but valuable:
- Async tests for message handling flow (mock the platform API)
- SSE/WebSocket reconnection logic
- Attachment processing
- Group message filtering

---

## Quick Verification

After implementing everything, verify with:

```bash
# All tests pass
python -m pytest tests/ -q

# Grep for your platform name to find any missed integration points
grep -r "telegram\|discord\|whatsapp\|slack" gateway/ tools/ agent/ cron/ hermes_cli/ toolsets.py \
  --include="*.py" -l | sort -u
# Check each file in the output — if it mentions other platforms but not yours, you missed it
```
