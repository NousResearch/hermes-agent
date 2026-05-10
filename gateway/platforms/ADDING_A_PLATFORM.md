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
(e.g., WhatsApp uses `enabled` flag, Signal uses `extra` dict).

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
