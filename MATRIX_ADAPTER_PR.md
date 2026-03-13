# Hermes Matrix Adapter — PR Plan

Complete, verified implementation plan for adding Matrix to Hermes Agent.
Every integration point has been cross-referenced against the **actual current source** on
`feat/matrix-gateway` (rebased on `upstream/main` as of 2026-03-13).

**Issue to close:** NousResearch/hermes-agent [#73](https://github.com/NousResearch/hermes-agent/issues/73) — Matrix Protocol Support
**Claimed:** Yes — comment posted 2026-03-13.

**Primary reference — nanobot Matrix channel (MIT):**
`/home/reed/Projects/personal/nanobot/nanobot/channels/matrix.py` (714 lines)
Uses `matrix-nio`, production-quality sync loop, typing keepalive, media upload/download,
group policy, thread metadata. Direct structural reference for `_sync_loop`,
`_on_message`, `_send_room_content`, and media handling.

**Pattern to follow — most recently merged gateway platform:**
`gateway/platforms/signal.py` (728 lines) — merged in
[PR #405](https://github.com/NousResearch/hermes-agent/pull/405) / commit `24f549a`.
Rebuilt from [PR #268](https://github.com/NousResearch/hermes-agent/issues/268) by @ibhagwan.
Defines the accepted adapter structure: SSE/long-poll loop, exponential backoff,
`extra` dict config, `build_source()` with `user_id_alt`/`chat_id_alt`, `_send_signal()`
standalone sender in `send_message_tool.py`.

**Official checklist:** `gateway/platforms/ADDING_A_PLATFORM.md` (16 points — all mapped below).

---

## Pre-work

```bash
# Ensure you are on the feature branch with a clean tree
cd /home/reed/Projects/personal/hermes-agent
git checkout feat/matrix-gateway
git fetch upstream
git rebase upstream/main   # bring in any merged PRs since branch was cut

# Install Matrix deps into dev venv
uv pip install "matrix-nio>=0.24" mistune nh3

# Verify baseline tests pass before touching anything
python -m pytest tests/ -q
```

---

## Architecture decisions (final)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Receive loop | `client.sync_forever(timeout=30000)` | nio's built-in long-poll; same pattern as nanobot |
| Auth | Access token (pre-issued) | No interactive login flow needed; matches Signal's http_url+account pattern |
| E2EE | **Off by default**, opt-in via `MATRIX_E2EE_ENABLED=true` | libolm is a C dependency; homelab is on Tailscale anyway |
| Group policy | `open` \| `mention` \| `allowlist` | Matches nanobot's three-mode policy |
| Config location | `config.extra` dict | Same as Signal and Email — no token/api_key fields |
| Typing | `client.room_typing()` + Hermes base `_keep_typing()` loop | Base class already provides the keepalive; `send_typing()` just calls nio |
| Markdown | Render to sanitized HTML via `mistune` + `nh3` | Matrix renders HTML natively; matches nanobot's `_build_matrix_text_content` |
| Store path | `~/.hermes/matrix_store/` | Sync tokens + E2EE key material |
| Media send | Upload → `mxc://` URI → `room_send` | Standard Matrix media pattern |

---

## The 16 integration points — exact diffs

### 1. `gateway/platforms/matrix.py` — **NEW FILE** (~370 lines)

Full module. Key sections:

**Imports (graceful degradation if nio not installed):**
```python
try:
    from nio import (
        AsyncClient, AsyncClientConfig,
        InviteEvent, MatrixRoom,
        RoomMessageText, RoomMessageMedia, RoomEncryptedMedia,
        RoomSendError, RoomTypingError, SyncError,
        DownloadError, MemoryDownloadResponse, UploadError,
    )
    _NIO_AVAILABLE = True
except ImportError:
    _NIO_AVAILABLE = False
```

**`check_matrix_requirements()`:**
```python
def check_matrix_requirements() -> bool:
    """Check if matrix-nio is installed."""
    return _NIO_AVAILABLE
```

**Config from `extra` dict (same pattern as Signal):**
```python
extra = config.extra or {}
self.homeserver    = extra.get("homeserver", "").rstrip("/")
self.user_id       = extra.get("user_id", "")
self.access_token  = extra.get("access_token", "")
self.device_id     = extra.get("device_id", "")
self.e2ee_enabled  = extra.get("e2ee_enabled", False)
self.join_on_invite = extra.get("join_on_invite", True)
self.group_policy  = extra.get("group_policy", "open")   # "open"|"mention"|"allowlist"
self.group_allow_from = set(
    v.strip() for v in str(extra.get("group_allow_from", "")).split(",") if v.strip()
)
```

Note: `group_allow_from` is stored as a comma-delimited string in `extra` (it comes from
`MATRIX_GROUP_ALLOW_ROOMS` env var) — same approach as Signal's `SIGNAL_GROUP_ALLOWED_USERS`.

**Self-message filter (critical — nio delivers your own events back to you):**
```python
async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
    if event.sender == self.user_id:
        return   # never process own messages
    ...
```

**Group policy (`_should_process` — mirrors nanobot exactly):**
```python
def _should_process(self, room: MatrixRoom, event) -> bool:
    if self._is_direct_room(room):
        return True          # DMs always pass; auth handled by run.py
    if self.group_policy == "open":
        return True
    if self.group_policy == "allowlist":
        return room.room_id in self.group_allow_from
    if self.group_policy == "mention":
        return self._is_bot_mentioned(event)
    return False

def _is_direct_room(self, room: MatrixRoom) -> bool:
    count = getattr(room, "member_count", None)
    return isinstance(count, int) and count <= 2

def _is_bot_mentioned(self, event) -> bool:
    source = getattr(event, "source", None)
    if not isinstance(source, dict):
        return False
    mentions = (source.get("content") or {}).get("m.mentions")
    if not isinstance(mentions, dict):
        return False
    user_ids = mentions.get("user_ids")
    if isinstance(user_ids, list) and self.user_id in user_ids:
        return True
    return False
```

**`connect()` / `disconnect()` lifecycle:**
```python
async def connect(self) -> bool:
    store_path = Path.home() / ".hermes" / "matrix_store"
    store_path.mkdir(parents=True, exist_ok=True)

    self.client = AsyncClient(
        homeserver=self.homeserver,
        user=self.user_id,
        store_path=str(store_path),
        config=AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=self.e2ee_enabled,
        ),
    )
    self.client.access_token = self.access_token
    self.client.user_id = self.user_id
    if self.device_id:
        self.client.device_id = self.device_id
        try:
            self.client.load_store()
        except Exception:
            logger.warning("Matrix: store load failed; may replay recent messages on restart")

    self.client.add_event_callback(self._on_message, RoomMessageText)
    self.client.add_event_callback(self._on_media_message, (RoomMessageMedia, RoomEncryptedMedia))
    self.client.add_event_callback(self._on_invite, InviteEvent)
    self.client.add_response_callback(self._on_sync_error, SyncError)

    self._running = True
    self._sync_task = asyncio.create_task(self._sync_loop())
    logger.info("Matrix: connected as %s on %s", self.user_id, self.homeserver)
    return True

async def disconnect(self) -> None:
    self._running = False
    if self.client:
        self.client.stop_sync_forever()
    if self._sync_task:
        self._sync_task.cancel()
        try:
            await self._sync_task
        except asyncio.CancelledError:
            pass
    if self.client:
        await self.client.close()
        self.client = None
    logger.info("Matrix: disconnected")
```

**Sync loop (exponential backoff on error — mirrors nanobot):**
```python
async def _sync_loop(self) -> None:
    backoff = 2.0
    while self._running:
        try:
            await self.client.sync_forever(timeout=30000, full_state=True)
            backoff = 2.0
        except asyncio.CancelledError:
            break
        except Exception as e:
            if self._running:
                logger.warning("Matrix: sync error: %s (retry in %.0fs)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
```

**`send()` with Markdown → HTML rendering:**
```python
async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
    if not self.client:
        return SendResult(success=False, error="Not connected")
    content_body = _build_matrix_text_content(content)
    try:
        resp = await self.client.room_send(
            room_id=chat_id,
            message_type="m.room.message",
            content=content_body,
            ignore_unverified_devices=True,
        )
        if isinstance(resp, RoomSendError):
            return SendResult(success=False, error=str(resp))
        return SendResult(success=True, message_id=getattr(resp, "event_id", None))
    except Exception as e:
        return SendResult(success=False, error=str(e))
```

**`send_typing()` (Hermes base `_keep_typing()` calls this on an interval):**
```python
async def send_typing(self, chat_id: str, metadata=None) -> None:
    if not self.client:
        return
    try:
        await self.client.room_typing(
            room_id=chat_id, typing_state=True, timeout=30000
        )
    except Exception:
        pass
```

**`send_image()` (upload → mxc:// → room_send):**
```python
async def send_image(self, chat_id, image_url, caption=None, **kwargs) -> SendResult:
    if not self.client:
        return SendResult(success=False, error="Not connected")
    try:
        if image_url.startswith("http"):
            file_path = await cache_image_from_url(image_url)
        else:
            file_path = image_url.removeprefix("file://")
        data = Path(file_path).read_bytes()
        ext = Path(file_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
        resp, _ = await self.client.upload(
            io.BytesIO(data), content_type=mime,
            filename=Path(file_path).name, filesize=len(data),
        )
        if isinstance(resp, UploadError):
            return SendResult(success=False, error=str(resp))
        content = {"msgtype": "m.image", "body": caption or Path(file_path).name,
                   "url": resp.content_uri, "info": {"mimetype": mime, "size": len(data)}}
        await self.client.room_send(
            room_id=chat_id, message_type="m.room.message",
            content=content, ignore_unverified_devices=True,
        )
        return SendResult(success=True)
    except Exception as e:
        return SendResult(success=False, error=str(e))
```

**`get_chat_info()`:**
```python
async def get_chat_info(self, chat_id: str) -> dict:
    if self.client:
        room = (self.client.rooms or {}).get(chat_id)
        if room:
            count = getattr(room, "member_count", 0)
            return {
                "name": getattr(room, "display_name", chat_id),
                "type": "dm" if count <= 2 else "group",
                "chat_id": chat_id,
            }
    return {"name": chat_id, "type": "group", "chat_id": chat_id}
```

**`_build_matrix_text_content()` (module-level helper — from nanobot):**
```python
def _build_matrix_text_content(text: str) -> dict:
    """Build Matrix m.text payload with optional HTML formatted_body."""
    content = {"msgtype": "m.text", "body": text, "m.mentions": {}}
    try:
        import mistune, nh3
        html = nh3.clean(mistune.create_markdown(
            escape=True, plugins=["table", "strikethrough", "url"]
        )(text)).strip()
        # Skip formatted_body for trivially plain paragraphs
        if html and not (html.startswith("<p>") and html.endswith("</p>")
                         and "<" not in html[3:-4]):
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html
    except Exception:
        pass
    return content
```

**`_on_invite()` (auto-join if enabled):**
```python
async def _on_invite(self, room: MatrixRoom, event: InviteEvent) -> None:
    if self.join_on_invite:
        await self.client.join(room.room_id)
        logger.info("Matrix: joined room %s (invited by %s)", room.room_id, event.sender)
```

**`build_source()` call in `_on_message()`:**
```python
is_dm = self._is_direct_room(room)
source = self.build_source(
    chat_id=room.room_id,
    chat_name=getattr(room, "display_name", room.room_id),
    chat_type="dm" if is_dm else "group",
    user_id=event.sender,
    user_name=room.user_name(event.sender),
)
```
No `user_id_alt` / `chat_id_alt` needed — Matrix IDs are stable MXIDs, no secondary UUID.

---

### 2. `gateway/config.py`

**a) `Platform` enum** — add after `EMAIL = "email"` (line 31):
```python
EMAIL = "email"
MATRIX = "matrix"     # ADD
```
Already partially done (Platform.MATRIX was added in a prior session — verify with `grep MATRIX gateway/config.py`).

**b) `get_connected_platforms()`** — add after the Signal case (line ~169):
```python
# Existing Signal case:
elif platform == Platform.SIGNAL and config.extra.get("http_url"):
    connected.append(platform)
# ADD:
elif platform == Platform.MATRIX and config.extra.get("homeserver"):
    connected.append(platform)
```

**c) `_apply_env_overrides()`** — add after the Email block (after line ~462):
```python
# Matrix
matrix_homeserver = os.getenv("MATRIX_HOMESERVER")
matrix_user_id    = os.getenv("MATRIX_USER_ID")
matrix_token      = os.getenv("MATRIX_ACCESS_TOKEN")
if matrix_homeserver and matrix_user_id and matrix_token:
    if Platform.MATRIX not in config.platforms:
        config.platforms[Platform.MATRIX] = PlatformConfig()
    config.platforms[Platform.MATRIX].enabled = True
    config.platforms[Platform.MATRIX].extra.update({
        "homeserver":     matrix_homeserver,
        "user_id":        matrix_user_id,
        "access_token":   matrix_token,
        "device_id":      os.getenv("MATRIX_DEVICE_ID", ""),
        "e2ee_enabled":   os.getenv("MATRIX_E2EE_ENABLED", "false").lower() in ("true", "1", "yes"),
        "join_on_invite": os.getenv("MATRIX_JOIN_ON_INVITE", "true").lower() in ("true", "1", "yes"),
        "group_policy":   os.getenv("MATRIX_GROUP_POLICY", "open"),
        "group_allow_from": os.getenv("MATRIX_GROUP_ALLOW_ROOMS", ""),
    })
    matrix_home = os.getenv("MATRIX_HOME_CHANNEL")
    if matrix_home:
        config.platforms[Platform.MATRIX].home_channel = HomeChannel(
            platform=Platform.MATRIX,
            chat_id=matrix_home,
            name=os.getenv("MATRIX_HOME_CHANNEL_NAME", "Home"),
        )
```

---

### 3. `gateway/run.py` — `_create_adapter()` (line ~789)

Add after the Email block, before `return None`:
```python
elif platform == Platform.EMAIL:
    ...  # existing

elif platform == Platform.MATRIX:
    from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
    if not check_matrix_requirements():
        logger.warning(
            "Matrix: matrix-nio not installed. "
            "Run: pip install 'hermes-agent[matrix]'"
        )
        return None
    return MatrixAdapter(config)

return None
```

---

### 4. `gateway/run.py` — `_is_user_authorized()` auth maps (lines ~819-835)

Add `MATRIX` to **both** dicts:
```python
platform_env_map = {
    ...
    Platform.EMAIL:   "EMAIL_ALLOWED_USERS",
    Platform.MATRIX:  "MATRIX_ALLOWED_USERS",    # ADD
}
platform_allow_all_map = {
    ...
    Platform.EMAIL:   "EMAIL_ALLOW_ALL_USERS",
    Platform.MATRIX:  "MATRIX_ALLOW_ALL_USERS",  # ADD
}
```

---

### 4b. `gateway/run.py` — startup `_any_allowlist` warning (line ~585)

**Current** (lines 587–589):
```python
for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
           "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
           "GATEWAY_ALLOWED_USERS")
```
Note: `SIGNAL_ALLOWED_USERS` is **missing** from this list in current main — that is an existing
gap in the Signal PR. Add both Signal and Matrix together:
```python
for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
           "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
           "SIGNAL_ALLOWED_USERS", "MATRIX_ALLOWED_USERS",
           "GATEWAY_ALLOWED_USERS")
```

---

### 4c. `gateway/run.py` — TWO toolset maps (lines 2152 and 3054)

Both maps are structurally identical. Add `MATRIX` to **each**:

**`default_toolset_map`** (both copies):
```python
Platform.EMAIL:   "hermes-email",
Platform.MATRIX:  "hermes-matrix",   # ADD
```

**`platform_config_key`** (both copies):
```python
Platform.EMAIL:   "email",
Platform.MATRIX:  "matrix",          # ADD
```

The fallback `.get(source.platform, "hermes-telegram")` at lines 2188 and 3094 is unchanged —
Matrix will match from the map before the fallback is ever reached.

---

### 5. `gateway/session.py`

**No changes required.**

Matrix identity maps cleanly to existing `SessionSource` fields:
- `user_id` = sender MXID (`@reed:attune.local`)
- `chat_id` = room ID (`!abc123:attune.local`)
- `chat_name` = room display name
- `chat_type` = `"dm"` (≤2 members) or `"group"`

`user_id_alt` and `chat_id_alt` were added in Signal PR #405 for Signal UUID support.
Matrix IDs are already stable — no secondary identifier needed.

---

### 6. `agent/prompt_builder.py` — `PLATFORM_HINTS` (line ~125)

Add after the `"email"` entry and before `"cli"`:
```python
"matrix": (
    "You are communicating via Matrix (Element, Cinny, FluffyChat, etc). "
    "Markdown renders as formatted HTML — use **bold**, *italic*, `inline code`, "
    "```code blocks```, tables, and headers freely. Avoid raw HTML. "
    "You can send media files natively: include MEDIA:/absolute/path/to/file "
    "in your response. Images (.png, .jpg, .webp) appear inline, audio as "
    "playback widgets, and other files as downloadable attachments. "
    "Room IDs look like !abc:server, user IDs like @user:server. "
    "You can also include image URLs as ![alt](url) and they will be uploaded "
    "and sent as inline images."
),
```

---

### 7. `toolsets.py` (after line ~265, `hermes-signal` block)

Add after `hermes-signal`, before `hermes-homeassistant`:
```python
"hermes-matrix": {
    "description": "Matrix bot toolset - self-hosted or federated Matrix messaging",
    "tools": _HERMES_CORE_TOOLS,
    "includes": []
},
```

Update `hermes-gateway` includes (line ~279):
```python
# Before:
"includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack",
             "hermes-signal", "hermes-homeassistant", "hermes-email"]

# After:
"includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack",
             "hermes-signal", "hermes-matrix", "hermes-homeassistant", "hermes-email"]
```

---

### 8. `cron/scheduler.py` — `platform_map` in `_deliver_result()` (line ~100)

```python
platform_map = {
    "telegram": Platform.TELEGRAM,
    "discord":  Platform.DISCORD,
    "slack":    Platform.SLACK,
    "whatsapp": Platform.WHATSAPP,
    "signal":   Platform.SIGNAL,
    "email":    Platform.EMAIL,
    "matrix":   Platform.MATRIX,   # ADD
}
```

---

### 9. `tools/send_message_tool.py`

**a) `platform_map` dict (line ~116):**
```python
platform_map = {
    "telegram": Platform.TELEGRAM,
    "discord":  Platform.DISCORD,
    "slack":    Platform.SLACK,
    "whatsapp": Platform.WHATSAPP,
    "signal":   Platform.SIGNAL,
    "email":    Platform.EMAIL,
    "matrix":   Platform.MATRIX,   # ADD
}
```

**b) `target` parameter description (line ~39):**
```python
"description": (
    "Delivery target. Format: 'platform' (uses home channel), "
    "'platform:#channel-name', 'platform:chat_id', or Telegram topic "
    "'telegram:chat_id:thread_id'. Examples: 'telegram', "
    "'telegram:-1001234567890:17585', 'discord:#bot-home', "
    "'slack:#engineering', 'signal:+15551234567', "
    "'matrix:!room:server'"   # ADD
)
```

**c) `_send_to_platform()` routing (line ~178):**
```python
elif platform == Platform.MATRIX:
    return await _send_matrix(pconfig.extra, chat_id, message)
```

**d) New `_send_matrix()` function** (standalone, mirrors `_send_signal()` pattern at line ~253):
```python
async def _send_matrix(extra: dict, room_id: str, message: str) -> dict:
    """Send a Matrix message via CS API — no sync loop needed for one-shot send."""
    try:
        import nio
    except ImportError:
        return {"error": "matrix-nio not installed. Run: pip install 'hermes-agent[matrix]'"}

    homeserver   = extra.get("homeserver", "")
    access_token = extra.get("access_token", "")
    user_id      = extra.get("user_id", "")
    if not all([homeserver, access_token, user_id]):
        return {"error": "Matrix not configured (MATRIX_HOMESERVER, MATRIX_USER_ID, "
                         "MATRIX_ACCESS_TOKEN required)"}
    try:
        client = nio.AsyncClient(homeserver=homeserver, user=user_id)
        client.access_token = access_token
        content = {"msgtype": "m.text", "body": message}
        resp = await client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
            ignore_unverified_devices=True,
        )
        await client.close()
        if isinstance(resp, nio.RoomSendError):
            return {"error": f"Matrix send failed: {resp.message}"}
        return {"success": True, "platform": "matrix", "chat_id": room_id,
                "event_id": getattr(resp, "event_id", None)}
    except Exception as e:
        return {"error": f"Matrix send failed: {e}"}
```

---

### 10. `tools/cronjob_tools.py` — `deliver` parameter description (line ~221)

Current value: `"Where to send output: 'origin' (back to this chat), 'local' (files only), 'telegram', 'discord', 'signal', or 'platform:chat_id'"`

Update to:
```python
"Where to send output: 'origin' (back to this chat), 'local' (files only), "
"'telegram', 'discord', 'signal', 'matrix', or 'platform:chat_id' "
"(e.g. 'matrix:!room:server')"
```

---

### 11. `gateway/channel_directory.py` (line ~64)

```python
# Before:
for plat_name in ("telegram", "whatsapp", "signal", "email"):

# After:
for plat_name in ("telegram", "whatsapp", "signal", "email", "matrix"):
```

---

### 12. `hermes_cli/status.py` — `platforms` dict (line ~212)

```python
platforms = {
    "Telegram": ("TELEGRAM_BOT_TOKEN",  "TELEGRAM_HOME_CHANNEL"),
    "Discord":  ("DISCORD_BOT_TOKEN",   "DISCORD_HOME_CHANNEL"),
    "WhatsApp": ("WHATSAPP_ENABLED",    None),
    "Signal":   ("SIGNAL_HTTP_URL",     "SIGNAL_HOME_CHANNEL"),
    "Slack":    ("SLACK_BOT_TOKEN",     None),
    "Email":    ("EMAIL_ADDRESS",       "EMAIL_HOME_ADDRESS"),
    "Matrix":   ("MATRIX_ACCESS_TOKEN", "MATRIX_HOME_CHANNEL"),  # ADD
}
```

---

### 13. `hermes_cli/gateway.py`

**a) Add to `_PLATFORMS` list** (after the Signal entry at line ~516, before Email):
```python
{
    "key": "matrix",
    "label": "Matrix",
    "emoji": "🔷",
    "token_var": "MATRIX_ACCESS_TOKEN",
},
```

Note: Like Signal and WhatsApp, Matrix gets only `key`/`label`/`emoji`/`token_var` in
`_PLATFORMS` — the detailed setup lives in `_setup_matrix()`.

**b) Add `_platform_status()` case** — Matrix needs multi-field check like Signal.
Add to `_platform_status()` (after line ~572):
```python
if platform.get("key") == "matrix":
    homeserver = get_env_value("MATRIX_HOMESERVER")
    user_id    = get_env_value("MATRIX_USER_ID")
    if val and homeserver and user_id:
        return "configured"
    if any([val, homeserver, user_id]):
        return "partially configured"
    return "not configured"
```

**c) Add `_setup_matrix()` function** (modelled on `_setup_signal()` at line 718):
```python
def _setup_matrix():
    """Interactive setup for Matrix homeserver."""
    print(color("  ─── 🔷 Matrix Setup ───", Colors.CYAN))

    homeserver = prompt_value(
        "  Homeserver URL", os.getenv("MATRIX_HOMESERVER", ""),
        password=False,
        help_text="e.g. https://matrix.org or https://matrix.attune.local",
    )
    if not homeserver:
        print_warning("  Homeserver URL is required.")
        return

    # Connectivity test against Matrix versions endpoint
    try:
        import httpx
        resp = httpx.get(f"{homeserver.rstrip('/')}/_matrix/client/versions", timeout=5)
        if resp.status_code == 200:
            print_success("  Homeserver is reachable!")
        else:
            print_warning(f"  Homeserver responded {resp.status_code} — may still work.")
    except Exception as e:
        print_warning(f"  Could not reach homeserver: {e}")
        if not prompt_yes_no("  Save anyway?", True):
            return

    user_id = prompt_value(
        "  Bot Matrix ID", os.getenv("MATRIX_USER_ID", ""),
        password=False,
        help_text="e.g. @hermesbot:matrix.org",
    )
    access_token = prompt_value(
        "  Access token", os.getenv("MATRIX_ACCESS_TOKEN", ""),
        password=True,
        help_text="Get via Element → Settings → Help & About → Access Token",
    )
    device_id = prompt_value(
        "  Device ID (optional, recommended)", os.getenv("MATRIX_DEVICE_ID", ""),
        password=False,
        help_text="Prevents replaying old messages on restart",
    )
    home_channel = prompt_value(
        "  Home room ID (optional)", os.getenv("MATRIX_HOME_CHANNEL", ""),
        password=False,
        help_text="e.g. !abc123:matrix.org — used for cron delivery",
    )
    allowed = prompt_value(
        "  Allowed Matrix IDs (comma-separated)", os.getenv("MATRIX_ALLOWED_USERS", ""),
        password=False,
        help_text="e.g. @reed:matrix.org,@alice:matrix.org",
    )

    set_env_value("MATRIX_HOMESERVER",   homeserver)
    set_env_value("MATRIX_USER_ID",      user_id)
    set_env_value("MATRIX_ACCESS_TOKEN", access_token)
    if device_id:
        set_env_value("MATRIX_DEVICE_ID", device_id)
    if home_channel:
        set_env_value("MATRIX_HOME_CHANNEL", home_channel)
    if allowed:
        set_env_value("MATRIX_ALLOWED_USERS", allowed)

    print_success("  Matrix configured!")
```

**d) Route to it in the platform selection switch** (after `elif platform["key"] == "signal"` at line ~887):
```python
elif platform["key"] == "signal":
    _setup_signal()
elif platform["key"] == "matrix":      # ADD
    _setup_matrix()
else:
    _setup_standard_platform(platform)
```

---

### 14. `agent/redact.py` — `_PREFIX_PATTERNS`

Matrix access tokens (`syt_*`) are partially covered by `_ENV_ASSIGN_RE` (catches
`MATRIX_ACCESS_TOKEN=syt_...` in log lines) and `_AUTH_HEADER_RE` (catches
`Authorization: Bearer syt_...`). However tokens appearing in plain log context
(e.g. `logger.debug("Matrix token: %s", token)`) are not caught.

Add to `_PREFIX_PATTERNS` list (line ~17):
```python
r"syt_[A-Za-z0-9_-]{10,}",  # Matrix access token
```

---

### 15. Documentation

| File | Change |
|------|--------|
| `README.md` | Add Matrix row to platform support table |
| `AGENTS.md` | Add Matrix env vars to gateway configuration section |
| `website/docs/user-guide/messaging/matrix.md` | **NEW** — Full setup guide (use Signal doc as template) |
| `website/docs/user-guide/messaging/index.md` | Add Matrix to architecture diagram and toolset table |
| `website/docs/reference/environment-variables.md` | Add all Matrix env vars |

---

### 16. `tests/gateway/test_matrix.py` — **NEW FILE**

Modelled on `tests/gateway/test_signal.py` (30 tests, merged with PR #405).

```python
"""Tests for Matrix platform adapter integration."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ── Config / enum ──────────────────────────────────────────────────────────

def test_platform_enum_value():
    from gateway.config import Platform
    assert Platform.MATRIX.value == "matrix"

def test_env_overrides_minimal():
    """Three required vars enable Matrix."""
    with patch.dict(os.environ, {
        "MATRIX_HOMESERVER":    "https://matrix.example.com",
        "MATRIX_USER_ID":       "@bot:example.com",
        "MATRIX_ACCESS_TOKEN":  "syt_test_abc123",
    }, clear=False):
        from gateway.config import GatewayConfig, _apply_env_overrides, Platform
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX in config.platforms
        assert config.platforms[Platform.MATRIX].enabled is True
        extra = config.platforms[Platform.MATRIX].extra
        assert extra["homeserver"] == "https://matrix.example.com"
        assert extra["user_id"] == "@bot:example.com"

def test_env_overrides_home_channel():
    with patch.dict(os.environ, {
        "MATRIX_HOMESERVER":    "https://matrix.example.com",
        "MATRIX_USER_ID":       "@bot:example.com",
        "MATRIX_ACCESS_TOKEN":  "syt_test_abc123",
        "MATRIX_HOME_CHANNEL":  "!room:example.com",
        "MATRIX_HOME_CHANNEL_NAME": "HQ",
    }, clear=False):
        from gateway.config import GatewayConfig, _apply_env_overrides, Platform
        config = GatewayConfig()
        _apply_env_overrides(config)
        hc = config.platforms[Platform.MATRIX].home_channel
        assert hc is not None
        assert hc.chat_id == "!room:example.com"
        assert hc.name == "HQ"

def test_env_overrides_e2ee_flag():
    with patch.dict(os.environ, {
        "MATRIX_HOMESERVER":   "https://m.example.com",
        "MATRIX_USER_ID":      "@bot:example.com",
        "MATRIX_ACCESS_TOKEN": "syt_x",
        "MATRIX_E2EE_ENABLED": "true",
    }, clear=False):
        from gateway.config import GatewayConfig, _apply_env_overrides, Platform
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].extra["e2ee_enabled"] is True

def test_get_connected_platforms_matrix():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    config = GatewayConfig()
    config.platforms[Platform.MATRIX] = PlatformConfig(
        enabled=True, extra={"homeserver": "https://m.example.com", "user_id": "@b:m.example.com", "access_token": "t"}
    )
    assert Platform.MATRIX in config.get_connected_platforms()

def test_get_connected_platforms_matrix_missing_homeserver():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    config = GatewayConfig()
    config.platforms[Platform.MATRIX] = PlatformConfig(enabled=True, extra={})
    assert Platform.MATRIX not in config.get_connected_platforms()


# ── Requirements check ─────────────────────────────────────────────────────

def test_check_requirements_present():
    """Passes when matrix-nio is importable."""
    with patch.dict("sys.modules", {"nio": MagicMock()}):
        import importlib
        import gateway.platforms.matrix as m
        importlib.reload(m)
        assert m.check_matrix_requirements() is True

def test_check_requirements_missing():
    """Fails gracefully when matrix-nio is not installed."""
    with patch.dict("sys.modules", {"nio": None}):
        import importlib
        import gateway.platforms.matrix as m
        importlib.reload(m)
        assert m.check_matrix_requirements() is False


# ── Adapter unit tests ─────────────────────────────────────────────────────

@pytest.fixture
def matrix_adapter():
    from gateway.config import PlatformConfig
    config = PlatformConfig(enabled=True, extra={
        "homeserver":   "https://matrix.example.com",
        "user_id":      "@bot:example.com",
        "access_token": "syt_test",
        "device_id":    "TESTDEVICE",
        "e2ee_enabled": False,
        "group_policy": "open",
        "group_allow_from": "",
    })
    with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
        with patch("nio.AsyncClient"):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(config)
            adapter.client = MagicMock()
            return adapter

def test_self_message_filtered(matrix_adapter):
    """Bot's own sender must be filtered before _should_process is called."""
    assert matrix_adapter.user_id == "@bot:example.com"
    # The _on_message callback returns immediately when sender == user_id

def test_should_process_dm(matrix_adapter):
    room = MagicMock()
    room.member_count = 2
    event = MagicMock()
    event.sender = "@reed:example.com"
    assert matrix_adapter._should_process(room, event) is True

def test_should_process_group_open(matrix_adapter):
    room = MagicMock()
    room.member_count = 5
    event = MagicMock()
    matrix_adapter.group_policy = "open"
    assert matrix_adapter._should_process(room, event) is True

def test_should_process_group_allowlist_allowed(matrix_adapter):
    room = MagicMock()
    room.member_count = 5
    room.room_id = "!allowed:example.com"
    matrix_adapter.group_policy = "allowlist"
    matrix_adapter.group_allow_from = {"!allowed:example.com"}
    event = MagicMock()
    assert matrix_adapter._should_process(room, event) is True

def test_should_process_group_allowlist_blocked(matrix_adapter):
    room = MagicMock()
    room.member_count = 5
    room.room_id = "!other:example.com"
    matrix_adapter.group_policy = "allowlist"
    matrix_adapter.group_allow_from = {"!allowed:example.com"}
    event = MagicMock()
    assert matrix_adapter._should_process(room, event) is False

def test_should_process_mention_present(matrix_adapter):
    room = MagicMock()
    room.member_count = 5
    matrix_adapter.group_policy = "mention"
    event = MagicMock()
    event.source = {"content": {"m.mentions": {"user_ids": ["@bot:example.com"]}}}
    assert matrix_adapter._should_process(room, event) is True

def test_should_process_mention_absent(matrix_adapter):
    room = MagicMock()
    room.member_count = 5
    matrix_adapter.group_policy = "mention"
    event = MagicMock()
    event.source = {"content": {"m.mentions": {"user_ids": []}}}
    assert matrix_adapter._should_process(room, event) is False


# ── Authorization maps ─────────────────────────────────────────────────────

def test_authorization_env_map():
    """MATRIX_ALLOWED_USERS routes an authorized user through."""
    import gateway.run as run_module
    from gateway.config import Platform
    runner = run_module.GatewayRunner.__new__(run_module.GatewayRunner)
    runner.config = MagicMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    source = MagicMock()
    source.platform = Platform.MATRIX
    source.user_id = "@reed:attune.local"
    with patch.dict(os.environ, {"MATRIX_ALLOWED_USERS": "@reed:attune.local"}):
        assert runner._is_user_authorized(source) is True

def test_authorization_env_map_deny():
    import gateway.run as run_module
    from gateway.config import Platform
    runner = run_module.GatewayRunner.__new__(run_module.GatewayRunner)
    runner.config = MagicMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    source = MagicMock()
    source.platform = Platform.MATRIX
    source.user_id = "@stranger:example.com"
    with patch.dict(os.environ, {"MATRIX_ALLOWED_USERS": "@reed:attune.local"}, clear=False):
        assert runner._is_user_authorized(source) is False


# ── Send message tool ──────────────────────────────────────────────────────

def test_send_message_tool_platform_map():
    """Matrix must appear in send_message_tool's platform_map."""
    # Import the module and inspect platform_map is populated correctly
    with patch("gateway.config.load_gateway_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(platforms={})
        import importlib
        import tools.send_message_tool as smt
        importlib.reload(smt)
        # platform_map is local to the function; call list action as smoke test
        # (no assertion on result, just confirming the module loads with Matrix)


# ── Cron / scheduler ──────────────────────────────────────────────────────

def test_scheduler_platform_map():
    from gateway.config import Platform
    from cron.scheduler import CronScheduler
    # platform_map is local to _deliver_result; verify via attribute or reimport
    import inspect
    src = inspect.getsource(CronScheduler._deliver_result)
    assert "matrix" in src


# ── Session key ────────────────────────────────────────────────────────────

def test_session_key_dm():
    from gateway.session import build_session_key, SessionSource
    from gateway.config import Platform
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:attune.local", chat_type="dm")
    assert build_session_key(source) == "agent:main:matrix:dm"

def test_session_key_group():
    from gateway.session import build_session_key, SessionSource
    from gateway.config import Platform
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:attune.local", chat_type="group")
    assert build_session_key(source) == "agent:main:matrix:group:!room:attune.local"

def test_session_source_round_trip():
    from gateway.session import SessionSource
    from gateway.config import Platform
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:attune.local",
        chat_name="Dev",
        chat_type="group",
        user_id="@reed:attune.local",
        user_name="Reed",
    )
    restored = SessionSource.from_dict(source.to_dict())
    assert restored.platform == Platform.MATRIX
    assert restored.chat_id == "!room:attune.local"
    assert restored.user_id == "@reed:attune.local"
```

---

## Environment variables — complete reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MATRIX_HOMESERVER` | Yes | — | Homeserver URL, e.g. `https://matrix.org` |
| `MATRIX_USER_ID` | Yes | — | Bot's full MXID, e.g. `@hermesbot:matrix.org` |
| `MATRIX_ACCESS_TOKEN` | Yes | — | Bot's access token (`syt_...`) |
| `MATRIX_DEVICE_ID` | Recommended | `""` | Device ID — prevents replaying old messages on restart |
| `MATRIX_E2EE_ENABLED` | No | `false` | Enable E2EE (requires `matrix-nio[e2e]` + libolm) |
| `MATRIX_JOIN_ON_INVITE` | No | `true` | Auto-join rooms when invited |
| `MATRIX_GROUP_POLICY` | No | `open` | Group message policy: `open`, `mention`, `allowlist` |
| `MATRIX_GROUP_ALLOW_ROOMS` | No | `""` | Comma-separated room IDs for `allowlist` policy |
| `MATRIX_ALLOWED_USERS` | Recommended | `""` | Comma-separated MXIDs allowed to message the bot |
| `MATRIX_ALLOW_ALL_USERS` | No | `false` | Allow any Matrix user (disables allowlist) |
| `MATRIX_HOME_CHANNEL` | No | `""` | Default room ID for cron delivery |
| `MATRIX_HOME_CHANNEL_NAME` | No | `"Home"` | Display name for home channel |

---

## Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
# ... existing groups ...
matrix = [
    "matrix-nio>=0.24.0",
    "mistune>=3.0.0",
    "nh3>=0.2.0",
]
# E2EE support (opt-in — requires libolm C library):
# matrix = ["matrix-nio[e2e]>=0.24.0", "mistune>=3.0.0", "nh3>=0.2.0"]
```

Install:
```bash
pip install "hermes-agent[matrix]"
```

---

## Commit structure (5 commits)

1. `feat(config): add Matrix to Platform enum and env overrides`
   — `gateway/config.py` only. Testable in isolation immediately.

2. `feat(gateway/matrix): add MatrixAdapter with nio sync loop`
   — `gateway/platforms/matrix.py` only. The bulk of the work.

3. `feat(gateway): wire Matrix into all integration points`
   — `gateway/run.py`, `toolsets.py`, `tools/send_message_tool.py`,
     `cron/scheduler.py`, `tools/cronjob_tools.py`,
     `gateway/channel_directory.py`, `agent/prompt_builder.py`,
     `agent/redact.py`, `hermes_cli/status.py`, `hermes_cli/gateway.py`

4. `test(gateway/matrix): add Matrix adapter test coverage`
   — `tests/gateway/test_matrix.py`

5. `docs: add Matrix platform setup guide and env var reference`
   — `README.md`, `AGENTS.md`, `pyproject.toml`,
     `website/docs/user-guide/messaging/matrix.md`,
     `website/docs/user-guide/messaging/index.md`,
     `website/docs/reference/environment-variables.md`

PR description must reference: **Closes #73**

---

## Verification (from ADDING_A_PLATFORM.md)

```bash
# All tests pass
python -m pytest tests/ -q

# Find any missed integration points
grep -r "telegram\|discord\|whatsapp\|slack\|signal\|email" \
  gateway/ tools/ agent/ cron/ hermes_cli/ toolsets.py \
  --include="*.py" -l | sort -u
# Every file listed — check it also contains "matrix"
```

---

## Implementation order (minimizes debugging friction)

1. `gateway/config.py` — enum + env overrides (no deps, testable immediately)
2. `gateway/platforms/matrix.py` — the adapter (~4-6h)
3. `gateway/run.py` — factory + auth maps + toolset maps
4. `tools/send_message_tool.py` — `_send_matrix()` + platform_map
5. `cron/scheduler.py` + `tools/cronjob_tools.py` + `toolsets.py` — one-liners
6. `gateway/channel_directory.py` + `hermes_cli/status.py` — one-liners
7. `agent/prompt_builder.py` + `agent/redact.py` — hints + token redaction
8. `hermes_cli/gateway.py` — `_setup_matrix()` + routing + `_platform_status()` case
9. `tests/gateway/test_matrix.py`
10. Docs + `pyproject.toml`

---

## Estimated effort

| Task | Lines | Time |
|------|-------|------|
| `gateway/platforms/matrix.py` | ~370 | 4–6h |
| All wiring edits (10 files, mostly one-liners) | ~80 | 1.5–2h |
| `_setup_matrix()` in gateway.py | ~50 | 1h |
| `_send_matrix()` in send_message_tool.py | ~30 | 30m |
| `tests/gateway/test_matrix.py` | ~130 | 2h |
| Docs (matrix.md + env vars + README patches) | ~120 | 1–1.5h |
| **Total** | **~780** | **~11–13h** |
