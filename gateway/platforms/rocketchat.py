"""Rocket.Chat gateway adapter.

Connects to a self-hosted (or cloud) Rocket.Chat instance via its REST API
(v1) and WebSocket (DDP / Meteor real-time API) for receiving messages.
No external Rocket.Chat library required — uses aiohttp which is already a
Hermes dependency.

Authentication (in priority order):
  1. Username + password  (ROCKETCHAT_USERNAME / ROCKETCHAT_PASSWORD) — recommended.
     The adapter logs in via POST /api/v1/login, obtains a session authToken
     and userId automatically, then uses the same token to authenticate the
     DDP WebSocket via the `resume` method.
  2. Personal Access Token fallback  (ROCKETCHAT_TOKEN / ROCKETCHAT_USER_ID)
     for setups that already have a PAT configured.

Environment variables:
    ROCKETCHAT_URL                   Server URL (e.g. https://chat.example.com)
    ROCKETCHAT_USERNAME              Bot account username  (recommended)
    ROCKETCHAT_PASSWORD              Bot account password  (recommended)
    ROCKETCHAT_TOKEN                 Personal access token (PAT fallback)
    ROCKETCHAT_USER_ID               User ID for PAT fallback
    ROCKETCHAT_ALLOWED_USERS         Comma-separated user IDs allowed to use the bot
    ROCKETCHAT_ALLOW_ALL_USERS       Set to "true" to allow all users
    ROCKETCHAT_HOME_CHANNEL          Room ID for cron/notification delivery
    ROCKETCHAT_REQUIRE_MENTION       Require @mention in channels (default: true)
    ROCKETCHAT_FREE_RESPONSE_CHANNELS  Room IDs where bot responds without @mention
    ROCKETCHAT_REPLY_IN_THREAD       Reply inside thread when message is in a thread, channels only (default: false)
    ROCKETCHAT_REACTIONS             Add emoji reactions (👀/✅/❌) while processing, channels only (default: true)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Rocket.Chat message size limit (server default).
MAX_MESSAGE_LENGTH = 5000

# Room type codes returned by the Rocket.Chat API.
_ROOM_TYPE_MAP = {
    "d": "dm",
    "c": "channel",
    "p": "group",
    "l": "channel",  # livechat → treat as channel
}

# Reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2

# DDP subscription ID for all incoming messages.
_SUBSCRIPTION_ID = "hermes-room-messages"


def _convert_double_bold(text: str) -> str:
    """Convert **text** → *text* for Rocket.Chat compatibility.

    Rocket.Chat uses *text* for bold, not **text**.  The standard-Markdown
    double-asterisk form causes the parser to mis-interpret inner boundaries
    (e.g. "**2.3°C**, **max**" renders with strikethrough artefacts because RC
    treats each lone * as a bold toggle).
    """
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)


def _balance_marker(text: str, marker: str) -> str:
    """Remove the last occurrence of *marker* if its count is odd.

    An odd number of bold/italic markers means one is unmatched, which causes
    Rocket.Chat to render the literal characters or produce unexpected formatting
    (e.g. strikethrough). Dropping the last orphan restores balanced Markdown.
    """
    parts = text.split(marker)
    if len(parts) % 2 == 0:  # odd number of markers → unbalanced
        last = text.rfind(marker)
        return text[:last] + text[last + len(marker):]
    return text


def check_rocketchat_requirements() -> bool:
    """Return True if the Rocket.Chat adapter can be used."""
    url = os.getenv("ROCKETCHAT_URL", "")
    if not url:
        logger.warning("Rocket.Chat: ROCKETCHAT_URL not set")
        return False

    # Accept username+password OR PAT.
    has_userpass = bool(
        os.getenv("ROCKETCHAT_USERNAME") and os.getenv("ROCKETCHAT_PASSWORD")
    )
    has_pat = bool(
        os.getenv("ROCKETCHAT_TOKEN") and os.getenv("ROCKETCHAT_USER_ID")
    )
    if not has_userpass and not has_pat:
        logger.debug(
            "Rocket.Chat: set ROCKETCHAT_USERNAME + ROCKETCHAT_PASSWORD "
            "(or ROCKETCHAT_TOKEN + ROCKETCHAT_USER_ID as fallback)"
        )
        return False

    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Rocket.Chat: aiohttp not installed")
        return False


class RocketChatAdapter(BasePlatformAdapter):
    """Gateway adapter for Rocket.Chat (self-hosted or cloud)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ROCKETCHAT)

        self._base_url: str = (
            config.extra.get("url", "")
            or os.getenv("ROCKETCHAT_URL", "")
        ).rstrip("/")

        # Credentials — username/password takes priority over PAT.
        self._username: str = (
            config.extra.get("username", "")
            or os.getenv("ROCKETCHAT_USERNAME", "")
        )
        self._password: str = (
            config.extra.get("password", "")
            or os.getenv("ROCKETCHAT_PASSWORD", "")
        )

        # PAT fallback — populated from env or set during login.
        self._auth_token: str = (
            config.token
            or config.extra.get("token", "")
            or os.getenv("ROCKETCHAT_TOKEN", "")
        )
        self._user_id: str = (
            config.extra.get("user_id", "")
            or os.getenv("ROCKETCHAT_USER_ID", "")
        )

        # Resolved at connect() time.
        self._bot_username: str = ""

        # aiohttp session + websocket handle.
        self._session: Any = None   # aiohttp.ClientSession
        self._ws: Any = None        # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._closing = False

        # Reply inside thread when the incoming message is in a thread (channels only).
        self._reply_in_thread: bool = (
            config.extra.get("reply_in_thread", "")
            or os.getenv("ROCKETCHAT_REPLY_IN_THREAD", "false")
        ).lower() in ("true", "1", "yes")

        # Add emoji reactions (👀 / ✅ / ❌) while processing messages.
        self._reactions_enabled: bool = (
            config.extra.get("reactions", "")
            or os.getenv("ROCKETCHAT_REACTIONS", "true")
        ).lower() not in ("false", "0", "no")

        # Dedup cache: msg_id → timestamp (prevent reprocessing duplicates).
        self._seen_msgs: Dict[str, float] = {}
        self._SEEN_MAX = 2000
        self._SEEN_TTL = 300  # 5 minutes

        # Recent-attachment buffer: room_id → list of (timestamp, local_path, mime).
        # Lets a follow-up text message reference a file uploaded moments earlier.
        self._recent_attachments: Dict[str, List[tuple]] = {}
        self._ATTACHMENT_TTL = 300  # 5 minutes

        # Room-type cache: room_id → "d" | "c" (avoid repeated API lookups).
        self._room_type_cache: Dict[str, str] = {}

        # DDP sequence counter.
        self._ddp_seq = 0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _login(self) -> bool:
        """Authenticate via username/password and store authToken + userId.

        Returns True on success.  Falls through to PAT if credentials are
        not configured.
        """
        if not self._username or not self._password:
            # Already have a PAT — nothing to do.
            return bool(self._auth_token and self._user_id)

        import aiohttp

        url = f"{self._base_url}/api/v1/login"
        payload = {"user": self._username, "password": self._password}
        try:
            async with self._session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                body = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("Rocket.Chat: login request failed: %s", exc)
            return False

        if body.get("status") != "success":
            logger.error(
                "Rocket.Chat: login failed for user '%s': %s",
                self._username,
                body.get("message", body.get("error", "unknown error")),
            )
            return False

        data = body.get("data", {})
        self._auth_token = data.get("authToken", "")
        self._user_id = data.get("userId", "")

        if not self._auth_token or not self._user_id:
            logger.error("Rocket.Chat: login response missing authToken or userId")
            return False

        logger.info(
            "Rocket.Chat: logged in as '%s' (userId=%s)", self._username, self._user_id
        )
        return True

    async def _logout(self) -> None:
        """Invalidate the current session token."""
        if not self._auth_token or not self._username:
            return  # PAT — never logged in, nothing to invalidate
        try:
            await self._api_post("logout", {})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Auth-Token": self._auth_token,
            "X-User-Id": self._user_id,
            "Content-Type": "application/json",
        }

    async def _api_get(
        self, path: str, params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """GET /api/v1/{path}."""
        import aiohttp
        url = f"{self._base_url}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.get(
                url, headers=self._headers(), params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("RC API GET %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("RC API GET %s network error: %s", path, exc)
            return {}

    async def _api_post(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/v1/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.post(
                url, headers=self._headers(), json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("RC API POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("RC API POST %s network error: %s", path, exc)
            return {}

    async def _upload_file(
        self,
        room_id: str,
        file_data: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        description: str = "",
    ) -> Optional[str]:
        """Upload a file to a room. Returns the message ID on success."""
        import aiohttp

        url = f"{self._base_url}/api/v1/rooms.upload/{room_id}"
        form = aiohttp.FormData()
        form.add_field("file", file_data, filename=filename, content_type=content_type)
        if description:
            form.add_field("description", description)

        headers = {
            "X-Auth-Token": self._auth_token,
            "X-User-Id": self._user_id,
        }
        try:
            async with self._session.post(
                url, headers=headers, data=form,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("RC file upload → %s: %s", resp.status, body[:200])
                    return None
                data = await resp.json()
                return data.get("message", {}).get("_id")
        except aiohttp.ClientError as exc:
            logger.error("RC file upload error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Authenticate and start the WebSocket listener."""
        import aiohttp

        if not self._base_url:
            logger.error("Rocket.Chat: ROCKETCHAT_URL not configured")
            return False

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._closing = False

        # Step 1 — authenticate (username/password or PAT).
        if not await self._login():
            await self._session.close()
            return False

        # Step 2 — verify identity and fetch bot username.
        me = await self._api_get("me")
        if not me or "username" not in me:
            logger.error(
                "Rocket.Chat: failed to verify identity after login — "
                "check ROCKETCHAT_URL and credentials"
            )
            await self._session.close()
            return False

        self._bot_username = me.get("username", self._username)
        logger.info(
            "Rocket.Chat: authenticated as @%s (%s) on %s",
            self._bot_username,
            self._user_id,
            self._base_url,
        )

        # Step 3 — start WebSocket listener in the background.
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Rocket.Chat and invalidate the session."""
        self._closing = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        await self._logout()

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Rocket.Chat: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (or multiple chunks) to a room."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        last_id = None
        for chunk in chunks:
            msg: Dict[str, Any] = {"rid": chat_id, "msg": chunk}
            is_dm = self._room_type_cache.get(chat_id) == "d"
            if reply_to and self._reply_in_thread and not is_dm:
                msg["tmid"] = reply_to

            data = await self._api_post("chat.sendMessage", {"message": msg})
            if not data or not data.get("success"):
                return SendResult(success=False, error="Failed to send message")
            last_id = data.get("message", {}).get("_id")

        return SendResult(success=True, message_id=last_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return room name and type."""
        data = await self._api_get("rooms.info", {"roomId": chat_id})
        room = data.get("room")
        if room:
            raw_type = room.get("t", "c")
            name = room.get("name") or (room.get("usernames") or [chat_id])[0]
            return {"name": name, "type": _ROOM_TYPE_MAP.get(raw_type, "channel")}
        return {"name": chat_id, "type": "channel"}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a typing indicator via the DDP real-time API."""
        if self._ws and not self._ws.closed:
            self._ddp_seq += 1
            try:
                await self._ws.send_json({
                    "msg": "method",
                    "method": "stream-notify-room",
                    "id": str(self._ddp_seq),
                    "params": [f"{chat_id}/typing", self._bot_username, True],
                })
            except Exception:
                pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop typing indicator."""
        if self._ws and not self._ws.closed:
            self._ddp_seq += 1
            try:
                await self._ws.send_json({
                    "msg": "method",
                    "method": "stream-notify-room",
                    "id": str(self._ddp_seq),
                    "params": [f"{chat_id}/typing", self._bot_username, False],
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    async def _set_reaction(self, message_id: str, emoji: str, should_react: bool) -> bool:
        """Add or remove an emoji reaction on a message.

        ``emoji`` should be a plain name without colons, e.g. ``"eyes"``.
        Returns True on success.
        """
        emoji_with_colons = f":{emoji}:"
        data = await self._api_post(
            "chat.react",
            {"messageId": message_id, "emoji": emoji_with_colons, "shouldReact": should_react},
        )
        return bool(data and data.get("success"))

    async def _add_reaction(self, message_id: str, emoji: str) -> bool:
        return await self._set_reaction(message_id, emoji, True)

    async def _remove_reaction(self, message_id: str, emoji: str) -> bool:
        return await self._set_reaction(message_id, emoji, False)

    def _reactions_active(self, event: "MessageEvent") -> bool:
        """Return True if emoji reactions should be used for this event."""
        return (
            self._reactions_enabled
            and event.message_id is not None
            and getattr(event.source, "chat_type", None) != "dm"
        )

    async def on_processing_start(self, event: "MessageEvent") -> None:
        """Add 👀 reaction while the agent is processing."""
        if self._reactions_active(event):
            await self._add_reaction(event.message_id, "eyes")

    async def on_processing_complete(self, event: "MessageEvent", success: bool) -> None:
        """Replace 👀 with ✅ or ❌ when processing is done."""
        if self._reactions_active(event):
            await self._remove_reaction(event.message_id, "eyes")
            await self._add_reaction(event.message_id, "white_check_mark" if success else "x")

    async def edit_message(
        self, chat_id: str, message_id: str, content: str
    ) -> SendResult:
        """Edit an existing message."""
        data = await self._api_post(
            "chat.update",
            {"roomId": chat_id, "msgId": message_id, "text": self.format_message(content)},
        )
        if not data or not data.get("success"):
            return SendResult(success=False, error="Failed to edit message")
        return SendResult(success=True, message_id=message_id)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_url_as_file(chat_id, image_url, caption, reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption, reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, file_path, caption, reply_to, file_name)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, audio_path, caption, reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, video_path, caption, reply_to)

    def format_message(self, content: str) -> str:
        """Rocket.Chat uses standard Markdown — pass through mostly.

        Strip image markdown to plain links since file uploads are
        handled separately via send_image / send_image_file.
        Ensure bold/italic markers are balanced to prevent rendering artifacts.
        """
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)
        content = _convert_double_bold(content)
        content = _balance_marker(content, "*")
        content = _balance_marker(content, "__")
        # Replace ASCII tilde with Unicode TILDE OPERATOR (U+223C) so that RC
        # does not interpret pairs of tildes as strikethrough delimiters.
        # The two characters are visually identical in every common font.
        content = content.replace("~", "\u223c")
        return content

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    async def _send_url_as_file(
        self,
        room_id: str,
        url: str,
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> SendResult:
        import aiohttp

        file_data = None
        ct = "application/octet-stream"
        fname = url.rsplit("/", 1)[-1].split("?")[0] or "image.png"

        for attempt in range(3):
            try:
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if resp.status >= 400:
                        return await self.send(
                            room_id, f"{caption or ''}\n{url}".strip(), reply_to
                        )
                    file_data = await resp.read()
                    ct = resp.content_type or ct
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                logger.warning("Rocket.Chat: download failed for %s: %s", url, exc)
                return await self.send(
                    room_id, f"{caption or ''}\n{url}".strip(), reply_to
                )

        if not file_data:
            return await self.send(
                room_id, f"{caption or ''}\n{url}".strip(), reply_to
            )

        msg_id = await self._upload_file(
            room_id, file_data, fname, ct, description=caption or ""
        )
        if not msg_id:
            return await self.send(
                room_id, f"{caption or ''}\n{url}".strip(), reply_to
            )
        return SendResult(success=True, message_id=msg_id)

    async def _send_local_file(
        self,
        room_id: str,
        file_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
    ) -> SendResult:
        import mimetypes

        p = Path(file_path)
        if not p.exists():
            return await self.send(
                room_id,
                f"{caption or ''}\n(file not found: {file_path})".strip(),
                reply_to,
            )

        fname = file_name or p.name
        ct = mimetypes.guess_type(fname)[0] or "application/octet-stream"

        msg_id = await self._upload_file(
            room_id, p.read_bytes(), fname, ct, description=caption or ""
        )
        if not msg_id:
            return SendResult(success=False, error="File upload failed")
        return SendResult(success=True, message_id=msg_id)

    # ------------------------------------------------------------------
    # WebSocket (DDP)
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to the DDP WebSocket and listen, reconnecting on failure."""
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                logger.warning(
                    "Rocket.Chat WS error: %s — reconnecting in %.0fs", exc, delay
                )

            if self._closing:
                return

            import random
            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single DDP WebSocket session: connect, auth via resume, subscribe, loop."""
        import aiohttp

        ws_url = re.sub(r"^http", "ws", self._base_url) + "/websocket"
        logger.info("Rocket.Chat: connecting WS to %s", ws_url)

        self._ws = await self._session.ws_connect(ws_url, heartbeat=30.0)

        # DDP connect handshake.
        await self._ws.send_json({
            "msg": "connect",
            "version": "1",
            "support": ["1"],
        })
        connected = await self._ddp_wait_for({"msg": "connected"}, timeout=15)
        if not connected:
            raise RuntimeError("DDP connect ack not received")

        # Authenticate using the session authToken obtained during login.
        # `resume` works for both PAT and session tokens.
        self._ddp_seq += 1
        login_seq = str(self._ddp_seq)
        await self._ws.send_json({
            "msg": "method",
            "method": "login",
            "id": login_seq,
            "params": [{"resume": self._auth_token}],
        })
        login_result = await self._ddp_wait_for(
            {"msg": "result", "id": login_seq}, timeout=15
        )
        if not login_result or login_result.get("error"):
            err = (login_result or {}).get("error", {})
            raise RuntimeError(
                f"DDP login failed: {err.get('message', 'unknown error')}"
            )

        logger.info("Rocket.Chat: DDP authenticated, subscribing to messages")

        # Subscribe to all messages the bot user can receive.
        self._ddp_seq += 1
        await self._ws.send_json({
            "msg": "sub",
            "id": _SUBSCRIPTION_ID,
            "name": "stream-room-messages",
            "params": ["__my_messages__", False],
        })

        # Main event loop.
        async for raw_msg in self._ws:
            if self._closing:
                return

            if raw_msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ddp_event(event)
            elif raw_msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                logger.info("Rocket.Chat: WebSocket closed (%s)", raw_msg.type)
                break

    async def _ddp_wait_for(
        self, match: Dict[str, Any], timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Read DDP frames until one matches all keys in `match`, or timeout."""
        import aiohttp

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                raw = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            if raw.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                continue
            try:
                frame = json.loads(raw.data)
            except (json.JSONDecodeError, TypeError):
                continue

            if frame.get("msg") == "ping":
                await self._ws.send_json({"msg": "pong"})
                continue

            if all(frame.get(k) == v for k, v in match.items()):
                return frame

    async def _handle_ddp_event(self, event: Dict[str, Any]) -> None:
        """Process a single DDP frame."""
        msg = event.get("msg")

        if msg == "ping":
            await self._ws.send_json({"msg": "pong"})
            return

        if msg != "changed" or event.get("collection") != "stream-room-messages":
            return

        fields = event.get("fields", {})
        args = fields.get("args", [])
        if not args or not isinstance(args[0], dict):
            return

        message = args[0]

        sender = message.get("u", {})
        if sender.get("_id") == self._user_id:
            return
        if message.get("t"):  # system message (user joined, etc.)
            return

        msg_id = message.get("_id", "")
        self._prune_seen()
        if msg_id in self._seen_msgs:
            return
        self._seen_msgs[msg_id] = time.time()

        room_id = message.get("rid", "")
        message_text = message.get("msg", "")

        # Determine room type via API (cached after first lookup).
        raw_room_type = await self._get_room_type(room_id)
        chat_type = _ROOM_TYPE_MAP.get(raw_room_type, "channel")

        # Mention-gating for non-DM rooms.
        if raw_room_type != "d":
            require_mention = os.getenv(
                "ROCKETCHAT_REQUIRE_MENTION", "true"
            ).lower() not in ("false", "0", "no")

            free_channels = {
                ch.strip()
                for ch in os.getenv("ROCKETCHAT_FREE_RESPONSE_CHANNELS", "").split(",")
                if ch.strip()
            }

            mention_patterns = [
                f"@{self._bot_username}",
                f"@{self._user_id}",
            ]
            has_mention = any(
                p.lower() in message_text.lower() for p in mention_patterns
            )

            if require_mention and room_id not in free_channels and not has_mention:
                logger.debug(
                    "Rocket.Chat: skipping non-DM message without @mention (room=%s)",
                    room_id,
                )
                return

            if has_mention:
                for p in mention_patterns:
                    message_text = re.sub(
                        re.escape(p), "", message_text, flags=re.IGNORECASE
                    ).strip()

        sender_id = sender.get("_id", "")
        sender_name = sender.get("username", "") or sender_id
        thread_id = message.get("tmid") or None

        msg_type_enum = MessageType.COMMAND if message_text.startswith("/") else MessageType.TEXT

        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = message.get("file")
        if file_info:
            await self._cache_file_attachment(file_info, media_urls, media_types)
        for att in message.get("attachments") or []:
            image_url = att.get("image_url") or att.get("title_link") or att.get("thumb_url")
            if image_url:
                full_url = (
                    image_url if image_url.startswith("http")
                    else f"{self._base_url}{image_url}"
                )
                await self._cache_attachment_url(full_url, att, media_urls, media_types)

        if media_urls:
            # Buffer so a follow-up text message can reference this file.
            for path, mime in zip(media_urls, media_types):
                self._store_attachment(room_id, path, mime)
            # File-only upload — wait for the accompanying text before dispatching.
            if not message_text:
                return
        else:
            recent_urls, recent_types = self._pop_recent_attachments(room_id)
            media_urls.extend(recent_urls)
            media_types.extend(recent_types)

        source = self.build_source(
            chat_id=room_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
        )

        await self.handle_message(MessageEvent(
            text=message_text,
            message_type=msg_type_enum,
            source=source,
            raw_message=message,
            message_id=msg_id,
            media_urls=media_urls or None,
            media_types=media_types or None,
        ))

    async def _cache_file_attachment(
        self,
        file_info: Dict[str, Any],
        media_urls: List[str],
        media_types: List[str],
    ) -> None:
        """Download and cache a Rocket.Chat file attachment."""
        import aiohttp

        file_id = file_info.get("_id", "")
        if not file_id:
            return

        mime = file_info.get("type", "application/octet-stream")
        fname = file_info.get("name", f"file_{file_id}")
        ext = Path(fname).suffix or ""
        dl_url = f"{self._base_url}/file-upload/{file_id}/{fname}"

        try:
            async with self._session.get(
                dl_url,
                headers={"X-Auth-Token": self._auth_token, "X-User-Id": self._user_id},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    logger.warning(
                        "Rocket.Chat: failed to download file %s: HTTP %s", file_id, resp.status
                    )
                    return
                data = await resp.read()

            from gateway.platforms.base import (
                cache_image_from_bytes,
                cache_audio_from_bytes,
                cache_document_from_bytes,
            )
            if mime.startswith("image/"):
                local_path = cache_image_from_bytes(data, ext or ".png")
            elif mime.startswith("audio/"):
                local_path = cache_audio_from_bytes(data, ext or ".ogg")
            else:
                local_path = cache_document_from_bytes(data, fname)

            media_urls.append(local_path)
            media_types.append(mime)
        except Exception as exc:
            logger.warning("Rocket.Chat: error caching file %s: %s", file_id, exc)

    async def _cache_attachment_url(
        self,
        url: str,
        att: Dict[str, Any],
        media_urls: List[str],
        media_types: List[str],
    ) -> None:
        """Download and cache a URL-based attachment."""
        import aiohttp

        mime = att.get("image_type") or "image/png"
        ext = ("." + mime.split("/")[-1]) if "/" in mime else ".png"

        try:
            async with self._session.get(
                url,
                headers={"X-Auth-Token": self._auth_token, "X-User-Id": self._user_id},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    return
                data = await resp.read()
                mime = resp.content_type or mime

            from gateway.platforms.base import cache_image_from_bytes
            media_urls.append(cache_image_from_bytes(data, ext))
            media_types.append(mime)
        except Exception as exc:
            logger.warning("Rocket.Chat: error caching attachment %s: %s", url, exc)

    async def _get_room_type(self, room_id: str) -> str:
        """Return 'd' for DM rooms, 'c' for everything else.

        Uses rooms.info which works for all room types in a single request.
        Results are cached so that each room is looked up only once via the
        REST API.  Fallback is 'c' (channel) to preserve the safe default of
        requiring @mention.
        """
        if room_id in self._room_type_cache:
            return self._room_type_cache[room_id]

        data = await self._api_get("rooms.info", {"roomId": room_id})
        raw = data.get("room", {}).get("t", "c")
        result = "d" if raw == "d" else "c"
        self._room_type_cache[room_id] = result
        return result

    def _store_attachment(self, room_id: str, local_path: str, mime: str) -> None:
        """Save a file attachment to the per-room buffer."""
        bucket = self._recent_attachments.setdefault(room_id, [])
        bucket.append((time.time(), local_path, mime))

    def _pop_recent_attachments(self, room_id: str) -> tuple:
        """Return (urls, types) for unexpired attachments in room, then clear them."""
        now = time.time()
        bucket = self._recent_attachments.pop(room_id, [])
        valid = [(p, m) for ts, p, m in bucket if now - ts < self._ATTACHMENT_TTL]
        if not valid:
            return [], []
        paths, mimes = zip(*valid)
        return list(paths), list(mimes)

    def _prune_seen(self) -> None:
        if len(self._seen_msgs) < self._SEEN_MAX:
            return
        now = time.time()
        self._seen_msgs = {
            mid: ts for mid, ts in self._seen_msgs.items() if now - ts < self._SEEN_TTL
        }
