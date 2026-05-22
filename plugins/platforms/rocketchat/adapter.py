"""Rocket.Chat gateway adapter plugin for Hermes Agent.

Connects to a self-hosted Rocket.Chat instance via its REST API (v1) for
outbound traffic and the Realtime (DDP) WebSocket for inbound messages.
No external Rocket.Chat library required — uses aiohttp, which is already
a Hermes dependency.

Design notes:
    Rocket.Chat's docs recommend REST for writes (chat.postMessage,
    chat.update, rooms.media) and DDP for reads (stream-room-messages).
    The bot subscribes to the ``__my_messages__`` virtual room id, which
    covers every channel/DM/group the bot is a member of — no per-room
    enumeration required.

    Personal Access Tokens double as DDP resume tokens, so a single
    ``ROCKETCHAT_TOKEN`` + ``ROCKETCHAT_USER_ID`` pair authenticates both
    surfaces. Generate the PAT with "Ignore Two Factor" checked to keep
    unattended REST calls working on 2FA-enabled workspaces.

Environment variables:
    ROCKETCHAT_URL              Server URL (e.g. https://rc.example.com)
    ROCKETCHAT_TOKEN            Personal Access Token (used as auth token)
    ROCKETCHAT_USER_ID          Bot user's _id (shown alongside the PAT)
    ROCKETCHAT_ALLOWED_USERS    Comma-separated user IDs
    ROCKETCHAT_ALLOW_ALL_USERS  Allow all users (dev only)
    ROCKETCHAT_HOME_CHANNEL     Room ID for cron/notification delivery
    ROCKETCHAT_REQUIRE_MENTION  Require @mention in channels (default: true)
    ROCKETCHAT_FREE_RESPONSE_CHANNELS  Rooms exempt from mention requirement
    ROCKETCHAT_REPLY_MODE       Reply mode: 'thread' or 'off' (default: off)
    ROCKETCHAT_REACTIONS        Add 👀/✅/❌ reactions to messages (default: true)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

# Rocket.Chat's default Message_MaxAllowedSize is 5000; admins can raise it
# but the safe default for multi-line messages is 5000.
MAX_MESSAGE_LENGTH = 5000

# Room type codes returned by the Rocket.Chat API.
#   d = direct message (1:1)
#   c = public channel
#   p = private group (private channel)
#   l = livechat / omnichannel
_ROOM_TYPE_MAP = {
    "d": "dm",
    "c": "channel",
    "p": "group",
    "l": "group",
}

# Reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2

# DDP protocol version. Rocket.Chat supports "1" across 7.x/8.x.
_DDP_PROTOCOL_VERSION = "1"

# ---------------------------------------------------------------------------
# Plugin-level helpers
# ---------------------------------------------------------------------------


def check_requirements() -> bool:
    """Return True if the Rocket.Chat adapter can be used."""
    token = os.getenv("ROCKETCHAT_TOKEN", "")
    url = os.getenv("ROCKETCHAT_URL", "")
    user_id = os.getenv("ROCKETCHAT_USER_ID", "")
    if not token:
        return False
    if not url:
        return False
    if not user_id:
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    url = os.getenv("ROCKETCHAT_URL") or extra.get("url", "")
    token = os.getenv("ROCKETCHAT_TOKEN") or getattr(config, "token", "") or extra.get("token", "")
    user_id = os.getenv("ROCKETCHAT_USER_ID") or extra.get("user_id", "")
    return bool(url and token and user_id)


def is_connected(config) -> bool:
    """Check whether Rocket.Chat is configured (env or config.yaml)."""
    return validate_config(config)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` reflects env-only configuration
    without instantiating the Rocket.Chat client.

    Returns ``None`` when Rocket.Chat isn't minimally configured.
    """
    url = os.getenv("ROCKETCHAT_URL", "").strip()
    token = os.getenv("ROCKETCHAT_TOKEN", "").strip()
    user_id = os.getenv("ROCKETCHAT_USER_ID", "").strip()
    if not (url and token and user_id):
        return None

    seed: dict = {
        "url": url,
        "token": token,
        "user_id": user_id,
    }

    reply_mode = os.getenv("ROCKETCHAT_REPLY_MODE", "").strip()
    if reply_mode:
        seed["reply_mode"] = reply_mode

    home = os.getenv("ROCKETCHAT_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("ROCKETCHAT_HOME_CHANNEL_NAME", home),
        }

    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Open an ephemeral REST-only connection to send a message for cron delivery.

    Uses ``chat.postMessage`` via aiohttp — no DDP WebSocket (too heavy for
    one-shot sends).

    ``thread_id`` and ``media_files`` are accepted for signature parity but
    ``media_files`` is not implemented yet for the standalone path.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    url = os.getenv("ROCKETCHAT_URL") or extra.get("url", "")
    token = os.getenv("ROCKETCHAT_TOKEN") or getattr(pconfig, "token", "") or extra.get("token", "")
    user_id = os.getenv("ROCKETCHAT_USER_ID") or extra.get("user_id", "")
    if not url or not token or not user_id:
        return {"error": "Rocket.Chat standalone send: ROCKETCHAT_URL, TOKEN, and USER_ID must be configured"}

    headers = {
        "X-Auth-Token": token,
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "roomId": chat_id,
        "text": message,
    }
    if thread_id:
        payload["tmid"] = thread_id

    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(
                f"{url.rstrip('/')}/api/v1/chat.postMessage",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return {"error": f"Rocket.Chat standalone send HTTP {resp.status}: {body[:200]}"}
                data = await resp.json()
                msg = data.get("message") or {}
                return {"success": True, "message_id": msg.get("_id", "")}
    except Exception as exc:
        logger.debug("Rocket.Chat standalone send raised", exc_info=True)
        return {"error": f"Rocket.Chat standalone send failed: {exc}"}


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class RocketchatAdapter(BasePlatformAdapter):
    """Gateway adapter for Rocket.Chat (self-hosted)."""

    def __init__(self, config, **kwargs):
        platform = Platform("rocketchat")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self._base_url: str = (
            extra.get("url", "")
            or os.getenv("ROCKETCHAT_URL", "")
        ).rstrip("/")
        self._token: str = getattr(config, "token", None) or extra.get("token", "") or os.getenv("ROCKETCHAT_TOKEN", "")
        self._bot_user_id: str = (
            extra.get("user_id", "")
            or os.getenv("ROCKETCHAT_USER_ID", "")
        )

        # Filled in by connect() once we look up the bot's username.
        self._bot_username: str = ""

        # aiohttp session + websocket handle
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None       # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closing = False

        # DDP bookkeeping
        self._ddp_next_id = 1
        self._ddp_subs: Dict[str, bool] = {}  # sub-id -> ready

        # Room type cache (roomId -> "dm"/"group"/"channel").
        self._room_type_cache: Dict[str, str] = {}

        # Reply mode: "thread" to nest replies, "off" for flat messages.
        self._reply_mode: str = (
            extra.get("reply_mode", "")
            or os.getenv("ROCKETCHAT_REPLY_MODE", "off")
        ).lower()

        # Dedup cache.
        self._dedup = MessageDeduplicator()

        # Title→topic sync state: rate-limit and last-known topic per room.
        self._last_topic_sync: Dict[str, float] = {}  # room_id → timestamp
        self._last_topic: Dict[str, str] = {}  # room_id → last known topic

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Auth-Token": self._token,
            "X-User-Id": self._bot_user_id,
            "Content-Type": "application/json",
        }

    async def _api_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET /api/v1/{path}."""
        import aiohttp
        url = f"{self._base_url}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.get(
                url, headers=self._headers(), params=params,
                timeout=aiohttp.ClientTimeout(total=30),
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
                timeout=aiohttp.ClientTimeout(total=30),
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
        content_type: str,
        caption: Optional[str] = None,
        tmid: Optional[str] = None,
    ) -> Optional[str]:
        """Upload a file via the two-step rooms.media flow.

        Step 1 uploads the bytes; step 2 confirms and creates the message.
        Returns the message _id on success, None on failure.
        """
        import aiohttp

        # Step 1: upload the file bytes.
        step1_url = f"{self._base_url}/api/v1/rooms.media/{room_id}"
        form = aiohttp.FormData()
        form.add_field(
            "file",
            file_data,
            filename=filename,
            content_type=content_type,
        )
        headers = {
            "X-Auth-Token": self._token,
            "X-User-Id": self._bot_user_id,
        }
        try:
            async with self._session.post(
                step1_url, headers=headers, data=form,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("RC rooms.media → %s: %s", resp.status, body[:200])
                    return None
                step1 = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("RC rooms.media network error: %s", exc)
            return None

        file_id = (step1.get("file") or {}).get("_id")
        if not file_id:
            logger.error("RC rooms.media returned no file id: %s", step1)
            return None

        # Step 2: confirm — this creates the message.
        step2_path = f"rooms.mediaConfirm/{room_id}/{file_id}"
        payload: Dict[str, Any] = {}
        if caption:
            payload["msg"] = caption
        if tmid and self._reply_mode == "thread":
            payload["tmid"] = tmid
        step2 = await self._api_post(step2_path, payload)
        msg = step2.get("message") or {}
        return msg.get("_id")

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Rocket.Chat and start the DDP listener."""
        import aiohttp

        if not self._base_url or not self._token or not self._bot_user_id:
            logger.error("Rocket.Chat: URL, token, or user id not configured")
            return False

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._closing = False

        # Verify credentials and fetch bot identity.
        me = await self._api_get("me")
        if not me or not me.get("success"):
            logger.error(
                "Rocket.Chat: failed to authenticate — check "
                "ROCKETCHAT_TOKEN, ROCKETCHAT_USER_ID, ROCKETCHAT_URL"
            )
            await self._session.close()
            return False

        if me.get("_id") and me["_id"] != self._bot_user_id:
            logger.warning(
                "Rocket.Chat: ROCKETCHAT_USER_ID (%s) doesn't match /me (%s) — using /me",
                self._bot_user_id, me["_id"],
            )
            self._bot_user_id = me["_id"]
        self._bot_username = me.get("username", "")
        logger.info(
            "Rocket.Chat: authenticated as @%s (%s) on %s",
            self._bot_username,
            self._bot_user_id,
            self._base_url,
        )

        # Start DDP WebSocket in background.
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Rocket.Chat."""
        self._closing = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()

        self._mark_disconnected()
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
            payload: Dict[str, Any] = {
                "roomId": chat_id,
                "text": chunk,
            }
            if reply_to and self._reply_mode == "thread":
                payload["tmid"] = reply_to

            data = await self._api_post("chat.postMessage", payload)
            if not data or not data.get("success"):
                return SendResult(success=False, error="Failed to post message")
            msg = data.get("message") or {}
            last_id = msg.get("_id") or last_id
            if msg:
                logger.info("Rocket.Chat: send() POST chat.postMessage → rid=%s tmid=%s msg_id=%s",
                            msg.get("rid"), msg.get("tmid"), msg.get("_id"))

        # After sending, sync session title → RC topic for DMs.
        # This fires on every outgoing message but is rate-limited and
        # short-circuits when the title hasn't changed.
        try:
            await self._sync_title_to_rc_topic(chat_id)
        except Exception:
            logger.debug("Title sync failed in send()", exc_info=True)

        return SendResult(success=True, message_id=last_id)

    @staticmethod
    def _set_topic_endpoint(chat_type: str) -> str:
        """Return the RC endpoint key for setting a room topic based on room type."""
        return {
            "dm": "dm.setTopic",
            "channel": "channels.setTopic",
            "group": "groups.setTopic",
        }.get(chat_type, "channels.setTopic")

    async def _sync_title_to_rc_topic(self, chat_id: str) -> None:
        """Sync Hermes session title to RC room topic for DMs/groups/channels.

        Called after every outgoing send().  Checks the current session title
        and updates the RC topic if they differ.  This covers:
          - Auto-title (first-reply title generated by Hermes)
          - /title command (already handled in _handle_message, but also
            catches manual session_db / CLI rename changes that happened
            between messages)
        Rate-limited to at most once every 5 seconds per room.
        """
        import time
        now = time.time()
        if chat_id in self._last_topic_sync and now - self._last_topic_sync[chat_id] < 5:
            return
        self._last_topic_sync[chat_id] = now

        # Only for DM/group/channel rooms where topic setting makes sense
        chat_type = self._room_type_cache.get(chat_id)
        if not chat_type:
            try:
                chat_type = await self._resolve_room_type(chat_id)
            except Exception:
                return
        if chat_type not in ("dm", "group", "channel"):
            return

        # Build a SessionSource and look up the session
        from gateway.config import Platform
        from gateway.session import SessionSource

        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return

        try:
            source = SessionSource(
                platform=Platform("rocketchat"),
                chat_id=chat_id,
                chat_type="dm",
            )
            entry = session_store.get_or_create_session(source)
        except Exception as exc:
            logger.debug("Title sync: session lookup failed: %s", exc)
            return

        # Get the session title from the SQLite DB
        db = getattr(session_store, "_db", None)
        if not db:
            return
        try:
            title = db.get_session_title(entry.session_id)
        except Exception as exc:
            logger.debug("Title sync: get_title failed: %s", exc)
            return
        if not title:
            return

        # Get the current RC topic
        data = await self._api_get("rooms.info", params={"roomId": chat_id})
        room = (data or {}).get("room") or {}
        current_topic = (room.get("topic") or "").strip()

        # Only call the API if topics differ
        if title != current_topic:
            endpoint = self._set_topic_endpoint(chat_type)
            try:
                resp = await self._api_post(endpoint, {
                    "roomId": chat_id,
                    "topic": title,
                })
                if resp and resp.get("success"):
                    self._last_topic[chat_id] = title
                    logger.info(
                        "Rocket.Chat: synced session title '%s' to %s topic (room=%s)",
                        title, chat_type, chat_id,
                    )
            except Exception as exc:
                logger.debug("Title sync: %s failed: %s", endpoint, exc)
        else:
            # Already in sync — just update the cache
            self._last_topic[chat_id] = current_topic

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return room name and type.

        Rocket.Chat exposes one unified ``rooms.info`` endpoint that works
        for channels, private groups, and DMs.
        """
        data = await self._api_get("rooms.info", params={"roomId": chat_id})
        room = (data or {}).get("room") or {}
        if not room:
            return {"name": chat_id, "type": "channel"}

        raw_type = room.get("t", "c")
        chat_type = _ROOM_TYPE_MAP.get(raw_type, "channel")
        self._room_type_cache[chat_id] = chat_type

        if chat_type == "dm":
            others = [
                u for u in room.get("usernames", [])
                if u and u != self._bot_username
            ]
            name = others[0] if others else chat_id
        else:
            name = room.get("fname") or room.get("name") or chat_id

        return {"name": name, "type": chat_type, "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify that the bot is typing.

        Rocket.Chat 6.x+ replaced the legacy ``/typing`` stream with
        ``/user-activity``, and 8.x expects the activity string ``"user-typing"``.
        """
        if not self._ws or self._ws.closed:
            return
        if not self._bot_username:
            return
        await self._ddp_method(
            "stream-notify-room",
            [f"{chat_id}/user-activity", self._bot_username, ["user-typing"], {}],
        )

    async def stop_typing(self, chat_id: str) -> None:
        """Clear the typing indicator (empty user-activity list)."""
        if not self._ws or self._ws.closed:
            return
        if not self._bot_username:
            return
        await self._ddp_method(
            "stream-notify-room",
            [f"{chat_id}/user-activity", self._bot_username, [], {}],
        )

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False
    ) -> SendResult:
        """Edit an existing message via chat.update."""
        formatted = self.format_message(content)
        data = await self._api_post(
            "chat.update",
            {"roomId": chat_id, "msgId": message_id, "text": formatted},
        )
        if not data or not data.get("success"):
            return SendResult(success=False, error="Failed to edit message")
        msg = data.get("message") or {}
        return SendResult(success=True, message_id=msg.get("_id", message_id))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download an image and upload it as a file attachment."""
        return await self._send_url_as_file(
            chat_id, image_url, caption, reply_to, "image"
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local image file."""
        return await self._send_local_file(
            chat_id, image_path, caption, reply_to
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file as a document."""
        return await self._send_local_file(
            chat_id, file_path, caption, reply_to, file_name
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload an audio file."""
        return await self._send_local_file(
            chat_id, audio_path, caption, reply_to
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a video file."""
        return await self._send_local_file(
            chat_id, video_path, caption, reply_to
        )

    def format_message(self, content: str) -> str:
        """Rocket.Chat renders Markdown natively and previews plain image
        URLs — strip image markdown to match Mattermost's behavior.

        Also strip Hermes-internal delivery directives (MEDIA:,
        [[audio_as_voice]], [[image]], [[file]]) — the gateway already
        delivers media via send_voice/send_image/send_document methods,
        and these tokens must not reach the Rocket.Chat API as text.
        """
        # Strip image markdown: ![alt](url) → url
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)
        # Strip entire lines and trailing content that start with media tags
        content = re.sub(
            r"(?m)^\s*(?:\[\[audio_as_voice\]\]|\[\[image\]\]|\[\[file\]\]|MEDIA)\s*:?.*(?:\n|$)",
            "",
            content,
        )
        # Also strip orphan MEDIA: references not at line start
        content = re.sub(r"\s*MEDIA:\S+\s*", " ", content)
        return content.strip()

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    async def _send_url_as_file(
        self,
        chat_id: str,
        url: str,
        caption: Optional[str],
        reply_to: Optional[str],
        kind: str = "file",
    ) -> SendResult:
        """Download a URL and upload it as a file attachment."""
        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            logger.warning("Rocket.Chat: blocked unsafe URL (SSRF protection)")
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        import aiohttp

        file_data = None
        ct = "application/octet-stream"
        fname = url.rsplit("/", 1)[-1].split("?")[0] or f"{kind}.png"

        for attempt in range(3):
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        if attempt < 2:
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    if resp.status >= 400:
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)
                    file_data = await resp.read()
                    ct = resp.content_type or "application/octet-stream"
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        if file_data is None:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        msg_id = await self._upload_file(
            chat_id, file_data, fname, ct, caption, reply_to,
        )
        if not msg_id:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)
        return SendResult(success=True, message_id=msg_id)

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Upload a local file via the two-step rooms.media flow."""
        import mimetypes

        p = Path(file_path)
        if not p.exists():
            return await self.send(
                chat_id, f"{caption or ''}\n(file not found: {file_path})", reply_to
            )

        fname = file_name or p.name
        ct = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        file_data = p.read_bytes()

        msg_id = await self._upload_file(
            chat_id, file_data, fname, ct, caption, reply_to,
        )
        if not msg_id:
            return SendResult(success=False, error="File upload failed")
        return SendResult(success=True, message_id=msg_id)

    # ------------------------------------------------------------------
    # DDP / WebSocket
    # ------------------------------------------------------------------

    async def _ddp_send(self, payload: Dict[str, Any]) -> None:
        """Send a DDP frame if the socket is open."""
        if not self._ws or self._ws.closed:
            return
        await self._ws.send_json(payload)

    async def _ddp_method(self, method: str, params: List[Any]) -> str:
        """Invoke a DDP method (fire-and-forget). Returns the method id."""
        call_id = str(self._ddp_next_id)
        self._ddp_next_id += 1
        await self._ddp_send({
            "msg": "method",
            "method": method,
            "id": call_id,
            "params": params,
        })
        return call_id

    async def _ddp_sub(self, name: str, params: List[Any]) -> str:
        """Subscribe to a DDP publication. Returns the sub id."""
        sub_id = str(uuid.uuid4())
        self._ddp_subs[sub_id] = False
        await self._ddp_send({
            "msg": "sub",
            "id": sub_id,
            "name": name,
            "params": params,
        })
        return sub_id

    async def _ws_loop(self) -> None:
        """Connect to the DDP socket and listen for events, reconnecting on failure."""
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
                import aiohttp
                if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in (401, 403):
                    logger.error("Rocket.Chat WS auth failed (HTTP %d) — stopping reconnect", exc.status)
                    return
                err_str = str(exc).lower()
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    logger.error("Rocket.Chat WS permanent error: %s — stopping reconnect", exc)
                    return
                logger.warning("Rocket.Chat WS error: %s — reconnecting in %.0fs", exc, delay)

            if self._closing:
                return

            import random
            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single DDP WebSocket session: connect, login, subscribe, listen."""
        ws_url = re.sub(r"^http", "ws", self._base_url) + "/websocket"
        logger.info("Rocket.Chat: DDP connecting to %s", ws_url)

        self._ws = await self._session.ws_connect(ws_url, heartbeat=None)
        self._ddp_subs.clear()

        await self._ddp_send({
            "msg": "connect",
            "version": _DDP_PROTOCOL_VERSION,
            "support": [_DDP_PROTOCOL_VERSION],
        })

        await self._ddp_method("login", [{"resume": self._token}])

        await self._ddp_sub("stream-room-messages", ["__my_messages__", False])
        logger.info("Rocket.Chat: DDP logged in and subscribed")

        async for raw_msg in self._ws:
            if self._closing:
                return

            if raw_msg.type in (raw_msg.type.TEXT, raw_msg.type.BINARY):
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ddp_frame(event)
            elif raw_msg.type in (
                raw_msg.type.ERROR, raw_msg.type.CLOSE,
                raw_msg.type.CLOSING, raw_msg.type.CLOSED,
            ):
                logger.info("Rocket.Chat: DDP WebSocket closed (%s)", raw_msg.type)
                break

    async def _handle_ddp_frame(self, event: Dict[str, Any]) -> None:
        """Dispatch a single DDP frame."""
        kind = event.get("msg")
        if kind == "ping":
            pong: Dict[str, Any] = {"msg": "pong"}
            if "id" in event:
                pong["id"] = event["id"]
            await self._ddp_send(pong)
            return

        if kind == "ready":
            for sub_id in event.get("subs", []):
                self._ddp_subs[sub_id] = True
            return

        if kind == "nosub":
            sub_id = event.get("id", "")
            err = event.get("error") or {}
            self._ddp_subs.pop(sub_id, None)
            if err:
                logger.warning("Rocket.Chat: sub %s rejected: %s", sub_id, err)
            return

        if kind == "changed":
            collection = event.get("collection")
            if collection != "stream-room-messages":
                return
            fields = event.get("fields") or {}
            args = fields.get("args") or []
            if not args:
                return
            await self._handle_message(args[0])
            return

    async def _handle_message(self, post: Dict[str, Any]) -> None:
        """Process an incoming Rocket.Chat message."""
        sender = post.get("u") or {}
        sender_id = sender.get("_id", "")
        sender_name = sender.get("username", "") or sender_id

        # Ignore own messages.
        if sender_id == self._bot_user_id:
            return

        post_id = post.get("_id", "")
        if self._dedup.is_duplicate(post_id):
            return

        room_id = post.get("rid", "")
        if not room_id:
            return

        # Look up room type lazily; cache forever.
        chat_type = self._room_type_cache.get(room_id)
        if chat_type is None:
            chat_type = await self._resolve_room_type(room_id)
            self._room_type_cache[room_id] = chat_type

        # Handle system messages: skip all except topic changes in DMs.
        t_type = post.get("t")
        if t_type:
            if t_type == "room_changed_topic" and chat_type == "dm":
                topic_text = (post.get("msg") or "").strip()
                if topic_text:
                    # Update topic cache immediately (avoids extra API call
                    # in _sync_title_to_rc_topic on the next send())
                    self._last_topic[room_id] = topic_text

                    source = self.build_source(
                        chat_id=room_id,
                        chat_type=chat_type,
                        user_id=sender_id,
                        user_name=sender_name,
                        thread_id=None,
                    )
                    from gateway.platforms.base import resolve_channel_prompt
                    channel_prompt = resolve_channel_prompt(
                        self.config.extra, room_id, None,
                    )
                    cmd_msg = MessageEvent(
                        text=f"/title {topic_text}",
                        message_type=MessageType.COMMAND,
                        source=source,
                        raw_message=post,
                        message_id=post_id,
                        channel_prompt=channel_prompt,
                    )
                    await self.handle_message(cmd_msg)
            return  # All other system messages: skip

        message_text = post.get("msg", "") or ""

        # Mention gating for non-DM rooms.
        if chat_type != "dm":
            require_mention = os.getenv(
                "ROCKETCHAT_REQUIRE_MENTION", "true"
            ).lower() not in ("false", "0", "no")

            free_channels_raw = os.getenv("ROCKETCHAT_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            is_free_channel = room_id in free_channels

            mentions = post.get("mentions") or []
            mention_ids = {m.get("_id") for m in mentions if isinstance(m, dict)}
            mention_names = {m.get("username") for m in mentions if isinstance(m, dict)}
            has_mention = (
                self._bot_user_id in mention_ids
                or self._bot_username in mention_names
                or "all" in mention_ids or "here" in mention_ids
            )
            if not has_mention and self._bot_username:
                pattern = re.compile(
                    rf"(?:^|\W)@{re.escape(self._bot_username)}(?:\W|$)",
                    re.IGNORECASE,
                )
                has_mention = bool(pattern.search(message_text))

            if require_mention and not is_free_channel and not has_mention:
                return

            if has_mention and self._bot_username:
                message_text = re.sub(
                    rf"(^|\W)@{re.escape(self._bot_username)}(\W|$)",
                    r"\1\2",
                    message_text,
                    flags=re.IGNORECASE,
                ).strip()

        

        thread_id = post.get("tmid") or None

        # Route RC-native slash commands back to Rocket.Chat.
        # Check both the raw post text AND the stripped message_text.
        # In DMs the @mention is never stripped, so raw_msg will contain
        # e.g. "@lobster.bot /dashboard". The dual-text loop handles that:
        # raw_msg has the mention prefix, message_text has it stripped — one
        # of them will have "/" at position 0 for a real slash command.
        #
        # IMPORTANT: we ONLY match "/" at position 0, NOT mid-sentence.
        # A message like "ich find /status doof" is NOT a slash command —
        # it's just text that happens to contain "/status".
        #
        # For known Hermes gateway commands (like /new, /approve, /dashboard,
        # /workspace, etc.) we skip the RC commands.run call entirely —
        # RC doesn't know them and would return 400.  This avoids spurious
        # "command does not exist" error logs and the unnecessary API round-trip.
        # Unknown/RC-native commands still get routed to RC first.
        raw_msg = post.get("msg", "") or ""
        _found_slash_cmd = False
        cmd_full = ""
        for candidate_text in (raw_msg, message_text):
            slash_pos = candidate_text.find("/")
            if slash_pos == 0:
                cmd_raw = candidate_text[slash_pos:]
                cmd_token = cmd_raw.split(None, 1)[0]
                cmd_params = cmd_raw[len(cmd_token):].strip()
                
                _found_slash_cmd = True
                cmd_full = cmd_raw
                
                # Skip RC routing for known Hermes gateway commands.
                _is_hermes_cmd = False
                try:
                    from hermes_cli.commands import is_gateway_known_command
                    # Strip leading "/" before checking — is_gateway_known_command
                    # expects the bare name (e.g. "new", not "/new").
                    bare_cmd = cmd_token.lstrip("/").lower()
                    _is_hermes_cmd = is_gateway_known_command(bare_cmd)
                except Exception:
                    pass  # defensive: if import fails, fall through to RC route
                
                if not _is_hermes_cmd:
                    rc_payload: Dict[str, Any] = {
                        "command": cmd_token,
                        "roomId": room_id,
                        "params": cmd_params,
                    }
                    if thread_id:
                        rc_payload["tmid"] = thread_id
                    data = await self._api_post("commands.run", rc_payload)
                    if data and data.get("success"):
                        logger.info(
                            "Rocket.Chat: routed command %s to RC (room=%s)",
                            cmd_token, room_id,
                        )
                        return  # RC handled it
                break  # tried one text, fall through to agent

        # If we found and tried to route a / command, replace message_text
        # with the extracted command so downstream (coerce_plaintext_gateway_command,
        # msg_type detection, etc.) sees the cleaned command text.
        if _found_slash_cmd:
            message_text = cmd_full

        # Bidirectional title sync: when /title is used, update RC topic.
        # This runs BEFORE the gateway processes the /title command so both happen:
        # RC topic is updated (here) and session title is set (in gateway).
        if _found_slash_cmd and cmd_full.startswith("/title "):
            _title_val = cmd_full[len("/title "):].strip()
            if _title_val:
                _topic_endpoint = self._set_topic_endpoint(chat_type)
                try:
                    data = await self._api_post(_topic_endpoint, {
                        "roomId": room_id,
                        "topic": _title_val,
                    })
                    if data and data.get("success"):
                        self._last_topic[room_id] = _title_val
                except Exception:
                    logger.debug("Failed to sync RC topic from /title via %s", _topic_endpoint, exc_info=True)

        msg_type = MessageType.TEXT
        if message_text.startswith("/"):
            msg_type = MessageType.COMMAND
        # Also handle the case where routing found a / but RC didn't know it
        # (message_text might still contain @mention in DMs)
        if _found_slash_cmd and msg_type != MessageType.COMMAND:
            msg_type = MessageType.COMMAND

        media_urls, media_types = await self._download_attachments(post)

        if media_types and msg_type == MessageType.TEXT:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
            elif any(m.startswith("audio/") for m in media_types):
                msg_type = MessageType.VOICE
            else:
                msg_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=room_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
        )

        from gateway.platforms.base import resolve_channel_prompt
        channel_prompt = resolve_channel_prompt(
            self.config.extra, room_id, None,
        )

        msg_event = MessageEvent(
            text=message_text,
            message_type=msg_type,
            source=source,
            raw_message=post,
            message_id=post_id,
            media_urls=media_urls if media_urls else None,
            media_types=media_types if media_types else None,
            channel_prompt=channel_prompt,
        )

        await self.handle_message(msg_event)

    async def _resolve_room_type(self, room_id: str) -> str:
        """Look up a room's type via REST. Defaults to 'channel' on failure."""
        data = await self._api_get("rooms.info", params={"roomId": room_id})
        room = (data or {}).get("room") or {}
        return _ROOM_TYPE_MAP.get(room.get("t", "c"), "channel")

    async def _download_attachments(
        self, post: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Download every file attached to *post* into the local cache."""
        import aiohttp

        media_urls: List[str] = []
        media_types: List[str] = []

        candidates: List[Dict[str, str]] = []

        # Primary single-file attachment.
        primary = post.get("file") or {}
        if isinstance(primary, dict) and primary.get("_id"):
            candidates.append({
                "id": primary["_id"],
                "name": primary.get("name", f"file_{primary['_id']}"),
                "type": primary.get("type", "application/octet-stream"),
            })

        # Multi-attachment payload.
        for att in post.get("attachments") or []:
            if not isinstance(att, dict):
                continue
            path = (
                att.get("image_url")
                or att.get("audio_url")
                or att.get("video_url")
                or att.get("title_link")
                or ""
            )
            m = re.match(r"^/file-upload/([^/?#]+)/([^/?#]+)", path)
            if not m:
                continue
            fid = m.group(1)
            if any(c["id"] == fid for c in candidates):
                continue
            fname = att.get("title") or m.group(2)
            if att.get("image_url"):
                mime = att.get("image_type") or "image/png"
            elif att.get("audio_url"):
                mime = att.get("audio_type") or "audio/ogg"
            elif att.get("video_url"):
                mime = att.get("video_type") or "video/mp4"
            else:
                mime = "application/octet-stream"
            candidates.append({"id": fid, "name": fname, "type": mime})

        for cand in candidates:
            try:
                url = f"{self._base_url}/file-upload/{cand['id']}/{cand['name']}"
                async with self._session.get(
                    url,
                    headers={
                        "X-Auth-Token": self._token,
                        "X-User-Id": self._bot_user_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning("Rocket.Chat: failed to download file %s: HTTP %s",
                                       cand["id"], resp.status)
                        continue
                    file_data = await resp.read()
                    mime = resp.content_type or cand["type"]
                    ext = Path(cand["name"]).suffix

                    from gateway.platforms.base import (
                        cache_image_from_bytes,
                        cache_audio_from_bytes,
                        cache_document_from_bytes,
                    )
                    if mime.startswith("image/"):
                        local_path = cache_image_from_bytes(file_data, ext or ".png")
                    elif mime.startswith("audio/"):
                        # Convert to MP3 first (Groq STT needs a widely-supported format)
                        raw_ext = ext or ".ogg"
                        raw_path = cache_audio_from_bytes(file_data, raw_ext)
                        local_path = await self._convert_audio_to_mp3(raw_path)
                        if local_path is None:
                            local_path = raw_path  # fallback: use original
                    else:
                        local_path = cache_document_from_bytes(file_data, cand["name"])
                    media_urls.append(local_path)
                    media_types.append(mime)
            except Exception as exc:
                logger.warning("Rocket.Chat: error downloading file %s: %s", cand["id"], exc)

        return media_urls, media_types

    # ── Audio conversion ──────────────────────────────────────────────

    async def _convert_audio_to_mp3(self, src_path: str) -> str | None:
        """Convert an audio file to MP3 using ffmpeg (for STT compatibility).

        Returns the converted MP3 path, or None if conversion failed.
        ffmpeg must be installed on the system.
        """
        if src_path.endswith(".mp3"):
            return src_path  # already MP3, skip
        dst_path = src_path.rsplit(".", 1)[0] + ".mp3"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1",
                "-b:a", "64k", dst_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                return dst_path
            logger.warning("Rocket.Chat: ffmpeg conversion failed (rc=%d)", proc.returncode)
        except FileNotFoundError:
            logger.warning("Rocket.Chat: ffmpeg not found — audio sent as-is to STT")
        except Exception as exc:
            logger.warning("Rocket.Chat: ffmpeg error: %s", exc)
        return None

    # ── Reactions ─────────────────────────────────────────────────────

    async def _add_reaction(self, message_id: str, emoji: str) -> bool:
        """Add an emoji reaction to a Rocket.Chat message.

        Rocket.Chat uses ``POST /api/v1/chat.react``. If the bot already
        reacted with this emoji, it removes the reaction (toggle).
        """
        data = await self._api_post(
            "chat.react",
            {"messageId": message_id, "emoji": emoji},
        )
        return bool(data and data.get("success"))

    async def _remove_reaction(self, message_id: str, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a message.

        ``chat.react`` toggles — calling it again removes the reaction.
        """
        return await self._add_reaction(message_id, emoji)

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("ROCKETCHAT_REACTIONS", "true").lower() not in {
            "false", "0", "no",
        }

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress 👀 reaction when processing begins."""
        if not self._reactions_enabled():
            return
        message_id = event.message_id
        if message_id:
            await self._add_reaction(message_id, ":eyes:")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the 👀 reaction for ✅ (success) or ❌ (failure)."""
        if not self._reactions_enabled():
            return
        message_id = event.message_id
        if not message_id:
            return
        await self._remove_reaction(message_id, ":eyes:")
        if outcome == ProcessingOutcome.SUCCESS:
            await self._add_reaction(message_id, ":white_check_mark:")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._add_reaction(message_id, ":x:")


# ---------------------------------------------------------------------------
# Interactive setup wizard
# ---------------------------------------------------------------------------


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow for the Rocket.Chat platform."""
    from hermes_cli.setup import (
        prompt,
        prompt_yes_no,
        save_env_value,
        get_env_value,
        print_header,
        print_info,
        print_warning,
        print_success,
    )

    print_header("Rocket.Chat")
    existing_url = get_env_value("ROCKETCHAT_URL")
    if existing_url:
        print_info(f"Rocket.Chat: already configured (server: {existing_url})")
        if not prompt_yes_no("Reconfigure Rocket.Chat?", False):
            return

    print_info("Connect Hermes to a self-hosted Rocket.Chat instance.")
    print_info("   Uses REST API v1 for outbound and DDP WebSocket for inbound messages.")
    print()

    url = prompt("Rocket.Chat server URL (e.g. https://rc.example.com)", default=existing_url or "")
    if not url:
        print_warning("Server URL is required — skipping Rocket.Chat setup")
        return
    save_env_value("ROCKETCHAT_URL", url.strip())

    print()
    print_info("🔑 Authentication")
    print_info("   Generate a Personal Access Token in your Rocket.Chat profile")
    print_info("   (My Account → Security → Personal Access Tokens)")
    print_info("   Make sure 'Ignore Two Factor' is checked.")

    token = prompt("Personal Access Token", password=True)
    if not token:
        print_warning("Token is required — skipping Rocket.Chat setup")
        return
    save_env_value("ROCKETCHAT_TOKEN", token.strip())

    user_id = prompt("Bot user _id (shown at PAT creation)")
    if not user_id:
        print_warning("User ID is required — skipping Rocket.Chat setup")
        return
    save_env_value("ROCKETCHAT_USER_ID", user_id.strip())

    print()
    print_info("⚙️  Options")

    reply_mode = prompt_yes_no("Use threaded replies in channels?", False)
    if reply_mode:
        save_env_value("ROCKETCHAT_REPLY_MODE", "thread")

    require_mention = prompt_yes_no("Require @mention in channels?", True)
    save_env_value("ROCKETCHAT_REQUIRE_MENTION", "true" if require_mention else "false")

    home = prompt("Home channel room ID (for cron/notification delivery)", default="")
    if home:
        save_env_value("ROCKETCHAT_HOME_CHANNEL", home.strip())

    print()
    print_info("🔒 Access control")
    if prompt_yes_no("Allow all users to talk to the bot?", False):
        save_env_value("ROCKETCHAT_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — any user on this instance can command the bot.")
    else:
        save_env_value("ROCKETCHAT_ALLOW_ALL_USERS", "false")
        allowed = prompt(
            "Allowed user IDs (comma-separated, leave empty to deny everyone)",
            default=get_env_value("ROCKETCHAT_ALLOWED_USERS") or "",
        )
        if allowed:
            save_env_value("ROCKETCHAT_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("ROCKETCHAT_ALLOWED_USERS", "")
            print_info("No users allowed — bot will ignore all messages until you add IDs.")

    print()
    print_success("Rocket.Chat configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="rocketchat",
        label="Rocket.Chat",
        adapter_factory=lambda cfg: RocketchatAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["ROCKETCHAT_URL", "ROCKETCHAT_TOKEN", "ROCKETCHAT_USER_ID"],
        install_hint="Uses aiohttp (already a Hermes dependency) — no extra packages needed",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="ROCKETCHAT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="ROCKETCHAT_ALLOWED_USERS",
        allow_all_env="ROCKETCHAT_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🚀",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Rocket.Chat. Rocket.Chat renders Markdown natively. "
            "In channels, users must @mention you for the bot to respond (unless the room "
            "is in the free-response list). Replies can be threaded (ROCKETCHAT_REPLY_MODE). "
            "Keep responses clear and concise."
        ),
    )
