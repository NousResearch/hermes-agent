"""SimpleX Chat platform adapter (Hermes plugin).

Connects to a simplex-chat daemon running in WebSocket mode.
Inbound messages arrive via a persistent WebSocket connection.
Outbound messages use the same WebSocket with JSON commands.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/simplex/``. The Hermes plugin loader scans the
directory at startup, calls ``register(ctx)``, and the platform
becomes available to ``gateway/run.py`` and ``tools/send_message_tool``
through the registry — no edits to core files are required.

SimpleX chat daemon setup:
    simplex-chat -p 5225          # start daemon on port 5225
    # or via Docker:
    # docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225

Required environment variables:
    SIMPLEX_WS_URL             WebSocket URL of the daemon
                               (default: ws://127.0.0.1:5225)

Optional environment variables:
    SIMPLEX_ALLOWED_USERS      Comma-separated contact IDs (allowlist)
    SIMPLEX_ALLOW_ALL_USERS    Set 'true' to allow all contacts
    SIMPLEX_HOME_CHANNEL       Default contact/group ID for cron delivery
    SIMPLEX_HOME_CHANNEL_NAME  Human label for the home channel

The ``websockets`` Python package is imported lazily. The plugin remains
discoverable and `hermes setup` can describe it even when the dependency
is missing; a configured gateway instance then fails loudly from
``connect()`` instead of disappearing during platform discovery.
"""

import asyncio
import json
import logging
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import: BasePlatformAdapter and friends live in the main repo.
# Imported at module top because they're stdlib-only inside Hermes — no
# external dependency that would block the plugin from loading.
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 16_000  # SimpleX has no hard limit; keep chunking sane
TYPING_INTERVAL = 10.0
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 120.0

# Correlation ID prefix for requests we send so we can ignore our own echoes.
_CORR_PREFIX = "hermes-"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a stripped list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _chat_ref_for_chat_id(chat_id: str) -> str:
    """Return SimpleX API chat reference syntax for a Hermes chat id."""
    if chat_id.startswith("group:"):
        return f"#{chat_id[6:]}"
    return f"@{chat_id}"


def _text_message_payload(content: str) -> List[dict]:
    return [
        {
            "msgContent": {
                "type": "text",
                "text": content,
            },
            "mentions": {},
        }
    ]


def _updated_text_payload(content: str) -> dict:
    return {
        "msgContent": {
            "type": "text",
            "text": content,
        },
        "mentions": {},
    }


def _text_send_command(chat_id: str, content: str, *, live: bool = False) -> str:
    """Build a SimpleX API text-send command for the WebSocket daemon."""
    payload = json.dumps(_text_message_payload(content), separators=(",", ":"))
    live_flag = " live=on" if live else ""
    return f"/_send {_chat_ref_for_chat_id(chat_id)}{live_flag} json {payload}"


def _text_update_command(chat_id: str, item_id: str, content: str, *, live: bool = False) -> str:
    """Build a SimpleX API message-update command."""
    payload = json.dumps(_updated_text_payload(content), separators=(",", ":"))
    live_flag = " live=on" if live else ""
    return f"/_update item {_chat_ref_for_chat_id(chat_id)} {item_id}{live_flag} json {payload}"


def _voice_send_command(
    chat_id: str,
    audio_path: str,
    *,
    duration: int,
    caption: str = "",
) -> str:
    composed_messages = [
        {
            "fileSource": {"filePath": audio_path},
            "msgContent": {
                "type": "voice",
                "text": caption or "",
                "duration": max(1, int(duration or 1)),
            },
            "mentions": {},
        }
    ]
    payload = json.dumps(composed_messages, separators=(",", ":"))
    return f"/_send {_chat_ref_for_chat_id(chat_id)} json {payload}"


def _native_call_offer_command(
    chat_id: str,
    offer: Any,
    *,
    media: str = "audio",
) -> str:
    if isinstance(offer, dict):
        rtc_session = offer.get("rtcSession", "")
        rtc_ice_candidates = offer.get("rtcIceCandidates", "")
        capabilities = dict(offer.get("capabilities") or {})
    else:
        rtc_session = getattr(offer, "rtc_session", "")
        rtc_ice_candidates = getattr(offer, "rtc_ice_candidates", "")
        capabilities = dict(getattr(offer, "capabilities", {}) or {})
    capabilities.setdefault("encryption", False)
    payload = {
        "callType": {
            "media": media,
            "capabilities": capabilities,
        },
        "rtcSession": {
            "rtcSession": rtc_session,
            "rtcIceCandidates": rtc_ice_candidates,
        },
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    return f"/_call offer {_chat_ref_for_chat_id(chat_id)} {encoded}"


def _extract_chat_item_id(resp: dict) -> Optional[str]:
    """Extract SimpleX chatItem.meta.itemId from command response shapes."""
    if not isinstance(resp, dict):
        return None
    items = resp.get("chatItems") or []
    if not items and resp.get("chatItem"):
        items = [resp.get("chatItem")]
    for wrapper in items:
        if not isinstance(wrapper, dict):
            continue
        chat_item = wrapper.get("chatItem") or wrapper
        meta = chat_item.get("meta") or {}
        item_id = meta.get("itemId")
        if item_id is not None:
            return str(item_id)
    return None


def _chat_error_text(resp: dict) -> str:
    err = resp.get("chatError") if isinstance(resp, dict) else None
    if isinstance(err, dict):
        return err.get("message") or err.get("type") or json.dumps(err, separators=(",", ":"))
    return str(err or resp.get("type") or "unknown SimpleX command error")


def _markdown_to_simplex(text: str) -> str:
    """Convert common Markdown into SimpleX's formatting dialect."""
    if not text:
        return text

    lines = text.split("\n")
    result: List[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            heading_text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", heading_text)
            heading_text = re.sub(r"__([^_]+)__", r"\1", heading_text)
            result.append(f"*{heading_text}*")
            continue

        if re.match(r"^[-*_]{3,}\s*$", stripped):
            result.append("--------")
            continue

        code_spans: List[str] = []

        def _save_code(match):
            code_spans.append(match.group(0))
            return f"\x00CODE{len(code_spans) - 1}\x00"

        converted = re.sub(r"`[^`]+`", _save_code, line)
        converted = re.sub(r"\*{3}([^*]+)\*{3}", r"*_\1_*", converted)
        converted = re.sub(r"_{3}([^_]+)_{3}", r"*_\1_*", converted)
        converted = re.sub(r"\*{2}([^*]+)\*{2}", r"*\1*", converted)
        converted = re.sub(r"__([^_]+)__", r"*\1*", converted)
        converted = re.sub(r"~~([^~]+)~~", r"~\1~", converted)
        converted = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[image: \1]", converted)
        converted = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", converted)

        for i, code in enumerate(code_spans):
            converted = converted.replace(f"\x00CODE{i}\x00", code)
        result.append(converted)

    return "\n".join(result)


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return ".mp4"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac"}


# ---------------------------------------------------------------------------
# SimpleX Adapter
# ---------------------------------------------------------------------------

class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API.

    Instantiated by the ``adapter_factory`` passed to
    ``ctx.register_platform()`` in :func:`register`.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = True
    REQUIRES_EDIT_FINALIZE = True
    STREAMS_WITH_LIVE_MESSAGES = True

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("simplex")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")
        self.files_folder = (
            extra.get("files_folder")
            or os.getenv("SIMPLEX_FILES_FOLDER", "").strip()
        )
        self.command_timeout = float(extra.get("command_timeout", 10.0) or 10.0)

        # Running state
        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._event_tasks: set[asyncio.Task] = set()
        self._running = False
        self._last_ws_activity = 0.0

        # Track sent correlation IDs to filter echoes
        self._pending_corr_ids: set = set()
        self._max_pending_corr = 200
        self._pending_responses: Dict[str, asyncio.Future] = {}

        logger.info("SimpleX adapter initialized: url=%s", self.ws_url)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the simplex-chat daemon and start the WebSocket listener."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error(
                "SimpleX: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return False

        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        # Quick connectivity check — try to open and immediately close
        try:
            import websockets as _wsclient
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())

        logger.info("SimpleX: connected to %s", self.ws_url)
        return True

    async def disconnect(self) -> None:
        """Stop WebSocket listener and clean up."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        for task in list(self._event_tasks):
            task.cancel()
        self._event_tasks.clear()
        for corr_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(ConnectionError("SimpleX WebSocket disconnected"))
            self._pending_responses.pop(corr_id, None)

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Maintain a persistent WebSocket connection to the daemon."""
        import websockets as _wsclient
        import websockets as _wsexc

        backoff = WS_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")

                    try:
                        await self._replay_unread_chats(ws)
                    except Exception:
                        logger.exception("SimpleX WS: failed to replay unread chats")

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            msg = json.loads(raw)
                            self._spawn_event_task(msg)
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")

            except asyncio.CancelledError:
                break
            except _wsexc.WebSocketException as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: error: %s (reconnecting in %.0fs)", e, backoff
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: unexpected error: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            finally:
                self._ws = None

            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Force reconnect if the WebSocket has been idle too long."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.warning(
                    "SimpleX: WS idle for %.0fs, forcing reconnect", elapsed
                )
                self._last_ws_activity = time.time()
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    def _spawn_event_task(self, msg: dict) -> None:
        task = asyncio.create_task(self._handle_event(msg))
        self._event_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._event_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("SimpleX WS: error handling event")

        task.add_done_callback(_done)

    async def _ws_request(self, ws: Any, cmd: str, timeout: float = 8.0) -> dict:
        """Send a daemon command on an existing socket and wait for its response."""
        corr_id = self._make_corr_id()
        await ws.send(json.dumps({"corrId": corr_id, "cmd": cmd}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            if msg.get("corrId") == corr_id:
                self._pending_corr_ids.discard(corr_id)
                return msg.get("resp") or msg
            self._spawn_event_task(msg)

    def _chat_ref_from_info(self, chat_info: dict) -> Optional[str]:
        chat_type = chat_info.get("type", "")
        if chat_type in {"group", "groupInfo"}:
            group_info = chat_info.get("groupInfo") or chat_info.get("group") or {}
            group_id = group_info.get("groupId") or group_info.get("id")
            return f"#{group_id}" if group_id else None
        contact = chat_info.get("contact") or {}
        contact_id = contact.get("contactId") or contact.get("id")
        return f"@{contact_id}" if contact_id else None

    def _is_received_unread_item(self, item: dict) -> bool:
        meta = item.get("meta") or {}
        status = meta.get("itemStatus") or {}
        status_type = status.get("type") if isinstance(status, dict) else str(status)
        content = item.get("content") or {}
        return status_type == "rcvNew" and content.get("type") == "rcvMsgContent"

    async def _replay_unread_chats(self, ws: Any) -> None:
        """Fetch unread messages missed while the persistent socket was down."""
        list_resp = await self._ws_request(
            ws,
            '/_get chats 1 pcc=on count=50 {"type":"filters","favorite":false,"unread":true}',
        )
        if list_resp.get("type") != "apiChats":
            return

        for chat in list_resp.get("chats") or []:
            chat_info = chat.get("chatInfo") or {}
            unread_count = int((chat.get("chatStats") or {}).get("unreadCount") or 0)
            if unread_count <= 0:
                continue
            chat_ref = self._chat_ref_from_info(chat_info)
            if not chat_ref:
                continue
            count = min(max(unread_count + 5, 6), 50)
            chat_resp = await self._ws_request(ws, f"/_get chat {chat_ref} count={count}")
            if chat_resp.get("type") != "apiChat":
                continue
            full_chat = chat_resp.get("chat") or {}
            full_chat_info = full_chat.get("chatInfo") or chat_info
            for item in full_chat.get("chatItems") or []:
                if self._is_received_unread_item(item):
                    await self._handle_new_chat_item(
                        {"chatInfo": full_chat_info, "chatItem": item}
                    )

    async def _mark_chat_items_read(self, chat_id: str, item_ids: List[Any]) -> None:
        ids = ",".join(str(item_id) for item_id in item_ids if item_id is not None)
        if not ids:
            return
        await self._send_ws(
            {
                "corrId": self._make_corr_id(),
                "cmd": f"/_read chat items {_chat_ref_for_chat_id(chat_id)} {ids}",
            }
        )

    async def _handle_event(self, event: dict) -> None:
        """Dispatch a daemon event to the appropriate handler."""
        payload = event.get("resp") if isinstance(event.get("resp"), dict) else event
        resp_type = payload.get("type", "")

        # Filter responses to our own commands (echoes)
        corr_id = event.get("corrId", "")
        if corr_id:
            future = self._pending_responses.pop(corr_id, None)
            if future is not None:
                self._pending_corr_ids.discard(corr_id)
                if not future.done():
                    future.set_result(payload)
                return
        if corr_id and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)
            return

        if resp_type == "newChatItem":
            await self._handle_new_chat_item(payload)
        elif resp_type == "newChatItems":
            # Batch variant — process each item
            chat_info = payload.get("chatInfo") or payload.get("chat") or {}
            items = payload.get("chatItems") or []
            for item_wrapper in items:
                if chat_info and not (
                    item_wrapper.get("chatInfo") or item_wrapper.get("chat")
                ):
                    item_wrapper = {"chatInfo": chat_info, "chatItem": item_wrapper}
                await self._handle_new_chat_item(item_wrapper)
        elif resp_type == "callInvitation":
            await self._handle_call_invitation(payload)
        # Ignore all other event types (delivery receipts, contact updates, etc.)

    async def _handle_call_invitation(self, payload: dict) -> None:
        """Process a SimpleX native-call invitation event."""
        invitation = payload.get("callInvitation")
        if not isinstance(invitation, dict):
            invitation = payload
        contact = invitation.get("contact") if isinstance(invitation, dict) else {}
        if not isinstance(contact, dict):
            contact = {}
        contact_id = str(contact.get("contactId") or contact.get("id") or "")
        if not contact_id:
            logger.warning("SimpleX: call invitation missing contact id: %s", payload)
            return
        contact_name = (
            contact.get("displayName")
            or contact.get("localDisplayName")
            or contact_id
        )
        source = self.build_source(
            chat_id=contact_id,
            chat_name=contact_name,
            chat_type="dm",
            user_id=contact_id,
            user_name=contact_name,
        )
        await self._handle_native_call_item(
            source,
            contact_id,
            {"type": "rcvCall", "status": "pending", "duration": 0},
            {},
        )

    async def _handle_new_chat_item(self, wrapper: dict) -> None:
        """Process a single newChatItem event into a MessageEvent."""
        # The daemon wraps the chat item differently depending on version;
        # normalise both layouts.
        chat_info = wrapper.get("chatInfo") or wrapper.get("chat") or {}
        chat_item = wrapper.get("chatItem") or wrapper.get("item") or {}
        if not chat_item and (
            "content" in wrapper or "meta" in wrapper or "chatDir" in wrapper
        ):
            chat_item = wrapper

        # Filter out messages sent by us (direction == "snd")
        meta = chat_item.get("meta") or {}
        direction = (meta.get("itemStatus") or {}).get("type", "")
        if direction in {"sndSent", "sndSentDirect", "sndSentViaProxy", "sndNew"}:
            return

        # Determine chat type and IDs
        chat_type_raw = chat_info.get("type", "")
        is_group = chat_type_raw in {"group", "groupInfo"}

        if is_group:
            group_info = chat_info.get("groupInfo") or chat_info.get("group") or {}
            group_id = str(group_info.get("groupId") or group_info.get("id") or "")
            group_name = group_info.get("displayName") or group_info.get("groupProfile", {}).get("displayName", "")
            chat_id = f"group:{group_id}" if group_id else ""
            chat_name = group_name
        else:
            contact_info = chat_info.get("contact") or {}
            contact_id = str(contact_info.get("contactId") or contact_info.get("id") or "")
            contact_name = (
                contact_info.get("displayName")
                or contact_info.get("localDisplayName")
                or contact_id
            )
            chat_id = contact_id
            chat_name = contact_name

        if not chat_id:
            logger.debug("SimpleX: ignoring event with no chat_id")
            return

        # Sender — for groups the message includes a chatItemMember sub-object
        member = chat_item.get("chatItemMember") or {}
        if is_group and member:
            sender_id = str(member.get("memberId") or member.get("id") or chat_id)
            sender_name = (
                member.get("displayName")
                or member.get("localDisplayName")
                or sender_id
            )
        else:
            sender_id = chat_id
            sender_name = chat_name

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name,
        )

        item_content = chat_item.get("content") or {}
        content_type = str(item_content.get("type") or "")
        if content_type == "rcvCall":
            await self._handle_native_call_item(source, chat_id, item_content, meta)
            return

        # Only process regular message content here.
        msg_content = item_content.get("msgContent") or {}
        if not msg_content:
            return

        # Extract text
        text = msg_content.get("text") or ""

        # Media attachments
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item.get("file") or {}
        file_status = file_info.get("fileStatus") if file_info else None
        file_status_type = (
            file_status.get("type") if isinstance(file_status, dict) else file_status
        )
        if file_info and file_status_type not in {"cancelled", "error"}:
            file_id = file_info.get("fileId")
            file_name = file_info.get("fileName", "file")
            if file_id:
                try:
                    cached = await self._fetch_file(file_id, file_name)
                    if cached:
                        ext = cached.rsplit(".", 1)[-1]
                        if _is_image_ext("." + ext):
                            media_types.append("image/" + ext.replace("jpg", "jpeg"))
                        elif _is_audio_ext("." + ext):
                            media_types.append("audio/" + ext)
                        else:
                            media_types.append("application/octet-stream")
                        media_urls.append(cached)
                except Exception:
                    logger.exception("SimpleX: failed to fetch file %s", file_id)

        # Timestamp
        ts_str = meta.get("itemTs") or meta.get("createdAt") or ""
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        # Message type
        msg_type = MessageType.TEXT
        simplex_msg_type = msg_content.get("type", "")
        if media_types:
            if simplex_msg_type == "voice" and any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.AUDIO
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO

        event_obj = MessageEvent(
            source=source,
            text=text,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message=wrapper,
        )

        await self.handle_message(event_obj)
        item_id = meta.get("itemId")
        if item_id is not None:
            await self._mark_chat_items_read(chat_id, [item_id])

    async def _handle_native_call_item(
        self,
        source,
        chat_id: str,
        item_content: dict,
        meta: dict,
    ) -> None:
        """Reject unsupported SimpleX native calls with a clear fallback."""
        item_id = meta.get("itemId")
        status = str(item_content.get("status") or "").lower()
        if status != "pending":
            logger.info(
                "SimpleX: native call event ignored: chat_id=%s status=%s item_id=%s",
                chat_id,
                status or "unknown",
                item_id,
            )
            if item_id is not None:
                await self._mark_chat_items_read(chat_id, [item_id])
            return

        authorized = True
        runner = getattr(self, "gateway_runner", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                authorized = bool(auth_fn(source))
            except Exception:
                authorized = False
                logger.exception(
                    "SimpleX: authorization check failed for native call from chat_id=%s",
                    chat_id,
                )

        if not authorized:
            logger.warning(
                "SimpleX: rejected unauthorized SimpleX native call from user_id=%s chat_id=%s",
                getattr(source, "user_id", None),
                chat_id,
            )
            await self._reject_native_call(chat_id)
            if item_id is not None:
                await self._mark_chat_items_read(chat_id, [item_id])
            return

        logger.warning(
            "SimpleX: rejecting native call from chat_id=%s because native WebRTC bridge is unavailable",
            chat_id,
        )
        rejected = await self._reject_native_call(chat_id)
        note = (
            "I saw your SimpleX native call, but I cannot answer native SimpleX calls yet. "
            "Use /call for the private browser fallback."
        )
        if not rejected:
            note = (
                "I saw your SimpleX native call, but I could not reject it automatically. "
                "Use /call for the private browser fallback."
            )
        result = await self.send(chat_id, note)
        if isinstance(result, SendResult) and not result.success:
            logger.error(
                "SimpleX: failed to send native-call fallback to chat_id=%s: %s",
                chat_id,
                result.error,
            )
        if item_id is not None:
            await self._mark_chat_items_read(chat_id, [item_id])

    async def send_native_call_offer(
        self,
        chat_id: str,
        offer: dict,
        *,
        media: str = "audio",
    ) -> bool:
        """Send a SimpleX native-call offer command."""
        await self._send_command(_native_call_offer_command(chat_id, offer, media=media))
        return True

    async def send_native_call_status(self, chat_id: str, status: str) -> bool:
        """Send a SimpleX native-call status command."""
        await self._send_command(
            f"/_call status {_chat_ref_for_chat_id(chat_id)} {status}"
        )
        return True

    async def end_native_call(self, chat_id: str) -> bool:
        """End an active SimpleX native call."""
        await self._send_command(f"/_call end {_chat_ref_for_chat_id(chat_id)}")
        return True

    async def reject_native_call(self, chat_id: str, reason_code: str = "") -> bool:
        """Tell SimpleX to reject a pending native call."""
        try:
            await self._send_command(f"/_call reject {_chat_ref_for_chat_id(chat_id)}")
            return True
        except Exception as exc:
            logger.error(
                "SimpleX: failed to reject native call for chat_id=%s: %s",
                chat_id,
                exc,
                exc_info=True,
            )
            return False

    async def _reject_native_call(self, chat_id: str) -> bool:
        """Backward-compatible private alias for native call rejection."""
        return await self.reject_native_call(chat_id)

    async def send_offer(self, contact_id: str, offer: Any) -> None:
        """NativeCallSignalingPort: send a media offer to a contact."""
        await self.send_native_call_offer(contact_id, offer)

    async def send_status(self, contact_id: str, status: str) -> None:
        """NativeCallSignalingPort: send call status to a contact."""
        await self.send_native_call_status(contact_id, status)

    async def reject(self, contact_id: str, reason_code: str) -> None:
        """NativeCallSignalingPort: reject a native call."""
        await self.reject_native_call(contact_id, reason_code)

    async def end(self, contact_id: str) -> None:
        """NativeCallSignalingPort: end a native call."""
        await self.end_native_call(contact_id)

    async def _fetch_file(self, file_id: Any, file_name: str) -> Optional[str]:
        """Ask the daemon to receive and return a file attachment."""
        # simplex-chat exposes `/api/v1/files/{fileId}` on an HTTP port
        # when started with --http-port. However, the canonical WebSocket API
        # does not have a direct binary download command; files are stored on
        # the local filesystem after the daemon accepts them.
        #
        # We request acceptance into an explicit local path, then cache from
        # that path only. This avoids broad folder scans and path traversal via
        # remote-supplied file names.
        safe_name = os.path.basename(str(file_name or "file")) or "file"
        hermes_home = os.getenv("HERMES_HOME", "").strip()
        receive_dir = (
            self.files_folder
            or os.getenv("SIMPLEX_FILES_FOLDER", "").strip()
            or (os.path.join(hermes_home, "simplex-bot", "files") if hermes_home else "")
            or os.path.join(tempfile.gettempdir(), "hermes-simplex-files")
        )
        receive_path = Path(receive_dir).expanduser() / safe_name
        receive_path.parent.mkdir(parents=True, exist_ok=True)

        await self._send_command(
            f"/freceive {file_id} approved_relays=on inline=on {receive_path}"
        )

        # The daemon can create a zero-byte placeholder before XFTP completes.
        for _ in range(20):
            if receive_path.exists() and receive_path.stat().st_size > 0:
                data = receive_path.read_bytes()
                ext = _guess_extension(data)
                original_ext = os.path.splitext(safe_name)[1]
                if _is_image_ext(ext):
                    return cache_image_from_bytes(data, ext)
                if _is_audio_ext(ext) or _is_audio_ext(original_ext):
                    return cache_audio_from_bytes(
                        data, ext if _is_audio_ext(ext) else original_ext
                    )
                return cache_document_from_bytes(data, safe_name)
            await asyncio.sleep(0.5)
        logger.error(
            "SimpleX: received file %s was not available at %s after waiting",
            file_id, receive_path,
        )
        return None

    # ------------------------------------------------------------------
    # Outbound messages
    # ------------------------------------------------------------------

    def _make_corr_id(self) -> str:
        """Generate a unique correlation ID for a request."""
        corr_id = f"{_CORR_PREFIX}{int(time.time() * 1000)}-{random.randint(0, 9999)}"
        self._pending_corr_ids.add(corr_id)
        if len(self._pending_corr_ids) > self._max_pending_corr:
            # Trim oldest — sets are unordered so just clear the oldest half
            to_remove = list(self._pending_corr_ids)[:self._max_pending_corr // 2]
            self._pending_corr_ids -= set(to_remove)
        return corr_id

    async def _send_ws(self, payload: dict) -> bool:
        """Send a JSON payload over the WebSocket."""
        ws = self._ws
        if not ws:
            logger.error("SimpleX: WebSocket not connected; outbound command failed")
            return False
        import websockets as _wsexc
        try:
            await ws.send(json.dumps(payload))
            return True
        except _wsexc.ConnectionClosed:
            logger.error("SimpleX: WebSocket closed while sending outbound command")
            return False
        except Exception as e:
            logger.error("SimpleX: WebSocket send error: %s", e, exc_info=True)
            return False

    async def _send_command(self, cmd: str, *, timeout: Optional[float] = None) -> dict:
        """Send a correlated SimpleX command and wait for its response."""
        if not self._ws:
            logger.error("SimpleX: WebSocket not connected; cannot run command: %s", cmd)
            raise ConnectionError("SimpleX WebSocket not connected")

        corr_id = self._make_corr_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_responses[corr_id] = future
        ok = await self._send_ws({"corrId": corr_id, "cmd": cmd})
        if not ok:
            self._pending_responses.pop(corr_id, None)
            self._pending_corr_ids.discard(corr_id)
            raise ConnectionError("SimpleX WebSocket send failed")

        try:
            resp = await asyncio.wait_for(
                future,
                timeout=self.command_timeout if timeout is None else timeout,
            )
        except asyncio.TimeoutError:
            self._pending_responses.pop(corr_id, None)
            self._pending_corr_ids.discard(corr_id)
            logger.error("SimpleX: command timed out: %s", cmd)
            raise

        if isinstance(resp, dict) and resp.get("type") == "chatCmdError":
            error = _chat_error_text(resp)
            logger.error("SimpleX: command failed: %s (%s)", cmd, error)
            raise RuntimeError(error)
        return resp

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a contact or group."""
        live = bool((metadata or {}).get("_hermes_live_stream"))
        formatted = self.format_message(content)
        cmd = _text_send_command(chat_id, formatted, live=live)
        try:
            resp = await self._send_command(cmd)
        except (ConnectionError, asyncio.TimeoutError) as e:
            return SendResult(success=False, error=str(e), retryable=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

        message_id = _extract_chat_item_id(resp)
        if live and not message_id:
            logger.error("SimpleX: live send did not return chat item id: %s", resp)
            return SendResult(
                success=False,
                error="SimpleX live send did not return chat item id",
                raw_response=resp,
                retryable=True,
            )
        return SendResult(success=True, message_id=message_id, raw_response=resp)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Update a SimpleX live message, finalizing it on the last edit."""
        formatted = self.format_message(content)
        cmd = _text_update_command(
            chat_id,
            str(message_id),
            formatted,
            live=not finalize,
        )
        try:
            resp = await self._send_command(cmd)
        except (ConnectionError, asyncio.TimeoutError) as e:
            return SendResult(success=False, error=str(e), retryable=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

        resp_type = resp.get("type") if isinstance(resp, dict) else ""
        if resp_type in {"chatItemUpdated", "chatItemNotChanged", "newChatItems"}:
            return SendResult(success=True, message_id=str(message_id), raw_response=resp)
        logger.error("SimpleX: unexpected edit response for %s: %s", message_id, resp)
        return SendResult(success=False, error=f"Unexpected SimpleX edit response: {resp_type}", raw_response=resp)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX does not expose a typing indicator API — no-op."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image (URL) as a message with optional caption.

        SimpleX has no native ``send_image`` over the WebSocket API — file
        attachments require the daemon's filesystem-backed flow which is
        not driven from this adapter. Fall back to a plain text message
        containing the URL and caption.
        """
        text = f"{caption}\n{image_url}".strip() if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a native SimpleX voice message from a local audio file."""
        if not os.path.isfile(audio_path):
            error = f"SimpleX voice file does not exist: {audio_path}"
            logger.error(error)
            return SendResult(success=False, error=error)
        duration = kwargs.get("duration") or self._estimate_audio_duration(audio_path)
        cmd = _voice_send_command(
            chat_id,
            audio_path,
            duration=duration,
            caption=caption or "",
        )
        try:
            resp = await self._send_command(cmd)
        except (ConnectionError, asyncio.TimeoutError) as e:
            return SendResult(success=False, error=str(e), retryable=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(
            success=True,
            message_id=_extract_chat_item_id(resp),
            raw_response=resp,
        )

    @staticmethod
    def _estimate_audio_duration(audio_path: str) -> int:
        try:
            size = os.path.getsize(audio_path)
        except OSError:
            return 1
        # Rough MP3/Opus estimate at 128kbps; enough for SimpleX display.
        return max(1, int(round(size / 16000)))

    def format_message(self, content: str) -> str:
        """Convert Markdown to SimpleX's native rendering dialect."""
        return _markdown_to_simplex(content)

    async def get_chat_info(self, chat_id: str) -> dict:
        """Return basic chat info."""
        if chat_id.startswith("group:"):
            return {"chat_id": chat_id, "type": "group", "name": chat_id[6:]}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: keep SimpleX visible so config validation can run.

    The WebSocket URL may come from environment variables or config.yaml,
    and dependency failures belong in ``connect()`` so configured SimpleX
    platforms fail loudly in gateway logs instead of disappearing during
    platform discovery.
    """
    return True


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def is_connected(config) -> bool:
    """Check whether SimpleX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the WebSocket
    client. Returns ``None`` when SimpleX isn't minimally configured.

    The special ``home_channel`` key in the returned dict is handled by
    the core hook — it becomes a proper ``HomeChannel`` dataclass on the
    ``PlatformConfig`` rather than being merged into ``extra``.
    """
    ws_url = os.getenv("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}
    home = os.getenv("SIMPLEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home,
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
    """Open an ephemeral WebSocket to the daemon, send, and close.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``). Without this hook,
    ``deliver=simplex`` cron jobs fail with "No live adapter for platform".

    ``thread_id`` and ``force_document`` are accepted for signature parity
    with other plugins but are not meaningful here. ``media_files`` is
    accepted but only the text body is delivered — SimpleX requires the
    daemon's filesystem-backed file flow which an ephemeral connection
    cannot drive safely.
    """
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}

    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "ws://127.0.0.1:5225")
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}

    try:
        payload = {
            "corrId": f"hermes-snd-{int(time.time() * 1000)}",
            "cmd": _text_send_command(chat_id, message),
        }

        async with _wsclient.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(raw)
                if msg.get("corrId") != payload["corrId"]:
                    continue
                resp = msg.get("resp") if isinstance(msg.get("resp"), dict) else msg
                break
            if isinstance(resp, dict) and resp.get("type") == "chatCmdError":
                return {"error": f"SimpleX send failed: {_chat_error_text(resp)}"}

        result = {"success": True, "platform": "simplex", "chat_id": chat_id}
        message_id = _extract_chat_item_id(resp if isinstance(resp, dict) else {})
        if message_id:
            result["message_id"] = message_id
        return result
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX.

    Prompts for the WebSocket URL and the optional allowlist / home channel.
    Writes to ``~/.hermes/.env`` via ``hermes_cli.config``.
    """
    print()
    print("SimpleX Chat setup")
    print("------------------")
    print("Requirements:")
    print("  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).")
    print("  2. Python package `websockets` installed (`pip install websockets`).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set SIMPLEX_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                import getpass
                value = getpass.getpass(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)")
    _prompt("SIMPLEX_ALLOWED_USERS", "Allowed contact IDs (comma-separated; blank=skip)")
    _prompt("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)")
    print("Done. Make sure the simplex-chat daemon is running before starting the gateway.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint="pip install websockets   # SimpleX adapter requires the websockets package",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups show up in `hermes gateway status` without
        # instantiating the adapter.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support — `deliver=simplex` cron jobs
        # route to SIMPLEX_HOME_CHANNEL when set.
        cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        # Out-of-process cron delivery. Without this hook, deliver=simplex
        # cron jobs fail with "No live adapter" when cron runs separately
        # from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
        # SimpleX has no hard line length; we still chunk for sanity.
        max_message_length=MAX_MESSAGE_LENGTH,
        # Display
        emoji="🔒",
        # SimpleX uses opaque contact IDs only — no phone numbers or
        # email addresses to redact.
        pii_safe=True,
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational."
        ),
    )
