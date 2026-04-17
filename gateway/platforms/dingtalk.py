"""
DingTalk platform adapter using Stream Mode.

Uses dingtalk-stream SDK for real-time message reception without webhooks.
Responses are sent via DingTalk's session webhook (markdown format).
Outbound rich media (image, voice, video, file) sent via DingTalk Robot OpenAPI.
Inbound media (image, audio, file) downloaded and cached for agent consumption.

Requires:
    pip install dingtalk-stream httpx
    DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET env vars

Configuration in config.yaml:
    platforms:
      dingtalk:
        enabled: true
        extra:
          client_id: "your-app-key"      # or DINGTALK_CLIENT_ID env var
          client_secret: "your-secret"   # or DINGTALK_CLIENT_SECRET env var
          robot_code: "optional-robot-code"  # or DINGTALK_ROBOT_CODE env var (defaults to client_id)
"""

import asyncio
import inspect
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

try:
    import dingtalk_stream
    from dingtalk_stream import ChatbotHandler, ChatbotMessage
    DINGTALK_STREAM_AVAILABLE = True
except ImportError:
    DINGTALK_STREAM_AVAILABLE = False
    dingtalk_stream = None  # type: ignore[assignment]

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 20000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
_SESSION_WEBHOOKS_MAX = 500
_SESSION_CONTEXT_MAX = 500
_DINGTALK_WEBHOOK_RE = re.compile(r'^https://api\.dingtalk\.com/')
_QUERY_SECRET_RE = re.compile(r"(?P<key>session|access_token)=([^&]+)", re.IGNORECASE)
_LONG_SECRET_RE = re.compile(r"([A-Za-z0-9+/=_-]{16,})")
_PLACEHOLDER_MEDIA_PATHS = frozenset({
    "/absolute/path",
    "/path/to/file",
    "/文件路径",
})
_PLACEHOLDER_PATH_PARTS = frozenset({"...", "…"})


def check_dingtalk_requirements() -> bool:
    """Check if DingTalk dependencies are available and configured."""
    if not DINGTALK_STREAM_AVAILABLE or not HTTPX_AVAILABLE:
        return False
    if not os.getenv("DINGTALK_CLIENT_ID") or not os.getenv("DINGTALK_CLIENT_SECRET"):
        return False
    return True


class DingTalkAdapter(BasePlatformAdapter):
    """DingTalk chatbot adapter using Stream Mode.

    The dingtalk-stream SDK maintains a long-lived WebSocket connection.
    Incoming messages arrive via a ChatbotHandler callback. Replies are
    sent via the incoming message's session_webhook URL using httpx.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DINGTALK)

        extra = config.extra or {}
        self._client_id: str = extra.get("client_id") or os.getenv("DINGTALK_CLIENT_ID", "")
        self._client_secret: str = extra.get("client_secret") or os.getenv("DINGTALK_CLIENT_SECRET", "")
        self._configured_robot_code: str = str(
            extra.get("robot_code")
            or os.getenv("DINGTALK_ROBOT_CODE", "")
            or ""
        )
        self._robot_code: str = str(
            self._configured_robot_code
            or self._client_id
            or ""
        )

        self._stream_client: Any = None
        self._stream_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

        # Message deduplication
        self._dedup = MessageDeduplicator(max_size=1000)
        # Map chat_id -> session_webhook for reply routing
        self._session_webhooks: Dict[str, str] = {}
        # Chat/session context for media upload-link replies
        self._session_context: Dict[str, Dict[str, Any]] = {}
        # Access token cache
        self._access_token: str = ""
        self._access_token_expires_at: float = 0.0

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Connect to DingTalk via Stream Mode."""
        if not DINGTALK_STREAM_AVAILABLE:
            logger.warning("[%s] dingtalk-stream not installed. Run: pip install dingtalk-stream", self.name)
            return False
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._client_id or not self._client_secret:
            logger.warning("[%s] DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required", self.name)
            return False

        # Acquire scoped lock to prevent duplicate connections with same credential
        if not self._acquire_platform_lock("dingtalk-client-id", self._client_id, "DingTalk client_id"):
            return False

        try:
            self._http_client = httpx.AsyncClient(timeout=30.0)

            credential = dingtalk_stream.Credential(self._client_id, self._client_secret)
            self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)

            # Capture the current event loop for cross-thread dispatch
            loop = asyncio.get_running_loop()
            handler = _IncomingHandler(self, loop)
            self._stream_client.register_callback_handler(
                dingtalk_stream.ChatbotMessage.TOPIC, handler
            )

            self._stream_task = asyncio.create_task(self._run_stream())
            self._mark_connected()
            logger.info("[%s] Connected via Stream Mode", self.name)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            self._release_platform_lock()
            return False

    async def _run_stream(self) -> None:
        """Run the blocking stream client with auto-reconnection."""
        backoff_idx = 0
        while self._running:
            try:
                logger.debug("[%s] Starting stream client...", self.name)
                await self._start_stream_client()
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Stream client error: %s", self.name, e)

            if not self._running:
                return

            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    async def _start_stream_client(self) -> None:
        """Start the SDK client, supporting both sync and async SDK variants."""
        start = getattr(self._stream_client, "start", None)
        if start is None:
            raise RuntimeError("DingTalk stream client missing start()")

        if inspect.iscoroutinefunction(start):
            await start()
            return

        result = await asyncio.to_thread(start)
        if inspect.isawaitable(result):
            await result

    async def disconnect(self) -> None:
        """Disconnect from DingTalk."""
        self._running = False
        self._mark_disconnected()

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._stream_client = None
        self._session_webhooks.clear()
        self._session_context.clear()
        self._dedup.clear()
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound message processing -----------------------------------------

    async def _on_message(self, message: "ChatbotMessage") -> None:
        """Process an incoming DingTalk chatbot message."""
        msg_id = getattr(message, "message_id", None) or uuid.uuid4().hex

        if self._should_ignore_message(message):
            logger.debug("[%s] Ignoring self/echo DingTalk message %s", self.name, self._redact(msg_id))
            return

        if self._dedup.is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return

        text = self._extract_text(message)
        raw_type = str(getattr(message, "message_type", "") or "").strip().lower()

        # Chat context
        conversation_id = getattr(message, "conversation_id", "") or ""
        conversation_type = getattr(message, "conversation_type", "1")
        is_group = str(conversation_type) == "2"
        sender_id = getattr(message, "sender_id", "") or ""
        sender_nick = getattr(message, "sender_nick", "") or sender_id
        sender_staff_id = getattr(message, "sender_staff_id", "") or ""
        robot_code = self._extract_message_robot_code(message)
        if not sender_staff_id:
            raw = getattr(message, "_raw_data", None) or {}
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            if isinstance(raw, dict):
                sender_staff_id = str(raw.get("senderStaffId") or raw.get("sender_staff_id") or "")

        # Strictly separate group vs DM chat_id to prevent cross-talk
        if is_group:
            if not conversation_id:
                logger.warning("[%s] Group message missing conversation_id, skipping. sender=%s",
                               self.name, self._redact(sender_id))
                return
            chat_id, chat_type = conversation_id, "group"
        else:
            chat_id = sender_id
            if not chat_id:
                logger.warning("[%s] DM message missing sender_id, skipping", self.name)
                return
            chat_type = "dm"

        # Store session webhook for reply routing (validate origin to prevent SSRF)
        session_webhook = getattr(message, "session_webhook", None) or ""
        if session_webhook and chat_id and _DINGTALK_WEBHOOK_RE.match(session_webhook):
            if len(self._session_webhooks) >= _SESSION_WEBHOOKS_MAX:
                # Evict oldest entry to cap memory growth
                try:
                    self._session_webhooks.pop(next(iter(self._session_webhooks)))
                except StopIteration:
                    pass
            self._session_webhooks[chat_id] = session_webhook

        # Capture session context for media sending
        if chat_id:
            if len(self._session_context) >= _SESSION_CONTEXT_MAX and chat_id not in self._session_context:
                try:
                    self._session_context.pop(next(iter(self._session_context)))
                except StopIteration:
                    pass
            context = self._session_context.setdefault(chat_id, {})
            context.update({
                "chat_type": chat_type,
                "conversation_id": conversation_id,
                "user_id": sender_id,
                "user_name": sender_nick,
                "sender_staff_id": sender_staff_id,
                "robot_code": robot_code,
                "session_webhook": session_webhook,
                "chat_name": getattr(message, "conversation_title", None) or "",
            })

        source = self.build_source(
            chat_id=chat_id,
            chat_name=getattr(message, "conversation_title", None),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_nick,
            user_id_alt=sender_staff_id if sender_staff_id else None,
        )

        # Parse timestamp
        create_at = getattr(message, "create_at", None)
        try:
            timestamp = datetime.fromtimestamp(int(create_at) / 1000, tz=timezone.utc) if create_at else datetime.now(tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        # Determine message type and download inbound media
        message_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []

        try:
            if (
                raw_type == "picture"
                or (raw_type == "richtext" and self._has_rich_text_media(message, "picture"))
                or self._reply_message_type(message) == "picture"
            ):
                cached_path, media_type = await self._download_inbound_picture(message)
                media_urls, media_types, message_type = [cached_path], [media_type], MessageType.PHOTO
            elif (
                raw_type == "file"
                or (raw_type == "richtext" and self._has_rich_text_media(message, "file"))
                or self._reply_message_type(message) == "file"
            ):
                cached_path, media_type = await self._download_inbound_file(message)
                media_urls, media_types = [cached_path], [media_type]
                message_type = MessageType.PHOTO if media_type.startswith("image/") else MessageType.DOCUMENT
                if message_type == MessageType.DOCUMENT and media_type.startswith("text/"):
                    try:
                        injected_text = Path(cached_path).read_text(encoding="utf-8")
                        text = f"{text}\n{injected_text}".strip()
                    except Exception:
                        pass
            elif raw_type == "audio" or self._reply_message_type(message) == "audio":
                cached_path, media_type = await self._download_inbound_audio(message)
                media_urls, media_types, message_type = [cached_path], [media_type], MessageType.AUDIO
            elif raw_type == "video" or self._reply_message_type(message) == "video":
                cached_path, media_type = await self._download_inbound_video(message)
                media_urls, media_types, message_type = [cached_path], [media_type], MessageType.VIDEO
        except Exception as exc:
            logger.warning("[%s] Failed to download inbound media (%s): %s", self.name, raw_type, exc)

        if not text and not media_urls:
            logger.debug("[%s] Empty message, skipping", self.name)
            return

        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            message_id=msg_id,
            raw_message=message,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )

        logger.debug("[%s] Message from %s in %s: %s",
                      self.name, sender_nick, chat_id[:20] if chat_id else "?", text[:50])
        await self.handle_message(event)

    @staticmethod
    def _extract_text(message: "ChatbotMessage") -> str:
        """Extract plain text from a DingTalk chatbot message."""
        text = getattr(message, "text", None) or ""
        if isinstance(text, dict):
            content = text.get("content", "").strip()
        elif hasattr(text, "content"):
            content = str(getattr(text, "content", "") or "").strip()
        else:
            content = str(text).strip()

        # Fall back to rich text if present
        if not content:
            rich_text = DingTalkAdapter._rich_text_items(message)
            if rich_text:
                parts = [item["text"] for item in rich_text
                         if isinstance(item, dict) and item.get("text")]
                content = " ".join(parts).strip()

        # Fall back to audio recognition text
        if not content:
            recognition = str(DingTalkAdapter._message_content(message).get("recognition") or "").strip()
            if recognition:
                content = recognition

        return content

    @classmethod
    def _should_ignore_message(cls, message: "ChatbotMessage") -> bool:
        """Return True for self-messages and echo/sync messages."""
        sender_id = cls._get_attr_variants(message, "sender_id", "senderId")
        chatbot_user_id = cls._get_attr_variants(message, "chatbot_user_id", "chatbotUserId")
        if sender_id and chatbot_user_id and str(sender_id) == str(chatbot_user_id):
            return True
        if bool(getattr(message, "is_echo", False) or getattr(message, "isEcho", False)):
            return True
        source = str(cls._get_attr_variants(message, "message_source", "messageSource") or "").strip().lower()
        return source in {"bot", "echo", "sync"}

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _mask_secret(value: str) -> str:
        """Mask a secret string, keeping only first/last 4 chars."""
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}***{value[-4:]}"

    def _redact(self, value: Any) -> str:
        """Redact secrets from a string for safe logging."""
        text = str(value)
        text = _QUERY_SECRET_RE.sub(lambda match: f"{match.group('key')}={self._mask_secret(match.group(2))}", text)
        return _LONG_SECRET_RE.sub(lambda match: self._mask_secret(match.group(1)), text)

    @staticmethod
    def _get_attr_variants(obj: Any, *names: str) -> Any:
        """Try multiple attribute names, return the first truthy value."""
        for name in names:
            val = getattr(obj, name, None)
            if val is not None:
                return val
        return None

    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, Any]:
        """Coerce a value into a dict, handling JSON strings and objects."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        if value is None:
            return {}
        if hasattr(value, "__dict__"):
            return {
                key: field_value
                for key, field_value in vars(value).items()
                if not key.startswith("_")
            }
        return {}

    @classmethod
    def _extract_message_robot_code(cls, message: "ChatbotMessage") -> str:
        """Extract robotCode from normalized SDK fields or preserved raw payload."""
        direct = str(cls._get_attr_variants(message, "robot_code", "robotCode") or "").strip()
        if direct:
            return direct
        for payload in (
            getattr(message, "_raw_data", None),
            getattr(message, "extensions", None),
        ):
            mapped = cls._coerce_mapping(payload)
            value = mapped.get("robotCode") or mapped.get("robot_code")
            if value:
                return str(value).strip()
        return ""

    # -- Message parsing ----------------------------------------------------

    @classmethod
    def _text_payload(cls, message: "ChatbotMessage") -> Dict[str, Any]:
        """Return the text field of a message as a dict."""
        return cls._coerce_mapping(getattr(message, "text", None))

    @staticmethod
    def _message_content(message: "ChatbotMessage") -> Dict[str, Any]:
        """Return the content/extensions dict of a message."""
        content = getattr(message, "content", None)
        if isinstance(content, dict):
            return content
        extensions = getattr(message, "extensions", None)
        if isinstance(extensions, dict):
            ext_content = extensions.get("content")
            if isinstance(ext_content, dict):
                return ext_content
        return {}

    @staticmethod
    def _rich_text_items(message: "ChatbotMessage") -> List[Dict[str, Any]]:
        """Extract rich text items from a message."""
        rich_text_content = getattr(message, "rich_text_content", None)
        rich_text_list = getattr(rich_text_content, "rich_text_list", None)
        if isinstance(rich_text_list, list):
            return [item for item in rich_text_list if isinstance(item, dict)]
        rich_text = getattr(message, "rich_text", None)
        if isinstance(rich_text, list):
            return [item for item in rich_text if isinstance(item, dict)]
        content = DingTalkAdapter._message_content(message)
        content_rich_text = content.get("richText") or content.get("richTextList") or []
        if isinstance(content_rich_text, list):
            return [item for item in content_rich_text if isinstance(item, dict)]
        return []

    @staticmethod
    def _rich_text_item_kind(item: Dict[str, Any]) -> str:
        """Determine the media kind of a rich text item."""
        item_type = str(item.get("type") or "").strip().lower()
        if item.get("pictureUrl"):
            return "picture"
        if not item.get("downloadCode"):
            return ""
        if item_type in {"", "picture", "image"}:
            return "picture"
        if item_type == "file":
            return "file"
        if item_type in {"audio", "voice"}:
            return "audio"
        if item_type == "video":
            return "video"
        return ""

    @classmethod
    def _find_rich_text_item(cls, message: "ChatbotMessage", kind: str) -> Dict[str, Any]:
        """Find the first rich text item matching *kind*."""
        for item in cls._rich_text_items(message):
            if cls._rich_text_item_kind(item) == kind:
                return item
        return {}

    @classmethod
    def _has_rich_text_media(cls, message: "ChatbotMessage", kind: str) -> bool:
        """Return True if the message contains a rich text item of *kind*."""
        return bool(cls._find_rich_text_item(message, kind))

    @classmethod
    def _reply_payload(cls, message: "ChatbotMessage") -> Dict[str, Any]:
        """Extract the replied message payload."""
        text_payload = cls._text_payload(message)
        reply = cls._coerce_mapping(text_payload.get("repliedMsg"))
        if reply:
            return reply
        return cls._coerce_mapping(cls._message_content(message).get("repliedMsg"))

    @classmethod
    def _reply_message_type(cls, message: "ChatbotMessage") -> str:
        """Return the message type of a replied message."""
        reply = cls._reply_payload(message)
        return str(reply.get("msgType") or reply.get("messageType") or "").strip().lower()

    @classmethod
    def _reply_content(cls, message: "ChatbotMessage") -> Dict[str, Any]:
        """Return the content dict of a replied message."""
        return cls._coerce_mapping(cls._reply_payload(message).get("content"))

    @classmethod
    def _reply_rich_text_item(cls, message: "ChatbotMessage", kind: str) -> Dict[str, Any]:
        """Find a rich text item of *kind* inside a replied message."""
        reply_content = cls._reply_content(message)
        rich_text = reply_content.get("richText") or reply_content.get("richTextList") or []
        if not isinstance(rich_text, list):
            return {}
        for item in rich_text:
            if isinstance(item, dict) and cls._rich_text_item_kind(item) == kind:
                return item
        return {}

    # -- Media download -----------------------------------------------------

    async def _download_inbound_picture(self, message: "ChatbotMessage") -> tuple:
        """Download an inbound picture message and cache it locally."""
        content = self._message_content(message)
        rich_text_item = self._find_rich_text_item(message, "picture")
        reply_type = self._reply_message_type(message)
        reply_content = self._reply_content(message)
        reply_rich_text_item = self._reply_rich_text_item(message, "picture")
        picture_url = (
            str(content.get("pictureUrl") or "")
            or str(rich_text_item.get("pictureUrl") or "")
            or (str(reply_content.get("pictureUrl") or "") if reply_type == "picture" else "")
            or str(reply_rich_text_item.get("pictureUrl") or "")
        )
        download_code = (
            self._get_attr_variants(message, "picture_download_code", "pictureDownloadCode")
            or getattr(getattr(message, "image_content", None), "download_code", None)
            or content.get("pictureDownloadCode")
            or self._get_attr_variants(message, "download_code", "downloadCode")
            or content.get("downloadCode")
            or rich_text_item.get("downloadCode")
            or (reply_content.get("downloadCode") if reply_type == "picture" else "")
            or reply_rich_text_item.get("downloadCode")
            or ""
        )
        if picture_url:
            response = await self._download_remote_response(str(picture_url))
            fallback = mimetypes.guess_type(urlsplit(str(picture_url)).path)[0] or "image/jpeg"
            media_type = self._response_media_type(response, fallback=fallback)
            ext = mimetypes.guess_extension(media_type) or ".jpg"
            return cache_image_from_bytes(response.content, ext=ext), media_type

        robot_code = self._get_attr_variants(message, "robot_code", "robotCode") or self._robot_code or ""
        if not download_code or not robot_code:
            raise RuntimeError("Missing DingTalk picture downloadCode or robotCode")

        response, media_type = await self._download_by_code(download_code, robot_code, "image/jpeg")
        ext = mimetypes.guess_extension(media_type) or ".jpg"
        return cache_image_from_bytes(response.content, ext=ext), media_type

    async def _download_inbound_file(self, message: "ChatbotMessage") -> tuple:
        """Download an inbound file message and cache it locally."""
        content = self._message_content(message)
        rich_text_item = self._find_rich_text_item(message, "file")
        reply_type = self._reply_message_type(message)
        reply_content = self._reply_content(message)
        reply_rich_text_item = self._reply_rich_text_item(message, "file")
        file_name = str(
            self._get_attr_variants(message, "file_name", "fileName")
            or content.get("fileName")
            or rich_text_item.get("fileName")
            or (reply_content.get("fileName") if reply_type == "file" else "")
            or reply_rich_text_item.get("fileName")
            or "attachment"
        )
        download_code = (
            self._get_attr_variants(message, "download_code", "downloadCode")
            or content.get("downloadCode")
            or rich_text_item.get("downloadCode")
            or (reply_content.get("downloadCode") if reply_type == "file" else "")
            or reply_rich_text_item.get("downloadCode")
            or ""
        )
        robot_code = self._get_attr_variants(message, "robot_code", "robotCode") or self._robot_code or ""
        if not download_code or not robot_code:
            raise RuntimeError("Missing DingTalk file downloadCode or robotCode")

        fallback = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        response, media_type = await self._download_by_code(download_code, robot_code, fallback)
        return cache_document_from_bytes(response.content, file_name), media_type

    async def _download_by_code(self, download_code: str, robot_code: str, fallback_mime: str) -> tuple:
        """Download file via downloadCode + robotCode, return (response, media_type)."""
        download_url = await self._query_message_file_download_url(
            download_code=str(download_code), robot_code=str(robot_code),
        )
        response = await self._download_remote_response(download_url)
        return response, self._response_media_type(response, fallback=fallback_mime)

    async def _download_inbound_media(
        self, message: "ChatbotMessage", media_kind: str, fallback_mime: str, default_ext: str,
    ) -> tuple:
        """Generic download for audio/video inbound media."""
        content = self._message_content(message)
        reply_type = self._reply_message_type(message)
        reply_content = self._reply_content(message)
        reply_rich_text_item = self._reply_rich_text_item(message, media_kind)
        download_code = (
            self._get_attr_variants(message, "download_code", "downloadCode")
            or content.get("downloadCode")
            or (reply_content.get("downloadCode") if reply_type == media_kind else "")
            or reply_rich_text_item.get("downloadCode")
            or ""
        )
        robot_code = self._get_attr_variants(message, "robot_code", "robotCode") or self._robot_code or ""
        if not download_code or not robot_code:
            raise RuntimeError(f"Missing DingTalk {media_kind} downloadCode or robotCode")

        response, media_type = await self._download_by_code(download_code, robot_code, fallback_mime)
        suffix = mimetypes.guess_extension(media_type) or default_ext
        if media_kind == "audio":
            return cache_audio_from_bytes(response.content, ext=suffix), media_type
        return cache_document_from_bytes(response.content, f"{media_kind}{suffix}"), media_type

    async def _download_inbound_audio(self, message: "ChatbotMessage") -> tuple:
        """Download inbound audio message."""
        return await self._download_inbound_media(message, "audio", "audio/ogg", ".ogg")

    async def _download_inbound_video(self, message: "ChatbotMessage") -> tuple:
        """Download inbound video message."""
        return await self._download_inbound_media(message, "video", "video/mp4", ".mp4")

    @staticmethod
    def _response_media_type(response: Any, *, fallback: str) -> str:
        """Determine the media type from a response, falling back as needed."""
        headers = getattr(response, "headers", {}) or {}
        media_type = headers.get("content-type") or headers.get("Content-Type") or fallback
        normalized = str(media_type).split(";")[0].strip() or fallback
        return fallback if (normalized == "application/octet-stream" and fallback) else normalized

    async def _download_remote_response(self, url: str) -> Any:
        """Download a remote URL and return the response."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        response = await self._http_client.get(url, timeout=30.0)
        if response.status_code >= 400:
            raise RuntimeError(f"DingTalk media download failed: HTTP {response.status_code}")
        return response

    async def _query_message_file_download_url(self, *, download_code: str, robot_code: str) -> str:
        """Query the download URL for a message file."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        access_token = await self._get_access_token()
        response = await self._http_client.post(
            "https://api.dingtalk.com/v1.0/robot/messageFiles/download",
            headers={"x-acs-dingtalk-access-token": access_token},
            json={"downloadCode": download_code, "robotCode": robot_code},
            timeout=15.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DingTalk message file download URL request failed: HTTP {response.status_code}")
        payload = response.json()
        download_url = str(payload.get("downloadUrl") or "")
        if not download_url:
            raise RuntimeError("DingTalk message file download response missing downloadUrl")
        return download_url

    # -- OpenAPI ------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Fetch or return a cached DingTalk access token."""
        now = time.time()
        if self._access_token and now < self._access_token_expires_at:
            return self._access_token
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        response = await self._http_client.post(
            "https://api.dingtalk.com/v1.0/oauth2/accessToken",
            json={"appKey": self._client_id, "appSecret": self._client_secret},
            timeout=15.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DingTalk accessToken request failed: HTTP {response.status_code}")
        payload = response.json()
        access_token = str(payload.get("accessToken") or "")
        expire_in = int(payload.get("expireIn") or 0)
        if not access_token:
            raise RuntimeError("DingTalk accessToken response missing accessToken")
        self._access_token = access_token
        self._access_token_expires_at = now + max(expire_in - 60, 0)
        return access_token

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a markdown reply via DingTalk session webhook."""
        metadata = metadata or {}

        context_webhook = ""
        if chat_id:
            context_webhook = str(
                (self._session_context.get(chat_id, {}) or {}).get("session_webhook") or ""
            )

        session_webhook = (
            metadata.get("session_webhook")
            or self._session_webhooks.get(chat_id)
            or context_webhook
        )
        if not session_webhook:
            return SendResult(success=False,
                              error="No session_webhook available. Reply must follow an incoming message.")

        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": "Hermes", "text": content[:self.MAX_MESSAGE_LENGTH]},
        }

        try:
            resp = await self._http_client.post(session_webhook, json=payload, timeout=15.0)
            if resp.status_code < 300:
                return SendResult(success=True, message_id=None)
            body = resp.text
            logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body[:200]}")
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sending message to DingTalk")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """DingTalk does not support typing indicators."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a DingTalk conversation."""
        context = self._session_context.get(chat_id, {})
        return {
            "chat_id": chat_id,
            "name": context.get("chat_name") or chat_id,
            "type": context.get("chat_type") or ("group" if "group" in chat_id.lower() else "dm"),
        }

    # -- Outbound media helpers ---------------------------------------------

    def _merged_context(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge stored session context with per-call metadata."""
        stored = dict(self._session_context.get(chat_id, {}))
        context = dict(stored)
        incoming = dict(metadata or {})

        # Never let ad-hoc metadata silently downgrade a known group session
        # into a DM target.  Preserve the routing fields captured from the live
        # inbound group message so replies stay in the originating group.
        if stored.get("chat_type") == "group":
            incoming.pop("chat_type", None)
            incoming.pop("conversation_id", None)
            incoming.pop("session_webhook", None)
            incoming.pop("sender_staff_id", None)
            incoming.pop("user_id", None)

        context.update(incoming)
        return context

    def _caption_metadata(self, chat_id: str, context: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the webhook metadata used for best-effort media captions."""
        session_webhook = context.get("session_webhook") or self._session_webhooks.get(chat_id)
        if session_webhook:
            return {"session_webhook": session_webhook}
        return metadata

    @staticmethod
    def _validate_local_file_path(file_path: str) -> Optional[str]:
        """Return a user-facing validation error for an outbound local path."""
        candidate = str(file_path or "").strip()
        if not candidate:
            return "File path is empty"
        if candidate in _PLACEHOLDER_MEDIA_PATHS:
            return f"Placeholder file path: {candidate}"
        if any(part in _PLACEHOLDER_PATH_PARTS for part in Path(candidate).parts):
            return f"Placeholder file path: {candidate}"
        if not os.path.exists(candidate):
            return f"File not found: {candidate}"
        if not os.path.isfile(candidate):
            return f"Not a file: {candidate}"
        return None

    def _resolve_outbound_robot_identity(self, chat_id: str, context: Dict[str, Any]) -> tuple[str, str]:
        """Return the effective outbound robotCode and where it came from."""
        stored = self._session_context.get(chat_id, {})
        for source, value in (
            ("session_or_metadata", context.get("robot_code")),
            ("stored_session", stored.get("robot_code")),
            ("configured", self._configured_robot_code),
        ):
            text = str(value or "").strip()
            if text:
                return text, source

        fallback = str(self._robot_code or "").strip()
        if fallback and fallback == str(self._client_id or "").strip():
            return fallback, "client_id_fallback"
        if fallback:
            return fallback, "adapter_default"
        return "", "missing"

    @staticmethod
    def _extract_session_token(session_webhook: str) -> str:
        """Extract the current DM session token from a DingTalk session webhook."""
        query = parse_qs(urlsplit(str(session_webhook or "")).query)
        for key in ("token", "session"):
            values = query.get(key) or []
            if values and str(values[0]).strip():
                return str(values[0]).strip()
        return ""

    async def _send_media_caption(
        self, chat_id: str, caption: Optional[str], *,
        reply_to: Optional[str], context: Dict[str, Any], metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Best-effort preface text for media messages."""
        if not caption:
            return
        result = await self.send(
            chat_id,
            caption,
            reply_to=reply_to,
            metadata=self._caption_metadata(chat_id, context, metadata),
        )
        if not result.success:
            logger.warning("[%s] Failed to send media caption for %s: %s",
                           self.name, self._redact(chat_id), result.error)

    async def _send_error_notice(
        self, chat_id: str, context: Dict[str, Any], error_text: str,
        reply_to: Optional[str] = None, prefix: str = "Operation failed.",
    ) -> None:
        """Best-effort user-visible error reply via the session webhook."""
        session_webhook = context.get("session_webhook") or self._session_webhooks.get(chat_id)
        if not session_webhook:
            return
        detail = re.sub(r"\s+", " ", str(error_text or "Unknown DingTalk error").strip())
        if len(detail) > 900:
            detail = detail[:897] + "..."
        try:
            await self.send(chat_id, f"{prefix}\nError: {detail}",
                            reply_to=reply_to, metadata={"session_webhook": session_webhook})
        except Exception as exc:
            logger.warning("[%s] Failed to send error notice: %s", self.name, exc)

    async def _preflight_local_media_path(
        self,
        chat_id: str,
        local_path: str,
        context: Dict[str, Any],
        *,
        kind_label: str,
        failure_prefix: str,
        reply_to: Optional[str] = None,
    ) -> Optional[str]:
        """Validate a local media path and reject group-to-DM downgrades."""
        stored = self._session_context.get(chat_id, {})
        path_error = self._validate_local_file_path(local_path)
        if path_error:
            await self._send_error_notice(chat_id, context, path_error, reply_to=reply_to, prefix=failure_prefix)
            return path_error
        if stored.get("chat_type") == "group" and context.get("chat_type") not in (None, "group"):
            error = f"Refusing to downgrade DingTalk group {kind_label} reply to DM target"
            await self._send_error_notice(chat_id, context, error, reply_to=reply_to, prefix=failure_prefix)
            return error
        return None

    async def _prepare_local_media_send(
        self,
        chat_id: str,
        local_path: str,
        *,
        kind_label: str,
        failure_prefix: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        display_name: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str, Optional[SendResult]]:
        """Build merged context and reject invalid local media sends early."""
        context = self._merged_context(chat_id, metadata)
        resolved_name = display_name or Path(local_path).name
        preflight_error = await self._preflight_local_media_path(
            chat_id,
            local_path,
            context,
            kind_label=kind_label,
            failure_prefix=failure_prefix,
            reply_to=reply_to,
        )
        if preflight_error:
            return context, resolved_name, SendResult(success=False, error=preflight_error)

        try:
            self._resolve_robot_target(chat_id, context, kind_label=kind_label)
        except RuntimeError as exc:
            error = str(exc)
            await self._send_error_notice(chat_id, context, error, reply_to=reply_to, prefix=failure_prefix)
            return context, resolved_name, SendResult(success=False, error=error)

        return context, resolved_name, None

    async def _finalize_local_media_send(
        self,
        chat_id: str,
        context: Dict[str, Any],
        result: SendResult,
        *,
        kind_label: str,
        display_name: str,
        reply_to: Optional[str],
        failure_prefix: str,
    ) -> SendResult:
        """Emit the shared success log or user-facing failure notice."""
        if not result.success:
            await self._send_error_notice(
                chat_id,
                context,
                result.error or f"Unknown {kind_label} send error",
                reply_to=reply_to,
                prefix=failure_prefix,
            )
            return result

        self._log_media_send_success(
            kind_label=kind_label,
            chat_id=chat_id,
            display_name=display_name,
            message_id=result.message_id,
        )
        return result

    async def _send_uploaded_local_media(
        self,
        chat_id: str,
        local_path: str,
        *,
        kind_label: str,
        failure_prefix: str,
        upload_media_type: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        operation_name: str,
        caption: Optional[str] = None,
        display_name: Optional[str] = None,
        sender: Callable[[str, str], Awaitable[SendResult]],
        prepared_context: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file, optionally send a caption, then send the native robot message."""
        if prepared_context is None:
            context, resolved_name, early_result = await self._prepare_local_media_send(
                chat_id,
                local_path,
                kind_label=kind_label,
                failure_prefix=failure_prefix,
                reply_to=reply_to,
                metadata=metadata,
                display_name=display_name,
            )
            if early_result:
                return early_result
        else:
            context = prepared_context
            resolved_name = display_name or Path(local_path).name

        try:
            media_id = await self._upload_media(local_path, media_type=upload_media_type)
            await self._send_media_caption(
                chat_id,
                caption,
                reply_to=reply_to,
                context=context,
                metadata=metadata,
            )
            result = await sender(media_id, resolved_name)
            return await self._finalize_local_media_send(
                chat_id,
                context,
                result,
                kind_label=kind_label,
                display_name=resolved_name,
                reply_to=reply_to,
                failure_prefix=failure_prefix,
            )
        except Exception as exc:
            error = str(exc)
            logger.error("[%s] %s error: %s", self.name, operation_name, error)
            await self._send_error_notice(chat_id, context, error, reply_to=reply_to, prefix=failure_prefix)
            return SendResult(success=False, error=error)

    def _log_media_send_success(
        self,
        *,
        kind_label: str,
        chat_id: str,
        display_name: str,
        message_id: Optional[str],
    ) -> None:
        """Emit a consistent success log for local media sends."""
        logger.info(
            "[%s] %s sent to DingTalk: chat=%s file=%s message_id=%s",
            self.name,
            kind_label.title(),
            self._redact(chat_id),
            self._redact(display_name),
            self._redact(message_id or "-"),
        )

    @staticmethod
    def _native_media_business_error(resp_data: Any) -> str:
        """Return a business-layer error string from a 2xx native media response."""
        if not isinstance(resp_data, dict):
            return ""

        process_query_key = str(resp_data.get("processQueryKey") or "").strip()
        if process_query_key:
            return ""

        success = resp_data.get("success")
        message = str(resp_data.get("message") or resp_data.get("errmsg") or "").strip()
        if success is False:
            return message or "success=false"

        errcode = resp_data.get("errcode")
        if errcode not in (None, "", 0, "0"):
            return f"errcode={errcode}" + (f" message={message}" if message else "")

        code = resp_data.get("code")
        normalized_code = str(code or "").strip().lower()
        if normalized_code and normalized_code not in {"0", "ok", "success"}:
            return f"code={code}" + (f" message={message}" if message else "")

        return ""

    def _resolve_robot_target(
        self, chat_id: str, context: Dict[str, Any], *, kind_label: str,
    ) -> tuple:
        """Resolve DM/group robot destination for a media message."""
        stored = self._session_context.get(chat_id, {})
        chat_type = str(
            context.get("chat_type")
            or stored.get("chat_type")
            or "dm"
        ).strip().lower()
        conversation_id = str(
            context.get("conversation_id")
            or stored.get("conversation_id")
            or ""
        ).strip()
        session_webhook = str(
            context.get("session_webhook")
            or stored.get("session_webhook")
            or ""
        ).strip()

        if chat_type == "group":
            conversation_id = conversation_id or str(chat_id or "").strip()
            if not conversation_id:
                raise RuntimeError(f"Missing conversation ID for group {kind_label} message")
            return (
                "https://api.dingtalk.com/v1.0/robot/groupMessages/send",
                {"openConversationId": conversation_id},
            )

        user_id = str(
            context.get("sender_staff_id")
            or stored.get("sender_staff_id")
            or context.get("user_id")
            or stored.get("user_id")
            or ""
        ).strip()
        if user_id:
            return (
                "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend",
                {"userIds": [user_id]},
            )

        if conversation_id:
            session_token = self._extract_session_token(session_webhook)
            if session_token:
                return (
                    "https://api.dingtalk.com/v1.0/robot/privateChatMessages/send",
                    {
                        "openConversationId": conversation_id,
                        "token": session_token,
                    },
                )
            raise RuntimeError(f"Missing session token for DM {kind_label} message")

        raise RuntimeError(f"Missing user ID for DM {kind_label} message")

    async def _send_robot_media_message(
        self,
        chat_id: str,
        *,
        msg_key: str,
        msg_param: Dict[str, Any],
        kind_label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native robot media message through the DingTalk robot API."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        context = self._merged_context(chat_id, metadata)
        try:
            endpoint, target = self._resolve_robot_target(chat_id, context, kind_label=kind_label)
        except RuntimeError as exc:
            return SendResult(success=False, error=str(exc))
        robot_code, robot_code_source = self._resolve_outbound_robot_identity(chat_id, context)
        if not robot_code:
            return SendResult(success=False, error="Missing robotCode for DingTalk native media send")

        access_token = await self._get_access_token()
        body: Dict[str, Any] = {
            "robotCode": robot_code,
            "msgKey": msg_key,
            "msgParam": json.dumps(msg_param),
            **target,
        }

        try:
            resp = await self._http_client.post(
                endpoint,
                headers={
                    "x-acs-dingtalk-access-token": access_token,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=15.0,
            )
            if resp.status_code < 400:
                resp_data = resp.json()
                business_error = self._native_media_business_error(resp_data)
                if business_error:
                    body_text = (
                        f"{json.dumps(resp_data, ensure_ascii=False)} "
                        f"(endpoint={urlsplit(endpoint).path}) "
                        f"(robotCodeSource={robot_code_source})"
                    )
                    logger.warning("[%s] %s send failed business response: %s",
                                   self.name, kind_label.title(), body_text[:200])
                    return SendResult(success=False, error=body_text[:200])
                logger.debug("[%s] %s message sent: %s", self.name, kind_label.title(), resp_data)
                return SendResult(
                    success=True,
                    message_id=str(resp_data.get("processQueryKey") or uuid.uuid4().hex[:12]),
                )
            body_text = (
                f"{resp.text} "
                f"(endpoint={urlsplit(endpoint).path}) "
                f"(robotCodeSource={robot_code_source})"
            )
            logger.warning("[%s] %s send failed HTTP %d: %s",
                           self.name, kind_label.title(), resp.status_code, body_text[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body_text[:200]}")
        except Exception as exc:
            logger.error("[%s] %s send error: %s", self.name, kind_label.title(), exc)
            return SendResult(success=False, error=str(exc))

    # -- Media upload -------------------------------------------------------

    async def _upload_media(self, file_path: str, media_type: str = "image") -> str:
        """Upload a file to DingTalk via the old oapi media/upload endpoint."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        if not os.path.exists(file_path):
            raise RuntimeError(f"File not found: {file_path}")
        access_token = await self._get_access_token()
        file_name = Path(file_path).name
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        with open(file_path, "rb") as f:
            files = {"media": (file_name, f, content_type)}
            response = await self._http_client.post(
                "https://oapi.dingtalk.com/media/upload",
                params={"access_token": access_token, "type": media_type},
                files=files,
                timeout=60.0,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"DingTalk media upload failed: HTTP {response.status_code}")
        payload = response.json()
        media_id = str(payload.get("media_id") or "")
        if not media_id:
            raise RuntimeError(f"DingTalk media upload response missing media_id: {payload}")
        logger.debug("[%s] Uploaded media %s -> %s", self.name, file_name, media_id)
        return media_id

    # -- Native message senders ---------------------------------------------

    async def _send_image_message(
        self, chat_id: str, media_id: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image message using the robot messaging API."""
        return await self._send_robot_media_message(
            chat_id,
            msg_key="sampleImageMsg",
            msg_param={"photoURL": media_id},
            kind_label="image",
            metadata=metadata,
        )

    async def _send_file_message(
        self, chat_id: str, media_id: str, file_name: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native DingTalk file message using the robot messaging API."""
        file_type = Path(file_name).suffix.lstrip(".").lower()
        if not file_type:
            return SendResult(success=False, error="Missing file extension for DingTalk file message")
        return await self._send_robot_media_message(
            chat_id,
            msg_key="sampleFile",
            msg_param={"mediaId": media_id, "fileName": file_name, "fileType": file_type},
            kind_label="file",
            metadata=metadata,
        )

    async def _send_audio_message(
        self, chat_id: str, media_id: str, duration_ms: int, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native DingTalk voice message using the robot messaging API."""
        return await self._send_robot_media_message(
            chat_id,
            msg_key="sampleAudio",
            msg_param={"mediaId": media_id, "duration": str(duration_ms)},
            kind_label="audio",
            metadata=metadata,
        )

    async def _send_video_message(
        self,
        chat_id: str,
        video_media_id: str,
        cover_media_id: str,
        duration_seconds: int,
        video_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native DingTalk video message using the robot messaging API."""
        return await self._send_robot_media_message(
            chat_id,
            msg_key="sampleVideo",
            msg_param={
                "duration": str(duration_seconds),
                "videoMediaId": video_media_id,
                "videoType": video_type,
                "picMediaId": cover_media_id,
            },
            kind_label="video",
            metadata=metadata,
        )

    # -- Public media send methods ------------------------------------------

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a local image file via media upload and native image message."""
        return await self._send_uploaded_local_media(
            chat_id,
            image_path,
            kind_label="image",
            failure_prefix="Image send failed.",
            upload_media_type="image",
            reply_to=reply_to,
            metadata=metadata,
            operation_name="send_image_file",
            caption=caption,
            sender=lambda media_id, _display_name: self._send_image_message(
                chat_id,
                media_id,
                metadata=metadata,
            ),
        )

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download a remote image and send it as a native image message."""
        context = self._merged_context(chat_id, metadata)
        try:
            response = await self._download_remote_response(image_url)
            media_type = self._response_media_type(response, fallback="image/jpeg")
            ext = mimetypes.guess_extension(media_type) or ".jpg"
            image_path = cache_image_from_bytes(response.content, ext=ext)
            return await self.send_image_file(
                chat_id, image_path, caption=caption, reply_to=reply_to, metadata=metadata,
            )
        except Exception as exc:
            error = str(exc)
            logger.error("[%s] send_image error: %s", self.name, error)
            await self._send_error_notice(chat_id, context, error, reply_to=reply_to, prefix="Image send failed.")
            return SendResult(success=False, error=error)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a document as a native DingTalk file message."""
        return await self._send_uploaded_local_media(
            chat_id,
            file_path,
            kind_label="file",
            failure_prefix="File send failed.",
            upload_media_type="file",
            reply_to=reply_to,
            metadata=metadata,
            operation_name="send_document",
            caption=caption,
            display_name=file_name,
            sender=lambda media_id, display_name: self._send_file_message(
                chat_id,
                media_id,
                display_name,
                metadata=metadata,
            ),
        )

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a voice message via native DingTalk robot media messaging.

        If ``duration_ms`` is provided via kwargs and the file is ogg/amr,
        sends a native ``sampleAudio`` message.  Otherwise falls back to
        ``sampleFile`` (file message) so the audio still reaches the user.
        """
        duration_ms = kwargs.get("duration_ms")
        audio_type = Path(audio_path).suffix.lstrip(".").lower()
        if duration_ms is not None and audio_type in {"ogg", "amr"}:
            return await self._send_uploaded_local_media(
                chat_id,
                audio_path,
                kind_label="audio",
                failure_prefix="Voice send failed.",
                upload_media_type="voice",
                reply_to=reply_to,
                metadata=metadata,
                operation_name="send_voice",
                caption=caption,
                sender=lambda media_id, _display_name: self._send_audio_message(
                    chat_id,
                    media_id,
                    int(duration_ms),
                    metadata=metadata,
                ),
            )

        return await self._send_uploaded_local_media(
            chat_id,
            audio_path,
            kind_label="audio",
            failure_prefix="Voice send failed.",
            upload_media_type="file",
            reply_to=reply_to,
            metadata=metadata,
            operation_name="send_voice",
            caption=caption,
            sender=lambda media_id, display_name: self._send_file_message(
                chat_id,
                media_id,
                display_name,
                metadata=metadata,
            ),
        )

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a video message via native DingTalk robot media messaging.

        If ``duration_seconds`` and ``thumbnail_path`` are both provided via
        kwargs, sends a native ``sampleVideo`` message.  Otherwise falls back
        to ``sampleFile`` (file message) so the video still reaches the user.
        """
        duration_seconds = kwargs.get("duration_seconds")
        thumbnail_path = (
            kwargs.get("thumbnail_path")
            or kwargs.get("cover_path")
            or kwargs.get("cover_image_path")
        )
        video_type = Path(video_path).suffix.lstrip(".").lower()
        context, display_name, early_result = await self._prepare_local_media_send(
            chat_id,
            video_path,
            kind_label="video",
            failure_prefix="Video send failed.",
            reply_to=reply_to,
            metadata=metadata,
        )
        if early_result:
            return early_result

        try:
            # Native sampleVideo path: requires duration + thumbnail + mp4
            if (
                duration_seconds is not None
                and thumbnail_path
                and os.path.exists(str(thumbnail_path))
                and video_type == "mp4"
            ):
                video_media_id = await self._upload_media(video_path, media_type="video")
                cover_media_id = await self._upload_media(str(thumbnail_path), media_type="image")
                await self._send_media_caption(
                    chat_id, caption, reply_to=reply_to, context=context, metadata=metadata,
                )
                result = await self._send_video_message(
                    chat_id, video_media_id, cover_media_id,
                    int(duration_seconds), video_type, metadata=metadata,
                )
                return await self._finalize_local_media_send(
                    chat_id,
                    context,
                    result,
                    kind_label="video",
                    display_name=display_name,
                    reply_to=reply_to,
                    failure_prefix="Video send failed.",
                )

            return await self._send_uploaded_local_media(
                chat_id,
                video_path,
                kind_label="video",
                failure_prefix="Video send failed.",
                upload_media_type="file",
                reply_to=reply_to,
                metadata=metadata,
                operation_name="send_video",
                caption=caption,
                display_name=display_name,
                prepared_context=context,
                sender=lambda media_id, resolved_name: self._send_file_message(
                    chat_id,
                    media_id,
                    resolved_name,
                    metadata=metadata,
                ),
            )
        except Exception as exc:
            error = str(exc)
            logger.error("[%s] send_video error: %s", self.name, error)
            await self._send_error_notice(chat_id, context, error, reply_to=reply_to, prefix="Video send failed.")
            return SendResult(success=False, error=error)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send an animation (delegates to send_image)."""
        return await self.send_image(
            chat_id, animation_url, caption=caption, reply_to=reply_to, metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Internal stream handler
# ---------------------------------------------------------------------------

class _IncomingHandler(ChatbotHandler if DINGTALK_STREAM_AVAILABLE else object):
    """dingtalk-stream ChatbotHandler that forwards messages to the adapter."""

    def __init__(self, adapter: DingTalkAdapter, loop: asyncio.AbstractEventLoop):
        if DINGTALK_STREAM_AVAILABLE:
            super().__init__()
        self._adapter = adapter
        self._loop = loop

    @staticmethod
    def _payload_to_message(payload: Dict[str, Any]) -> SimpleNamespace:
        """Normalize callback payload keys into ChatbotMessage-like attributes."""
        normalized = dict(payload)
        alias_map = {
            "msgId": "message_id",
            "senderId": "sender_id",
            "senderStaffId": "sender_staff_id",
            "senderNick": "sender_nick",
            "robotCode": "robot_code",
            "conversationId": "conversation_id",
            "conversationType": "conversation_type",
            "conversationTitle": "conversation_title",
            "sessionWebhook": "session_webhook",
            "createAt": "create_at",
            "msgtype": "message_type",
        }
        for source_key, target_key in alias_map.items():
            if source_key in payload and target_key not in normalized:
                normalized[target_key] = payload[source_key]
        text_value = normalized.get("text")
        if isinstance(text_value, dict):
            normalized["text"] = SimpleNamespace(**text_value)
        return SimpleNamespace(**normalized)

    async def process(self, message):
        """Called by dingtalk-stream when a callback message arrives."""
        if DINGTALK_STREAM_AVAILABLE and isinstance(message, ChatbotMessage):
            incoming = message
        else:
            payload = getattr(message, "data", None) or {}
            if isinstance(payload, dict):
                if DINGTALK_STREAM_AVAILABLE:
                    incoming = ChatbotMessage.from_dict(payload)
                else:
                    incoming = self._payload_to_message(payload)
            else:
                incoming = payload
            if isinstance(payload, dict) and incoming is not None:
                try:
                    setattr(incoming, "_raw_data", payload)
                except Exception:
                    pass

        loop = self._loop
        if loop is None or loop.is_closed():
            logger.error("[DingTalk] Event loop unavailable, cannot dispatch message")
            return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        try:
            await self._adapter._on_message(incoming)
        except Exception:
            logger.exception("[DingTalk] Error processing incoming message")
        return dingtalk_stream.AckMessage.STATUS_OK, "OK"

    async def raw_process(self, callback):
        """Compatibility shim for SDK callback payload processing."""
        payload = getattr(callback, "data", None) or {}
        if isinstance(payload, dict):
            if DINGTALK_STREAM_AVAILABLE:
                incoming = ChatbotMessage.from_dict(payload)
            else:
                incoming = self._payload_to_message(payload)
            try:
                setattr(incoming, "_raw_data", payload)
            except Exception:
                pass
        else:
            incoming = payload

        loop = self._loop
        if loop is None or loop.is_closed():
            logger.error("[DingTalk] Event loop unavailable, cannot dispatch message")
        else:
            try:
                await self._adapter._on_message(incoming)
            except Exception:
                logger.exception("[DingTalk] Error processing incoming callback")

        if DINGTALK_STREAM_AVAILABLE and dingtalk_stream is not None:
            return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        headers = getattr(callback, "headers", None) or SimpleNamespace()
        return SimpleNamespace(code=200, headers=headers)
