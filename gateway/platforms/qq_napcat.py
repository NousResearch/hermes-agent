"""QQ platform adapter via NapCat / OneBot 11 WebSocket API."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, unquote, urlsplit, urlunsplit

try:
    import aiohttp

    QQ_NAPCAT_AVAILABLE = True
except ImportError:
    aiohttp = None
    QQ_NAPCAT_AVAILABLE = False

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
)

logger = logging.getLogger(__name__)


def check_qq_napcat_requirements() -> bool:
    """Return True when the optional transport dependencies are installed."""
    return QQ_NAPCAT_AVAILABLE


def _guess_ext_from_name(name: str, default: str) -> str:
    suffix = Path(_decoded_path_from_ref(name)).suffix.lower()
    return suffix or default


def _decoded_path_from_ref(value: str) -> str:
    """Return the path portion of a URL/URI with escapes decoded."""
    text = str(value or "").strip()
    parsed = urlsplit(text)
    return unquote(parsed.path or text)


def _with_access_token(ws_url: str, access_token: str) -> str:
    """Append/replace access_token without corrupting an existing path or query."""
    ws_url = str(ws_url or "").strip()
    token = str(access_token or "").strip()
    if not token:
        return ws_url

    parsed = urlsplit(ws_url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "access_token"
    ]
    query_items.append(("access_token", token))
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(query_items),
        parsed.fragment,
    ))


def resolve_qq_napcat_group_id(value: Any) -> int:
    """Normalize a QQ group target into its numeric group id."""
    text = str(value or "").strip()
    if text.startswith("qq_napcat:"):
        text = text.split(":", 1)[1]
    if text.startswith("group:"):
        return int(text.split(":", 1)[1])
    if text.startswith("dm:"):
        raise ValueError("QQ NapCat group file actions require a group target, not dm:<id>")
    if text.lstrip("-").isdigit():
        return int(text)
    raise ValueError("QQ NapCat group target must use 'group:<id>' or a numeric group id")


def normalize_qq_napcat_local_path(path: str) -> str:
    """Expand and absolutize a local file path for NapCat file APIs."""
    return str(Path(os.path.abspath(os.path.expanduser(path))))


class QqNapCatAdapter(BasePlatformAdapter):
    """Hermes platform adapter for QQ via NapCat's OneBot websocket."""

    platform = Platform.QQ_NAPCAT
    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQ_NAPCAT)
        extra = config.extra or {}
        self.ws_url = str(extra.get("ws_url") or "").strip()
        self.access_token = str(extra.get("access_token") or "").strip()
        self.reconnect_interval = int(extra.get("reconnect_interval") or 5)
        self.allowed_groups = {str(group) for group in (extra.get("allowed_groups") or []) if str(group).strip()}
        self.allow_all_groups = bool(extra.get("allow_all_groups", False))
        self._mention_patterns = self._compile_mention_patterns()
        self._bot_user_id = ""
        self._followup_window_seconds = int(extra.get("followup_window_seconds") or 900)
        self._group_followup_windows: Dict[Tuple[str, str], float] = {}
        self._recent_group_bot_messages: Dict[str, Tuple[str, float]] = {}
        self._max_recent_group_messages = 500

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._pending_calls: Dict[str, asyncio.Future] = {}
        self._echo_counter = itertools.count(1)
        self._chat_types: Dict[str, str] = {}

    def _qq_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in ("true", "1", "yes", "on")
            return bool(configured)
        return os.getenv("QQ_NAPCAT_REQUIRE_MENTION", "false").lower() in ("true", "1", "yes", "on")

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("QQ_NAPCAT_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    patterns = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not patterns:
                        patterns = [part.strip() for part in raw.split(",") if part.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning("[%s] qq_napcat mention_patterns must be a list or string; got %s", self.name, type(patterns).__name__)
            return []

        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid QQ mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d QQ mention pattern(s)", self.name, len(compiled))
        return compiled

    @staticmethod
    def _extract_reply_message_id(payload: Dict[str, Any]) -> Optional[str]:
        segments = payload.get("message")
        if isinstance(segments, list):
            for segment in segments:
                if str(segment.get("type") or "").lower() != "reply":
                    continue
                reply_id = str((segment.get("data") or {}).get("id") or "").strip()
                if reply_id:
                    return reply_id
        reply_id = str(payload.get("reply_to_message_id") or "").strip()
        return reply_id or None

    def _message_mentions_bot(self, payload: Dict[str, Any]) -> bool:
        bot_id = str(payload.get("self_id") or self._bot_user_id or "").strip()
        if bot_id:
            self._bot_user_id = bot_id

        segments = payload.get("message")
        if isinstance(segments, list):
            for segment in segments:
                if str(segment.get("type") or "").lower() != "at":
                    continue
                qq = str((segment.get("data") or {}).get("qq") or "").strip()
                if qq and bot_id and qq == bot_id:
                    return True
                if qq and self._bot_user_id and qq == self._bot_user_id:
                    return True

        raw_text = str(payload.get("raw_message") or "")
        for candidate in filter(None, {bot_id, self._bot_user_id}):
            if f"@{candidate}" in raw_text:
                return True
        return False

    def _message_matches_mention_patterns(self, payload: Dict[str, Any]) -> bool:
        if not self._mention_patterns:
            return False
        body = str(payload.get("raw_message") or "")
        if not body:
            segments = payload.get("message")
            if isinstance(segments, list):
                body = "".join(
                    str((segment.get("data") or {}).get("text") or "")
                    for segment in segments
                    if str(segment.get("type") or "").lower() == "text"
                )
        return any(pattern.search(body) for pattern in self._mention_patterns)

    def _clean_bot_mention_text(self, text: str, payload: Dict[str, Any]) -> str:
        bot_id = str(payload.get("self_id") or self._bot_user_id or "").strip()
        if bot_id:
            self._bot_user_id = bot_id

        segments = payload.get("message")
        if isinstance(segments, list):
            parts = []
            for segment in segments:
                seg_type = str(segment.get("type") or "").lower()
                data = segment.get("data") or {}
                if seg_type == "at":
                    qq = str(data.get("qq") or "").strip()
                    if qq and bot_id and qq == bot_id:
                        continue
                    if qq and self._bot_user_id and qq == self._bot_user_id:
                        continue
                    label = str(data.get("name") or qq or "").strip()
                    if label:
                        parts.append(label if label.startswith("@") else f"@{label}")
                    continue
                if seg_type == "text":
                    parts.append(str(data.get("text") or ""))
            return "".join(parts).strip()

        cleaned = text or ""
        for candidate in filter(None, {bot_id, self._bot_user_id}):
            cleaned = re.sub(rf"@{re.escape(candidate)}\b[,:\-]*\s*", "", cleaned)
        return cleaned.strip()

    def _cleanup_group_tracking_state(self) -> None:
        now = time.time()

        expired_followups = [
            key for key, expires_at in self._group_followup_windows.items()
            if expires_at <= now
        ]
        for key in expired_followups:
            self._group_followup_windows.pop(key, None)

        expired_message_ids = [
            message_id
            for message_id, (_, expires_at) in self._recent_group_bot_messages.items()
            if expires_at <= now
        ]
        for message_id in expired_message_ids:
            self._recent_group_bot_messages.pop(message_id, None)

        overflow = len(self._recent_group_bot_messages) - self._max_recent_group_messages
        if overflow > 0:
            stale_items = sorted(
                self._recent_group_bot_messages.items(),
                key=lambda item: item[1][1],
            )[:overflow]
            for message_id, _ in stale_items:
                self._recent_group_bot_messages.pop(message_id, None)

    def _message_is_reply_to_bot(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        reply_id = self._extract_reply_message_id(payload)
        if not group_id or not reply_id:
            return False

        self._cleanup_group_tracking_state()
        tracked = self._recent_group_bot_messages.get(reply_id)
        if not tracked:
            return False
        tracked_group_id, expires_at = tracked
        if expires_at <= time.time():
            self._recent_group_bot_messages.pop(reply_id, None)
            return False
        return tracked_group_id == group_id

    def _has_followup_window(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        user_id = str(payload.get("user_id") or "").strip()
        if not group_id or not user_id:
            return False

        self._cleanup_group_tracking_state()
        expires_at = self._group_followup_windows.get((group_id, user_id))
        if not expires_at:
            return False
        if expires_at <= time.time():
            self._group_followup_windows.pop((group_id, user_id), None)
            return False
        return True

    def _should_process_group_message(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return False
        if not self.allow_all_groups and self.allowed_groups and group_id not in self.allowed_groups:
            return False
        if not self.allow_all_groups and not self.allowed_groups:
            return False
        if not self._qq_require_mention():
            return True

        raw_text = str(payload.get("raw_message") or "").strip()
        if raw_text.startswith("/"):
            return True
        if self._message_mentions_bot(payload):
            return True
        if self._message_is_reply_to_bot(payload):
            return True
        if self._has_followup_window(payload):
            return True
        return self._message_matches_mention_patterns(payload)

    async def connect(self) -> bool:
        if not QQ_NAPCAT_AVAILABLE:
            logger.error("QQ NapCat: aiohttp is not installed")
            return False
        if not self.ws_url:
            logger.error("QQ NapCat: missing ws_url in platform config")
            return False

        await self.disconnect()

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            )
            await self._connect_websocket()
        except Exception as exc:
            logger.error("QQ NapCat: websocket connect failed: %s", exc)
            await self.disconnect()
            return False

        self._running = True
        self._mark_connected()
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.info("QQ NapCat connected to %s", self.ws_url)
        return True

    async def disconnect(self) -> None:
        self._running = False

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

        for future in self._pending_calls.values():
            if not future.done():
                future.cancel()
        self._pending_calls.clear()
        self._mark_disconnected()

    async def _connect_websocket(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            )

        url = _with_access_token(self.ws_url, self.access_token)
        self._ws = await self._session.ws_connect(url, heartbeat=30)

    async def _reader_loop(self) -> None:
        while self._running:
            ws = self._ws
            if ws is None:
                try:
                    await self._connect_websocket()
                    self._mark_connected()
                    ws = self._ws
                except Exception as exc:
                    logger.warning(
                        "QQ NapCat reconnect failed, retrying in %ss: %s",
                        self.reconnect_interval,
                        exc,
                    )
                    await asyncio.sleep(self.reconnect_interval)
                    continue

            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                        continue

                    data = msg.json(loads=json.loads)
                    echo = data.get("echo")
                    if echo:
                        future = self._pending_calls.pop(str(echo), None)
                        if future and not future.done():
                            future.set_result(data)
                        continue

                    if data.get("post_type") == "message":
                        asyncio.create_task(self._handle_payload(data))

                if not self._running:
                    return
                self._ws = None
                await asyncio.sleep(self.reconnect_interval)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "QQ NapCat reader error, retrying in %ss: %s",
                    self.reconnect_interval,
                    exc,
                )
                self._ws = None
                await asyncio.sleep(self.reconnect_interval)

    async def _call_api(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("QQ NapCat websocket is not connected")

        echo = f"hermes-qq-napcat-{next(self._echo_counter)}"
        payload = {"action": action, "params": params, "echo": echo}

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_calls[echo] = future

        async with self._send_lock:
            await self._ws.send_json(payload)

        try:
            response = await asyncio.wait_for(future, timeout=30)
        finally:
            self._pending_calls.pop(echo, None)

        if response.get("status") != "ok" or response.get("retcode") not in (0, None):
            message = response.get("message") or "unknown error"
            raise RuntimeError(f"NapCat API error ({response.get('retcode')}): {message}")
        return response.get("data") or {}

    def _build_message_event(self, payload: Dict[str, Any]) -> MessageEvent:
        message_type = str(payload.get("message_type") or "private").lower()
        is_group = message_type == "group"
        chat_id = str(payload.get("group_id") if is_group else payload.get("user_id") or "")
        user_id = str(payload.get("user_id") or "")
        sender = payload.get("sender") or {}
        nickname = sender.get("nickname") or user_id
        user_name = sender.get("card") or nickname if is_group else nickname

        self._chat_types[chat_id] = "group" if is_group else "private"

        segments = payload.get("message")
        text = str(payload.get("raw_message") or "").strip()
        if not text and isinstance(segments, list):
            text_parts = []
            for segment in segments:
                seg_type = str(segment.get("type") or "")
                data = segment.get("data") or {}
                if seg_type == "text":
                    text_parts.append(str(data.get("text") or ""))
                elif seg_type == "at":
                    text_parts.append(f"@{data.get('qq')}")
            text = "".join(text_parts).strip()

        normalized_type = MessageType.TEXT
        if isinstance(segments, list):
            seg_types = {str(segment.get("type") or "").lower() for segment in segments}
            if "record" in seg_types:
                normalized_type = MessageType.VOICE
            elif "video" in seg_types:
                normalized_type = MessageType.VIDEO
            elif "image" in seg_types:
                normalized_type = MessageType.PHOTO
            elif "file" in seg_types:
                normalized_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(payload.get("group_id") or chat_id) if is_group else user_name,
            chat_type="group" if is_group else "dm",
            user_id=user_id or None,
            user_name=user_name or None,
        )
        return MessageEvent(
            text=text,
            message_type=normalized_type,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("message_id")) if payload.get("message_id") is not None else None,
            reply_to_message_id=self._extract_reply_message_id(payload),
            timestamp=datetime.fromtimestamp(payload.get("time", time.time())),
        )

    def _record_successful_response_context(
        self,
        event: MessageEvent,
        sent_message_ids: list[str],
    ) -> None:
        source = getattr(event, "source", None)
        if not source or str(getattr(source, "chat_type", "")).lower() != "group":
            return

        group_id = str(getattr(source, "chat_id", "") or "").strip()
        user_id = str(getattr(source, "user_id", "") or "").strip()
        if not group_id or not user_id or self._followup_window_seconds <= 0:
            return

        self._cleanup_group_tracking_state()
        expires_at = time.time() + self._followup_window_seconds
        self._group_followup_windows[(group_id, user_id)] = expires_at

        for message_id in sent_message_ids:
            normalized_message_id = str(message_id or "").strip()
            if normalized_message_id:
                self._recent_group_bot_messages[normalized_message_id] = (
                    group_id,
                    expires_at,
                )

        self._cleanup_group_tracking_state()

    async def _handle_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("post_type") != "message":
            return

        message_type = str(payload.get("message_type") or "").lower()
        if message_type == "group" and not self._should_process_group_message(payload):
            return

        try:
            event = self._build_message_event(payload)
            if message_type == "group":
                event.text = self._clean_bot_mention_text(event.text, payload)
            await self._populate_media(event, payload)
            await self.handle_message(event)
        except Exception:
            logger.exception("QQ NapCat: failed to handle payload")

    async def _populate_media(self, event: MessageEvent, payload: Dict[str, Any]) -> None:
        segments = payload.get("message")
        if not isinstance(segments, list):
            return

        for segment in segments:
            seg_type = str(segment.get("type") or "").lower()
            if seg_type not in {"image", "record", "video", "file"}:
                continue
            data = segment.get("data") or {}
            try:
                cached_path, mime_type = await self._cache_segment_media(seg_type, data)
            except Exception as exc:
                logger.debug("QQ NapCat: failed to cache %s segment: %s", seg_type, exc)
                continue
            if cached_path:
                event.media_urls.append(cached_path)
                event.media_types.append(mime_type)

    async def _cache_segment_media(self, seg_type: str, data: Dict[str, Any]) -> tuple[Optional[str], str]:
        url = str(data.get("url") or data.get("file") or "").strip()
        if not url:
            return None, "application/octet-stream"

        if url.startswith("file://"):
            local_path = unquote(urlsplit(url).path)
            if os.path.exists(local_path):
                return local_path, self._mime_for_segment(seg_type, local_path)
            return None, "application/octet-stream"

        if os.path.isabs(url) and os.path.exists(url):
            return url, self._mime_for_segment(seg_type, url)

        if seg_type == "image":
            image_path = await cache_image_from_url(url, ext=_guess_ext_from_name(url, ".jpg"))
            return image_path, self._mime_for_segment(seg_type, image_path)

        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            raise ValueError("unsafe media URL rejected")

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw = response.content

        if seg_type == "record":
            ext = _guess_ext_from_name(url, ".ogg")
            return cache_audio_from_bytes(raw, ext=ext), self._mime_for_segment(seg_type, ext)

        ext = _guess_ext_from_name(url, ".bin")
        filename = Path(_decoded_path_from_ref(url)).name or f"napcat{ext}"
        return cache_document_from_bytes(raw, filename), self._mime_for_segment(seg_type, filename)

    @staticmethod
    def _mime_for_segment(seg_type: str, value: str) -> str:
        suffix = Path(str(value)).suffix.lower()
        if seg_type == "image":
            return {
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
        if seg_type == "record":
            return {
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
            }.get(suffix, "audio/ogg")
        if seg_type == "video":
            return "video/mp4"
        return "application/octet-stream"

    @staticmethod
    def _local_file_uri(path: str) -> str:
        return Path(os.path.abspath(os.path.expanduser(path))).as_uri()

    def _resolve_target(self, chat_id: str) -> tuple[str, int]:
        if chat_id.startswith("group:"):
            return "group", int(chat_id.split(":", 1)[1])
        if chat_id.startswith("dm:"):
            return "private", int(chat_id.split(":", 1)[1])
        remembered = self._chat_types.get(str(chat_id), "private")
        return remembered, int(str(chat_id))

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            chat_type, numeric_id = self._resolve_target(chat_id)
            action = "send_group_msg" if chat_type == "group" else "send_private_msg"
            id_key = "group_id" if chat_type == "group" else "user_id"
            message = []
            if reply_to:
                message.append({"type": "reply", "data": {"id": str(reply_to)}})
            message.append({"type": "text", "data": {"text": self.format_message(content)}})
            data = await self._call_api(action, {id_key: numeric_id, "message": message})
            return SendResult(
                success=True,
                message_id=str(data.get("message_id")) if data.get("message_id") is not None else None,
                raw_response=data,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _send_media(
        self,
        chat_id: str,
        segment_type: str,
        path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        try:
            if not os.path.exists(path):
                return SendResult(success=False, error=f"Media file not found: {path}")

            chat_type, numeric_id = self._resolve_target(chat_id)
            action = "send_group_msg" if chat_type == "group" else "send_private_msg"
            id_key = "group_id" if chat_type == "group" else "user_id"
            message = []
            if reply_to:
                message.append({"type": "reply", "data": {"id": str(reply_to)}})
            if caption:
                message.append({"type": "text", "data": {"text": caption}})
            message.append(
                {
                    "type": segment_type,
                    "data": {"file": self._local_file_uri(path)},
                }
            )
            data = await self._call_api(action, {id_key: numeric_id, "message": message})
            return SendResult(
                success=True,
                message_id=str(data.get("message_id")) if data.get("message_id") is not None else None,
                raw_response=data,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "image", image_path, caption=caption, reply_to=reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "record", audio_path, caption=caption, reply_to=reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "video", video_path, caption=caption, reply_to=reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del file_name
        return await self._send_media(chat_id, "file", file_path, caption=caption, reply_to=reply_to)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del chat_id, metadata
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type, _ = self._resolve_target(chat_id)
        return {"name": str(chat_id), "type": "group" if chat_type == "group" else "dm"}

    async def upload_group_file(
        self,
        group_id: str,
        file_path: str,
        folder_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        local_path = normalize_qq_napcat_local_path(file_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Group file not found: {local_path}")

        params: Dict[str, Any] = {
            "group_id": resolve_qq_napcat_group_id(group_id),
            "file": local_path,
            "name": str(file_name or Path(local_path).name),
        }
        normalized_folder_id = str(folder_id or "").strip() or None
        if normalized_folder_id and normalized_folder_id != "/":
            params["folder"] = normalized_folder_id
        return await self._call_api("upload_group_file", params)

    async def get_group_root_files(self, group_id: str) -> Dict[str, Any]:
        return await self._call_api(
            "get_group_root_files",
            {"group_id": resolve_qq_napcat_group_id(group_id)},
        )

    async def get_group_files_by_folder(self, group_id: str, folder_id: str) -> Dict[str, Any]:
        if not str(folder_id or "").strip():
            raise ValueError("folder_id is required when listing a QQ group folder")
        return await self._call_api(
            "get_group_files_by_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "folder_id": str(folder_id),
            },
        )

    async def delete_group_file(self, group_id: str, file_id: str, busid: int) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to delete a QQ group file")
        return await self._call_api(
            "delete_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "busid": int(busid),
            },
        )

    async def create_group_file_folder(
        self,
        group_id: str,
        name: str,
        parent_id: str = "/",
    ) -> Dict[str, Any]:
        if not str(name or "").strip():
            raise ValueError("name is required to create a QQ group folder")
        parent = str(parent_id or "").strip() or "/"
        if parent != "/":
            raise ValueError("NapCat group folder creation currently supports only the root parent_id '/'")
        return await self._call_api(
            "create_group_file_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "name": str(name),
                "parent_id": parent,
            },
        )

    async def delete_group_folder(self, group_id: str, folder_id: str) -> Dict[str, Any]:
        if not str(folder_id or "").strip():
            raise ValueError("folder_id is required to delete a QQ group folder")
        return await self._call_api(
            "delete_group_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "folder_id": str(folder_id),
            },
        )

    async def get_group_file_system_info(self, group_id: str) -> Dict[str, Any]:
        return await self._call_api(
            "get_group_file_system_info",
            {"group_id": resolve_qq_napcat_group_id(group_id)},
        )

    async def get_group_file_url(self, group_id: str, file_id: str, busid: int) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to fetch a QQ group file URL")
        return await self._call_api(
            "get_group_file_url",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "busid": int(busid),
            },
        )

    async def move_group_file(self, group_id: str, file_id: str, target_dir: str) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to move a QQ group file")
        if not str(target_dir or "").strip():
            raise ValueError("target_dir is required to move a QQ group file")
        return await self._call_api(
            "move_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "target_dir": str(target_dir),
            },
        )

    async def rename_group_file(
        self,
        group_id: str,
        file_id: str,
        current_parent_directory: str,
        new_name: str,
    ) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to rename a QQ group file")
        if not str(current_parent_directory or "").strip():
            raise ValueError("current_parent_directory is required to rename a QQ group file")
        if not str(new_name or "").strip():
            raise ValueError("new_name is required to rename a QQ group file")
        return await self._call_api(
            "rename_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "current_parent_directory": str(current_parent_directory),
                "new_name": str(new_name),
            },
        )

    async def trans_group_file(self, group_id: str, file_id: str, target_group_id: str) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to forward a QQ group file")
        return await self._call_api(
            "trans_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "target_group_id": resolve_qq_napcat_group_id(target_group_id),
            },
        )
