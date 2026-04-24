"""
LINE Messaging API platform adapter.

Receives webhooks from a LINE Official Account, verifies the request
signature, normalizes inbound events into Hermes MessageEvent objects, and
delivers outbound responses through the LINE push-message API.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency gate
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    utf16_len,
)

logger = logging.getLogger(__name__)

LINE_API_BASE = "https://api.line.me"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_WEBHOOK_PATH = "/line/webhook"
DEFAULT_HEALTH_PATH = "/health"
MAX_MESSAGE_LENGTH = 1200
MAX_MESSAGES_PER_PUSH = 5
EVENT_DEDUP_TTL_SECONDS = 3600


def check_line_requirements() -> bool:
    """Return True when runtime dependencies for LINE are available."""
    return AIOHTTP_AVAILABLE


def _safe_id(value: Optional[str], keep: int = 8) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "?"
    if len(raw) <= keep:
        return raw
    return raw[:keep]


def _guess_extension(content_type: str, fallback: str) -> str:
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed:
        return guessed
    return fallback


class LineAdapter(BasePlatformAdapter):
    """Webhook-based LINE adapter."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINE)
        self._token = str(config.token or config.extra.get("channel_access_token") or "").strip()
        self._channel_secret = str(config.extra.get("channel_secret") or "").strip()
        self._host = str(config.extra.get("webhook_host") or DEFAULT_HOST).strip() or DEFAULT_HOST
        self._port = int(config.extra.get("webhook_port") or DEFAULT_PORT)
        self._webhook_path = str(config.extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH).strip() or DEFAULT_WEBHOOK_PATH
        if not self._webhook_path.startswith("/"):
            self._webhook_path = f"/{self._webhook_path}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._profile_cache: Dict[str, tuple[str, float]] = {}
        self._seen_events: Dict[str, float] = {}

    async def connect(self) -> bool:
        if not self._token:
            message = "LINE startup failed: LINE_CHANNEL_ACCESS_TOKEN is required"
            self._set_fatal_error("line_missing_token", message, retryable=False)
            logger.warning("[%s] %s", self.name, message)
            return False
        if not self._channel_secret:
            message = "LINE startup failed: LINE_CHANNEL_SECRET is required"
            self._set_fatal_error("line_missing_secret", message, retryable=False)
            logger.warning("[%s] %s", self.name, message)
            return False

        self._session = aiohttp.ClientSession(trust_env=True)
        app = web.Application()
        app.router.add_get(DEFAULT_HEALTH_PATH, self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()
        logger.info("[%s] Listening on %s:%d%s", self.name, self._host, self._port, self._webhook_path)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        self._runner = None
        self._site = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text to a LINE user/group/room via push messages."""
        del reply_to, metadata
        if not self._session:
            return SendResult(success=False, error="LINE client session is not connected", retryable=True)

        formatted = self.format_message(content).strip()
        if not formatted:
            return SendResult(success=True)

        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
        logger.info("[%s] Sending %d LINE chunk(s) to %s", self.name, len(chunks), _safe_id(chat_id))
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        last_request_id = None
        try:
            for i in range(0, len(chunks), MAX_MESSAGES_PER_PUSH):
                batch = chunks[i:i + MAX_MESSAGES_PER_PUSH]
                payload = {
                    "to": chat_id,
                    "messages": [{"type": "text", "text": chunk} for chunk in batch],
                }
                async with self._session.post(
                    f"{LINE_API_BASE}/v2/bot/message/push",
                    headers=headers,
                    json=payload,
                ) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        return SendResult(
                            success=False,
                            error=f"LINE push failed ({resp.status}): {body[:500]}",
                            retryable=resp.status >= 500 or resp.status == 429,
                        )
                    last_request_id = resp.headers.get("x-line-request-id")
            return SendResult(success=True, message_id=last_request_id)
        except Exception as exc:
            return SendResult(success=False, error=f"LINE push failed: {exc}", retryable=True)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """LINE typing/loading indicators are optional here; treat as no-op."""
        del chat_id, metadata

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    def format_message(self, content: str) -> str:
        text = (content or "").replace("\r\n", "\n").strip()
        if not text:
            return ""

        # LINE renders markdown literally, so normalize to chat-friendly plain text.
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"_(.+?)_", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"```", "", text)
        text = re.sub(r"`(.+?)`", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"^\s*\*\s+", "・", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*-\s+", "・", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        del request
        return web.json_response({"status": "ok", "platform": "line"})

    def _is_signature_valid(self, body: bytes, signature: str) -> bool:
        if not signature:
            return False
        computed = base64.b64encode(
            hmac.new(
                self._channel_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return hmac.compare_digest(computed, signature)

    def _remember_event(self, event_id: str) -> bool:
        if not event_id:
            return False
        cutoff = time.time() - EVENT_DEDUP_TTL_SECONDS
        stale = [key for key, ts in self._seen_events.items() if ts < cutoff]
        for key in stale:
            self._seen_events.pop(key, None)
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = time.time()
        return False

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        signature = str(request.headers.get("x-line-signature", "") or "")
        if not self._is_signature_valid(body, signature):
            logger.warning("[%s] Rejected webhook with invalid signature from %s", self.name, request.remote)
            return web.Response(status=401, text="Invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return web.Response(status=400, text="Invalid JSON")

        events = payload.get("events", [])
        logger.info(
            "[%s] Webhook accepted destination=%s events=%d body=%s",
            self.name,
            _safe_id(payload.get("destination")),
            len(events) if isinstance(events, list) else -1,
            json.dumps(payload, ensure_ascii=False)[:1200],
        )

        for event in events:
            try:
                await self._process_event(event)
            except Exception as exc:
                logger.exception("[%s] Failed to process LINE event: %s", self.name, exc)

        return web.json_response({"ok": True})

    async def _process_event(self, event: Dict[str, Any]) -> None:
        event_id = str(event.get("webhookEventId") or "")
        if event_id and self._remember_event(event_id):
            logger.debug("[%s] Skipping duplicate LINE event %s", self.name, event_id)
            return

        if event.get("type") != "message":
            logger.info("[%s] Ignoring non-message LINE event type=%s", self.name, event.get("type"))
            return

        source_info = event.get("source") or {}
        source_type = str(source_info.get("type") or "user")
        if source_type == "user":
            chat_id = str(source_info.get("userId") or "")
            chat_type = "dm"
        elif source_type == "group":
            chat_id = str(source_info.get("groupId") or "")
            chat_type = "group"
        elif source_type == "room":
            chat_id = str(source_info.get("roomId") or "")
            chat_type = "group"
        else:
            return

        user_id = str(source_info.get("userId") or "") or None
        if not chat_id:
            logger.warning("[%s] LINE event missing chat_id source_type=%s payload=%s", self.name, source_type, json.dumps(source_info, ensure_ascii=False))
            return

        user_name = await self._resolve_user_name(user_id) if user_id else None
        message = event.get("message") or {}
        logger.info(
            "[%s] Received LINE message event type=%s source_type=%s chat_id=%s user_id=%s message_type=%s",
            self.name,
            event.get("type"),
            source_type,
            _safe_id(chat_id),
            _safe_id(user_id),
            message.get("type"),
        )
        normalized = await self._normalize_message(message)
        if normalized is None:
            logger.warning("[%s] LINE message normalization returned None message=%s", self.name, json.dumps(message, ensure_ascii=False)[:500])
            return

        timestamp_ms = event.get("timestamp")
        if isinstance(timestamp_ms, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        inbound = MessageEvent(
            source=self.build_source(
                chat_id=chat_id,
                chat_name=user_name if chat_type == "dm" else chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name or user_id,
            ),
            text=normalized["text"],
            message_type=normalized["message_type"],
            media_urls=normalized["media_urls"],
            media_types=normalized["media_types"],
            message_id=str(message.get("id") or event_id or ""),
            raw_message=event,
            timestamp=timestamp,
        )
        logger.debug("[%s] Message from %s in %s", self.name, _safe_id(user_id), _safe_id(chat_id))
        await self.handle_message(inbound)

    async def _resolve_user_name(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id or not self._session:
            return user_id
        cached = self._profile_cache.get(user_id)
        now = time.time()
        if cached and cached[1] > now:
            return cached[0]
        try:
            async with self._session.get(
                f"{LINE_API_BASE}/v2/bot/profile/{user_id}",
                headers={"Authorization": f"Bearer {self._token}"},
            ) as resp:
                if resp.status >= 400:
                    return user_id
                data = await resp.json()
                display_name = str(data.get("displayName") or user_id)
                self._profile_cache[user_id] = (display_name, now + 3600)
                return display_name
        except Exception:
            return user_id

    async def _normalize_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        message_type = str(message.get("type") or "")
        if message_type == "text":
            return {
                "text": str(message.get("text") or ""),
                "message_type": MessageType.TEXT,
                "media_urls": [],
                "media_types": [],
            }

        message_id = str(message.get("id") or "")
        if not message_id:
            return None

        content = await self._download_message_content(message_id)
        if not content:
            return {
                "text": f"[Unsupported LINE message type: {message_type}]",
                "message_type": MessageType.TEXT,
                "media_urls": [],
                "media_types": [],
            }

        body, content_type = content
        if message_type == "image":
            path = cache_image_from_bytes(body, _guess_extension(content_type, ".jpg"))
            return {
                "text": "",
                "message_type": MessageType.PHOTO,
                "media_urls": [path],
                "media_types": [content_type or "image/jpeg"],
            }
        if message_type in {"audio", "voice"}:
            path = cache_audio_from_bytes(body, _guess_extension(content_type, ".m4a"))
            return {
                "text": "",
                "message_type": MessageType.VOICE if message_type == "voice" else MessageType.AUDIO,
                "media_urls": [path],
                "media_types": [content_type or "audio/m4a"],
            }
        if message_type == "video":
            filename = f"line-video{_guess_extension(content_type, '.mp4')}"
            path = cache_document_from_bytes(body, filename)
            return {
                "text": "",
                "message_type": MessageType.VIDEO,
                "media_urls": [path],
                "media_types": [content_type or "video/mp4"],
            }
        if message_type == "file":
            filename = str(message.get("fileName") or f"line-file{_guess_extension(content_type, '.bin')}")
            path = cache_document_from_bytes(body, filename)
            return {
                "text": "",
                "message_type": MessageType.DOCUMENT,
                "media_urls": [path],
                "media_types": [content_type or "application/octet-stream"],
            }

        return {
            "text": f"[Unsupported LINE message type: {message_type}]",
            "message_type": MessageType.TEXT,
            "media_urls": [],
            "media_types": [],
        }

    async def _download_message_content(self, message_id: str) -> Optional[tuple[bytes, str]]:
        if not self._session:
            return None
        try:
            async with self._session.get(
                f"{LINE_API_BASE}/v2/bot/message/{message_id}/content",
                headers={"Authorization": f"Bearer {self._token}"},
            ) as resp:
                if resp.status >= 400:
                    logger.warning("[%s] LINE content fetch failed (%s) for %s", self.name, resp.status, message_id)
                    return None
                body = await resp.read()
                return body, str(resp.headers.get("Content-Type", "") or "")
        except Exception as exc:
            logger.warning("[%s] LINE content fetch failed for %s: %s", self.name, message_id, exc)
            return None


async def send_line_direct(
    *,
    token: Optional[str],
    extra: Dict[str, Any],
    chat_id: str,
    message: str,
) -> Dict[str, Any]:
    """One-shot send helper for send_message and cron delivery."""
    resolved_token = str(token or extra.get("channel_access_token") or os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")).strip()
    if not resolved_token:
        return {"error": "LINE channel access token missing. Configure LINE_CHANNEL_ACCESS_TOKEN or platforms.line.token."}

    async with aiohttp.ClientSession(trust_env=True) as session:
        adapter = LineAdapter(
            PlatformConfig(
                enabled=True,
                token=resolved_token,
                extra=dict(extra or {}),
            )
        )
        adapter._session = session
        adapter._token = resolved_token
        result = await adapter.send(chat_id, message)
        if not result.success:
            return {"error": result.error or "LINE send failed"}
        return {
            "success": True,
            "platform": "line",
            "chat_id": chat_id,
            "message_id": result.message_id,
        }
