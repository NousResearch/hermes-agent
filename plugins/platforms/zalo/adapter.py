"""
Zalo Bot Platform adapter for Hermes Agent.

This adapter supports webhook mode for production Bot Platform use and long
polling for lightweight local/dev profiles. Webhooks are selected when
ZALO_WEBHOOK_URL and ZALO_WEBHOOK_SECRET are configured; otherwise the adapter
falls back to getUpdates.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, urlparse

import httpx

try:
    from aiohttp import web
except Exception:  # pragma: no cover - optional dependency guard
    web = None  # type: ignore[assignment]

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_url,
    cache_image_from_url,
)

logger = logging.getLogger(__name__)

ZALO_API_BASE = "https://bot-api.zaloplatforms.com"
ZALO_MAX_MESSAGE_LENGTH = 2000
DEFAULT_POLL_TIMEOUT_SECONDS = 25
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
POLL_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0)
MAX_BACKOFF_JITTER_RATIO = 0.25
DEFAULT_PARSE_MODE = "markdown"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 18787
DEFAULT_WEBHOOK_PATH = "/zalo/webhook"
WEBHOOK_BODY_MAX_BYTES = 1_048_576
POLL_TIMEOUT_ERROR_CODE = 408
WEBHOOK_SECRET_MIN_LENGTH = 8
WEBHOOK_SECRET_MAX_LENGTH = 256
RETRYABLE_ERROR_CODES = {408, 429, 500, 502, 503, 504}
FATAL_ERROR_CODES = {401}
EVENT_TEXT = "message.text.received"
EVENT_IMAGE = "message.image.received"
EVENT_STICKER = "message.sticker.received"
EVENT_VOICE = "message.voice.received"
EVENT_UNSUPPORTED = "message.unsupported.received"
EVENT_LINK = "message.link.received"
ZALO_NOISY_STATUS_RE = re.compile(
    r"("
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    r"|compacting\s+context\s+[—-]\s+summarizing\s+earlier\s+conversation"
    r"|preflight\s+compression"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    r")",
    re.IGNORECASE | re.DOTALL,
)
TOP_LEVEL_UPDATE_KEYS = {
    "message",
    "event",
    "data",
    "result",
    "updates",
    "messages",
    "events",
    "text",
    "event_name",
    "message_id",
    "msg_id",
    "chat_id",
    "sender",
    "recipient",
    "from",
    "user",
    "photo_url",
    "photo",
    "image",
    "image_url",
    "voice_url",
    "audio_url",
    "sticker",
    "sticker_url",
    "attachment",
    "document",
    "file",
    "video",
    "video_url",
    "link",
    "media",
    "preview",
}
TEXT_PATHS = (
    "text",
    "content.text",
    "message.text",
    "data.text",
    "event.text",
    "result.text",
    "content",
)
CHAT_ID_PATHS = (
    "chat.id",
    "chat.chat_id",
    "chat_id",
    "conversation.id",
    "from.id",
    "sender.id",
    "user.id",
    "user_id",
    "message.chat.id",
    "data.chat.id",
    "event.chat.id",
    "result.chat.id",
)
USER_ID_PATHS = (
    "from.id",
    "sender.id",
    "user.id",
    "user_id",
    "message.from.id",
    "message.sender.id",
    "data.from.id",
    "event.from.id",
    "result.from.id",
)
USER_NAME_PATHS = (
    "from.display_name",
    "sender.display_name",
    "user.display_name",
    "from.name",
    "sender.name",
    "user.name",
    "message.from.display_name",
    "message.sender.display_name",
    "data.from.display_name",
    "event.from.display_name",
    "result.from.display_name",
)
PHOTO_URL_PATHS = (
    "photo",
    "photo_url",
    "image",
    "image_url",
    "media.photo",
    "media.photo_url",
    "media.image",
    "media.url",
    "attachment.photo",
    "attachment.photo_url",
    "attachment.url",
    "message.photo",
    "message.photo_url",
    "message.image",
    "message.image_url",
    "message.media.photo",
    "message.media.photo_url",
    "message.media.url",
    "data.photo",
    "data.photo_url",
    "data.image",
    "data.image_url",
    "data.media.url",
    "event.photo",
    "event.photo_url",
    "event.image",
    "event.image_url",
    "event.media.url",
    "result.photo",
    "result.photo_url",
    "result.image",
    "result.image_url",
)
VOICE_URL_PATHS = (
    "voice_url",
    "voice",
    "audio_url",
    "media.voice_url",
    "media.voice",
    "media.audio_url",
    "media.url",
    "attachment.voice_url",
    "attachment.audio_url",
    "attachment.url",
    "message.voice_url",
    "message.voice",
    "message.audio_url",
    "message.media.voice_url",
    "message.media.url",
    "data.voice_url",
    "data.voice",
    "data.audio_url",
    "data.media.url",
    "event.voice_url",
    "event.voice",
    "event.audio_url",
    "event.media.url",
    "result.voice_url",
    "result.voice",
    "result.audio_url",
)
STICKER_URL_PATHS = (
    "url",
    "sticker.url",
    "sticker_url",
    "message.url",
    "message.sticker.url",
    "message.sticker_url",
    "data.url",
    "data.sticker.url",
    "data.sticker_url",
    "event.url",
    "event.sticker.url",
    "event.sticker_url",
    "result.url",
    "result.sticker.url",
    "result.sticker_url",
)
CAPTION_PATHS = (
    "caption",
    "content.caption",
    "message.caption",
    "data.caption",
    "event.caption",
    "result.caption",
)
DOCUMENT_URL_PATHS = (
    "document.url",
    "file.url",
    "video.url",
    "video_url",
    "attachment.file_url",
    "attachment.document_url",
    "attachment.video_url",
    "attachment.url",
    "media.file_url",
    "media.document_url",
    "media.video_url",
    "message.document.url",
    "message.file.url",
    "message.video.url",
    "message.video_url",
    "data.document.url",
    "data.file.url",
    "data.video.url",
    "data.video_url",
    "event.document.url",
    "event.file.url",
    "event.video.url",
    "event.video_url",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def _yaml_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return sorted(_csv(value))
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    item = str(value).strip()
    return [item] if item else []


def _first(mapping: Dict[str, Any], *paths: str) -> Any:
    for path in paths:
        node: Any = mapping
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node not in (None, ""):
            return node
    return None


def _first_from(payloads: Iterable[Dict[str, Any]], *paths: str) -> Any:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        value = _first(payload, *paths)
        if value not in (None, ""):
            return value
    return None


def _safe_str(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _looks_public_http_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _link_preview_text(*payloads: Dict[str, Any]) -> str:
    """Best-effort text for Zalo rich-link/URL preview events."""
    parts: list[str] = []

    def add(value: Any) -> None:
        if value in (None, ""):
            return
        text = str(value).strip()
        if text:
            parts.append(text)

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key).lower())
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str):
            return

        normalized_key = key.lower()
        stripped = value.strip()
        if _looks_public_http_url(stripped):
            add(stripped)
        elif normalized_key in {
            "url",
            "href",
            "link",
            "title",
            "description",
            "summary",
            "text",
            "caption",
            "content",
        }:
            add(stripped)

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for path in (
            "link.title",
            "link.description",
            "link.url",
            "link.href",
            "attachment.title",
            "attachment.description",
            "attachment.url",
            "media.title",
            "media.description",
            "media.url",
            "preview.title",
            "preview.description",
            "preview.url",
            "title",
            "description",
            "href",
        ):
            value = _first(payload, path)
            if value not in (None, ""):
                parts.append(str(value).strip())
        link_value = payload.get("link")
        if isinstance(link_value, str) and link_value.strip():
            parts.append(link_value.strip())
        for container_key in ("link", "links", "attachment", "attachments", "media", "preview", "payload"):
            if container_key in payload:
                visit(payload[container_key], container_key)
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return "\n".join(deduped)


def _reconstruct_obfuscated_google_url(text: str) -> str:
    """Turn Zalo-safe spaced Google links back into URLs when users avoid previews."""
    if not text:
        return ""
    lowered = text.lower()
    if "google" not in lowered or not any(host in lowered for host in ("drive", "docs")):
        return ""

    candidate = text.strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r"\s*(?:\\[dot\\]|\\(dot\\)| dot |\\.)\s*", ".", candidate, flags=re.I)
    candidate = re.sub(r"\s*(?:\\[slash\\]|\\(slash\\)| slash |/)\s*", "/", candidate, flags=re.I)
    candidate = re.sub(r"\s*:\s*/\s*/\s*", "://", candidate)
    candidate = candidate.replace(" ", "")
    if candidate.startswith("www."):
        candidate = f"https://{candidate}"
    if candidate.startswith(("drive.google.com/", "docs.google.com/")):
        candidate = f"https://{candidate}"
    return candidate if _looks_public_http_url(candidate) else ""


@dataclass
class ZaloApiError(Exception):
    method: str
    error_code: Optional[int]
    description: str
    payload: Any = None

    @property
    def retryable(self) -> bool:
        return self.error_code in RETRYABLE_ERROR_CODES

    @property
    def fatal(self) -> bool:
        return self.error_code in FATAL_ERROR_CODES

    def __str__(self) -> str:
        code = f" error_code={self.error_code}" if self.error_code is not None else ""
        return f"{self.method} failed{code}: {self.description}"


class ZaloAdapter(BasePlatformAdapter):
    """Text-first Zalo Bot Platform gateway adapter."""

    supports_code_blocks = False

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("zalo"))
        extra = getattr(config, "extra", {}) or {}

        self.bot_token = os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token", "")
        self.api_base = (
            os.getenv("ZALO_API_BASE")
            or extra.get("api_base")
            or ZALO_API_BASE
        ).rstrip("/")
        self.parse_mode = (
            os.getenv("ZALO_PARSE_MODE")
            or extra.get("parse_mode")
            or DEFAULT_PARSE_MODE
        ).strip().lower()
        if self.parse_mode not in {"", "markdown", "html"}:
            logger.warning("Zalo: unsupported parse_mode=%r; sending plain text", self.parse_mode)
            self.parse_mode = ""
        self.suppress_noisy_status = self._config_bool(
            "suppress_noisy_status",
            "ZALO_SUPPRESS_NOISY_STATUS",
            True,
        )
        self.allow_all = _truthy(os.getenv("ZALO_ALLOW_ALL_USERS")) or bool(
            extra.get("allow_all_users", False)
        )
        self.allowed_users = _csv(os.getenv("ZALO_ALLOWED_USERS", "")) | {
            str(user_id) for user_id in extra.get("allowed_users", []) if str(user_id)
        }
        self.dm_only = (
            _truthy(os.getenv("ZALO_DM_ONLY"))
            or bool(extra.get("dm_only", False))
            or bool(extra.get("private_only", False))
        )

        try:
            self.poll_timeout_seconds = int(
                os.getenv("ZALO_POLL_TIMEOUT_SECONDS")
                or extra.get("poll_timeout_seconds", DEFAULT_POLL_TIMEOUT_SECONDS)
            )
        except (TypeError, ValueError):
            self.poll_timeout_seconds = DEFAULT_POLL_TIMEOUT_SECONDS

        try:
            self.poll_interval_seconds = float(
                os.getenv("ZALO_POLL_INTERVAL_SECONDS")
                or extra.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
            )
        except (TypeError, ValueError):
            self.poll_interval_seconds = DEFAULT_POLL_INTERVAL_SECONDS

        self.connection_mode = str(
            os.getenv("ZALO_CONNECTION_MODE")
            or extra.get("connection_mode")
            or "auto"
        ).strip().lower()
        if self.connection_mode not in {"auto", "polling", "webhook"}:
            logger.warning("Zalo: unsupported connection_mode=%r; using auto", self.connection_mode)
            self.connection_mode = "auto"
        self.webhook_url = str(
            os.getenv("ZALO_WEBHOOK_URL")
            or os.getenv("ZALO_WEBHOOK_PUBLIC_URL")
            or extra.get("webhook_url")
            or extra.get("webhook_public_url")
            or ""
        ).strip()
        self.webhook_secret = str(
            os.getenv("ZALO_WEBHOOK_SECRET") or extra.get("webhook_secret") or ""
        ).strip()
        self.webhook_path = str(
            os.getenv("ZALO_WEBHOOK_PATH")
            or extra.get("webhook_path")
            or DEFAULT_WEBHOOK_PATH
        ).strip() or DEFAULT_WEBHOOK_PATH
        if not self.webhook_path.startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self.webhook_host = str(
            os.getenv("ZALO_WEBHOOK_HOST")
            or extra.get("webhook_host")
            or DEFAULT_WEBHOOK_HOST
        ).strip() or DEFAULT_WEBHOOK_HOST
        try:
            self.webhook_port = int(
                os.getenv("ZALO_WEBHOOK_PORT")
                or extra.get("webhook_port", DEFAULT_WEBHOOK_PORT)
            )
        except (TypeError, ValueError):
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self.webhook_auto_register = _truthy(
            os.getenv("ZALO_WEBHOOK_AUTO_REGISTER")
            or str(extra.get("webhook_auto_register", ""))
        )
        self.delete_webhook_on_polling_start = self._config_bool(
            "delete_webhook_on_polling_start",
            "ZALO_DELETE_WEBHOOK_ON_POLLING_START",
            False,
        )
        self.delete_webhook_on_disconnect = self._config_bool(
            "delete_webhook_on_disconnect",
            "ZALO_DELETE_WEBHOOK_ON_DISCONNECT",
            False,
        )
        self.url_intake_public_base = str(
            os.getenv("ZALO_URL_INTAKE_PUBLIC_BASE")
            or extra.get("url_intake_public_base")
            or ""
        ).strip().rstrip("/")
        self.url_intake_pending_file = str(
            os.getenv("ZALO_URL_INTAKE_PENDING_FILE")
            or extra.get("url_intake_pending_file")
            or ""
        ).strip()

        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._web_runner: Any = None
        self._web_site: Any = None
        self._seen: deque[str] = deque(maxlen=2000)
        self._seen_set: set[str] = set()
        self._lock_key: Optional[str] = None
        self._bot_id: Optional[str] = None

    @property
    def _bot_base_url(self) -> str:
        return f"{self.api_base}/bot{self.bot_token}"

    async def connect(self) -> bool:
        if not self.bot_token:
            self._set_fatal_error(
                "config_missing",
                "ZALO_BOT_TOKEN must be set",
                retryable=False,
            )
            return False

        try:
            from gateway.status import acquire_scoped_lock

            token_hash = hashlib.sha256(self.bot_token.encode()).hexdigest()[:16]
            if not acquire_scoped_lock("zalo", token_hash):
                self._set_fatal_error(
                    "lock_conflict",
                    "Zalo bot token already in use by another profile",
                    retryable=False,
                )
                return False
            self._lock_key = token_hash
        except Exception:
            self._lock_key = None

        self._client = httpx.AsyncClient(timeout=self.poll_timeout_seconds + 10)

        try:
            response = await self._api("getMe", {})
            result = response.get("result") if isinstance(response, dict) else None
            if isinstance(result, dict) and result.get("id"):
                self._bot_id = str(result["id"])
        except Exception as exc:
            self._set_fatal_error(
                "auth_failed",
                f"Zalo getMe failed: {exc}",
                retryable=False,
            )
            await self.disconnect()
            return False

        self._mark_connected()

        if self._webhook_config_incomplete:
            self._set_fatal_error(
                "webhook_config_incomplete",
                "ZALO_WEBHOOK_URL/ZALO_WEBHOOK_PUBLIC_URL and ZALO_WEBHOOK_SECRET must be configured together",
                retryable=False,
            )
            await self.disconnect()
            return False

        if self._webhook_enabled:
            if not self._webhook_secret_valid:
                self._set_fatal_error(
                    "webhook_secret_invalid",
                    "ZALO_WEBHOOK_SECRET must be 8-256 characters",
                    retryable=False,
                )
                await self.disconnect()
                return False
            if not await self._start_webhook_server():
                await self.disconnect()
                return False
            if self.webhook_auto_register:
                try:
                    await self._api(
                        "setWebhook",
                        {"url": self.webhook_url, "secret_token": self.webhook_secret},
                    )
                    logger.info("Zalo: webhook registered with Bot Platform")
                except Exception as exc:
                    self._set_fatal_error(
                        "webhook_register_failed",
                        f"Zalo setWebhook failed: {exc}",
                        retryable=False,
                    )
                    await self.disconnect()
                    return False
            logger.info(
                "Zalo: webhook server started on %s:%d%s public_url=%s",
                self.webhook_host,
                self.webhook_port,
                self.webhook_path,
                self.webhook_url,
            )
        else:
            if self.delete_webhook_on_polling_start:
                try:
                    await self._api("deleteWebhook", {})
                    logger.info("Zalo: deleted existing webhook before starting long polling")
                except Exception as exc:
                    logger.warning("Zalo: deleteWebhook before polling failed: %s", exc)
            self._poll_task = asyncio.create_task(self._poll_loop(), name="zalo-poll")
            logger.info("Zalo: long polling started")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

        if self.delete_webhook_on_disconnect and self._client is not None:
            try:
                await self._api("deleteWebhook", {})
                logger.info("Zalo: deleted webhook on disconnect")
            except Exception as exc:
                logger.warning("Zalo: deleteWebhook on disconnect failed: %s", exc)

        if self._web_runner is not None:
            try:
                await self._web_runner.cleanup()
            except Exception:
                logger.debug("Zalo: webhook server cleanup failed", exc_info=True)
            self._web_runner = None
            self._web_site = None

        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Zalo: poll task ended with error", exc_info=True)
            self._poll_task = None

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

        if self._lock_key:
            try:
                from gateway.status import release_scoped_lock

                release_scoped_lock("zalo", self._lock_key)
            except Exception:
                pass
            self._lock_key = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not chat_id:
            return SendResult(success=False, error="missing chat_id")

        chunks = self.truncate_message(content or " ", ZALO_MAX_MESSAGE_LENGTH) or [" "]
        message_ids: list[str] = []
        raw_response: Any = None

        for chunk in chunks:
            payload = {"chat_id": str(chat_id), "text": chunk or " "}
            if self.parse_mode:
                payload["parse_mode"] = self.parse_mode
            try:
                raw_response = await self._api("sendMessage", payload)
            except ZaloApiError as exc:
                return SendResult(
                    success=False,
                    error=str(exc),
                    raw_response=exc.payload,
                    retryable=exc.retryable,
                )
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)

            msg_id = self._extract_message_id(raw_response)
            if msg_id:
                message_ids.append(str(msg_id))

        return SendResult(
            success=True,
            message_id=message_ids[-1] if message_ids else None,
            raw_response=raw_response,
            continuation_message_ids=tuple(message_ids[:-1]),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        try:
            await self._api("sendChatAction", {"chat_id": str(chat_id), "action": "typing"})
        except ZaloApiError as exc:
            if exc.error_code not in {400, 404}:
                logger.debug("Zalo: sendChatAction failed: %s", exc)
        except Exception as exc:
            logger.debug("Zalo: sendChatAction failed: %s", exc)
            return

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"id": str(chat_id), "name": str(chat_id), "type": "dm"}

    def prepare_gateway_status_message(self, event_type: str, message: str) -> Optional[str]:
        """Filter transient gateway status chatter before it reaches Zalo users."""
        text = str(message or "").strip()
        if not text:
            return None
        if self.suppress_noisy_status and ZALO_NOISY_STATUS_RE.search(text):
            return None
        return text

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not _looks_public_http_url(image_url):
            return await super().send_image(
                chat_id=chat_id,
                image_url=image_url,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )

        payload: Dict[str, Any] = {"chat_id": str(chat_id), "photo": image_url}
        if caption:
            payload["caption"] = caption[:ZALO_MAX_MESSAGE_LENGTH]
        try:
            response = await self._api("sendPhoto", payload)
            return SendResult(
                success=True,
                message_id=self._extract_message_id(response),
                raw_response=response,
            )
        except ZaloApiError as exc:
            logger.debug("Zalo: native sendPhoto failed, falling back to text: %s", exc)
            return await super().send_image(
                chat_id=chat_id,
                image_url=image_url,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native Zalo sticker when the animation URL is a sticker id."""
        _ = caption, reply_to, metadata
        sticker_id = str(animation_url or "").strip()
        if not sticker_id:
            return SendResult(success=False, error="missing sticker id")
        try:
            response = await self._api(
                "sendSticker",
                {"chat_id": str(chat_id), "sticker": sticker_id},
            )
            return SendResult(
                success=True,
                message_id=self._extract_message_id(response),
                raw_response=response,
            )
        except ZaloApiError as exc:
            return SendResult(
                success=False,
                error=str(exc),
                raw_response=exc.payload,
                retryable=exc.retryable,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _api(self, method: str, payload: Dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("Zalo HTTP client is not connected")
        try:
            response = await self._client.post(f"{self._bot_base_url}/{method}", json=payload)
        except httpx.HTTPError as exc:
            raise ZaloApiError(
                method=method,
                error_code=None,
                description=str(exc),
            ) from exc
        if response.status_code >= 400:
            description = response.text[:500] if response.text else response.reason_phrase
            raise ZaloApiError(
                method=method,
                error_code=response.status_code,
                description=description,
                payload={"status_code": response.status_code, "body": response.text},
            )
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Zalo {method} returned invalid JSON") from exc
        if isinstance(data, dict) and data.get("ok") is False:
            raise ZaloApiError(
                method=method,
                error_code=data.get("error_code"),
                description=str(data.get("description") or "unknown Zalo API error"),
                payload=data,
            )
        return data

    async def _poll_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                updates = await self._api(
                    "getUpdates",
                    {"timeout": str(self.poll_timeout_seconds)},
                )
                extracted = self._extract_updates(updates)
                if extracted:
                    logger.debug("Zalo: getUpdates returned %d event(s)", len(extracted))
                for update in extracted:
                    await self._handle_update(update)
                backoff_idx = 0
            except ZaloApiError as exc:
                if exc.error_code == POLL_TIMEOUT_ERROR_CODE:
                    logger.debug("Zalo: long poll timed out with no events")
                    backoff_idx = 0
                elif exc.fatal:
                    self._set_fatal_error("api_auth_failed", str(exc), retryable=False)
                    break
                else:
                    delay = self._poll_backoff_sleep(backoff_idx)
                    logger.warning("Zalo: polling failed: %s; retrying in %.1fs", exc, delay)
                    await asyncio.sleep(delay)
                    backoff_idx = min(backoff_idx + 1, len(POLL_BACKOFF_SECONDS) - 1)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delay = self._poll_backoff_sleep(backoff_idx)
                logger.warning("Zalo: polling failed: %s; retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                backoff_idx = min(backoff_idx + 1, len(POLL_BACKOFF_SECONDS) - 1)
                continue

            await asyncio.sleep(max(self.poll_interval_seconds, 0.0))

    @property
    def _webhook_enabled(self) -> bool:
        if self.connection_mode == "polling":
            return False
        return bool(self.webhook_url and self.webhook_secret)

    @property
    def _webhook_config_incomplete(self) -> bool:
        if self.connection_mode == "polling":
            return False
        if self.connection_mode == "webhook":
            return not bool(self.webhook_url and self.webhook_secret)
        return bool(self.webhook_url) != bool(self.webhook_secret)

    @property
    def _webhook_secret_valid(self) -> bool:
        return WEBHOOK_SECRET_MIN_LENGTH <= len(self.webhook_secret) <= WEBHOOK_SECRET_MAX_LENGTH

    def _poll_backoff_sleep(self, backoff_idx: int) -> float:
        cap = POLL_BACKOFF_SECONDS[min(backoff_idx, len(POLL_BACKOFF_SECONDS) - 1)]
        jitter = cap * MAX_BACKOFF_JITTER_RATIO * random.random()
        return cap + jitter

    async def _start_webhook_server(self) -> bool:
        if web is None:
            self._set_fatal_error(
                "webhook_dependency_missing",
                "aiohttp must be installed to use Zalo webhook mode",
                retryable=False,
            )
            return False

        app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        app.router.add_get("/health", self._handle_webhook_health)
        app.router.add_post(self.webhook_path, self._handle_webhook_request)

        self._web_runner = web.AppRunner(app)
        try:
            await self._web_runner.setup()
            self._web_site = web.TCPSite(
                self._web_runner,
                self.webhook_host,
                self.webhook_port,
            )
            await self._web_site.start()
        except Exception as exc:
            self._set_fatal_error(
                "webhook_listen_failed",
                f"Zalo webhook server failed to listen on "
                f"{self.webhook_host}:{self.webhook_port}{self.webhook_path}: {exc}",
                retryable=True,
            )
            logger.error("Zalo: webhook server failed to start: %s", exc)
            if self._web_runner is not None:
                await self._web_runner.cleanup()
                self._web_runner = None
            self._web_site = None
            return False
        return True

    async def _handle_webhook_health(self, request: Any) -> Any:
        return web.Response(text="ok") if web is not None else None

    async def _handle_webhook_request(self, request: Any) -> Any:
        if web is None:
            return None

        supplied_secret = request.headers.get("X-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(supplied_secret, self.webhook_secret):
            logger.warning("Zalo: rejected webhook request with invalid secret")
            return web.json_response({"ok": False, "error": "forbidden"}, status=403)

        try:
            payload = await request.json()
        except Exception:
            logger.warning("Zalo: rejected webhook request with invalid JSON")
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        updates = self._extract_updates(payload)
        if not updates:
            logger.debug("Zalo: webhook request contained no supported updates")
            return web.json_response({"ok": True, "processed": 0})

        processed = 0
        for update in updates:
            try:
                await self._handle_update(update)
                processed += 1
            except Exception:
                logger.exception("Zalo: webhook update handling failed")
                return web.json_response(
                    {"ok": False, "error": "handler_failed"},
                    status=500,
                )

        return web.json_response({"ok": True, "processed": processed})

    def _extract_updates(self, payload: Any) -> list[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            for key in ("updates", "messages", "events"):
                nested = result.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [result]
        for key in ("updates", "messages", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(key in payload for key in TOP_LEVEL_UPDATE_KEYS):
            return [payload]
        return []

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        dedup_key = self._dedup_key(update)
        if dedup_key in self._seen_set:
            return
        self._remember(dedup_key)

        message = self._message_payload(update)
        if self._is_self_message(message):
            logger.debug("Zalo: ignoring self message")
            return

        chat_id = self._extract_chat_id(update, message)
        if not chat_id:
            logger.debug("Zalo: skipped update without chat_id")
            return

        user_id = self._extract_user_id(update, message) or chat_id
        if not self._allowed_source(str(user_id), str(chat_id)):
            logger.info("Zalo: rejecting unauthorized user_id=%s chat_id=%s", user_id, chat_id)
            return

        chat_type = self._chat_type(update, message)
        if self.dm_only and chat_type != "dm":
            logger.debug("Zalo: ignoring %s chat because ZALO_DM_ONLY=true chat_id=%s", chat_type, chat_id)
            return

        text, message_type, media_urls, media_types = await self._event_content(update, message)
        intake_text = self._pop_url_intake_text(str(chat_id))
        if intake_text:
            text = f"{text}\n\n{intake_text}".strip()
        if not text and not media_urls:
            logger.info(
                "Zalo: skipped update without text or supported media event_name=%s keys=%s message_keys=%s",
                _safe_str(self._event_name(update, message)),
                sorted(str(key) for key in update.keys())[:16],
                sorted(str(key) for key in message.keys())[:16],
            )
            return
        user_name = self._extract_user_name(update, message)

        source = self.build_source(
            chat_id=str(chat_id),
            chat_type=chat_type,
            user_id=str(user_id),
            user_name=user_name,
            message_id=str(self._extract_message_id(update) or ""),
        )
        logger.info(
            "Zalo: received %s message from chat_id=%s user_id=%s",
            chat_type,
            chat_id,
            user_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else message_type,
            source=source,
            raw_message=update,
            message_id=str(self._extract_message_id(update) or dedup_key),
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(event)

    def _message_payload(self, update: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("message", "event", "data"):
            value = update.get(key)
            if isinstance(value, dict):
                return value
        return update

    def _event_name(self, update: Dict[str, Any], message: Optional[Dict[str, Any]] = None) -> str:
        value = _first_from(
            (update, message or {}),
            "event_name",
            "message.event_name",
            "event.event_name",
            "data.event_name",
            "result.event_name",
        )
        return str(value or "").strip()

    def _allowed_source(self, user_id: str, chat_id: str) -> bool:
        if self.allow_all or not self.allowed_users:
            return True
        return user_id in self.allowed_users or chat_id in self.allowed_users

    async def _event_content(
        self,
        update: Dict[str, Any],
        message: Dict[str, Any],
    ) -> tuple[str, MessageType, list[str], list[str]]:
        event_name = self._event_name(update, message)
        text = self._extract_text(update, message)
        media_urls: list[str] = []
        media_types: list[str] = []
        message_type = MessageType.TEXT

        payloads = (message, update)
        photo_url = str(_first_from(payloads, *PHOTO_URL_PATHS) or "")
        voice_url = str(_first_from(payloads, *VOICE_URL_PATHS) or "")
        sticker_url = str(_first_from(payloads, *STICKER_URL_PATHS) or "")
        document_url = str(_first_from(payloads, *DOCUMENT_URL_PATHS) or "")
        caption = str(_first_from(payloads, *CAPTION_PATHS) or "").strip()

        if not text and (event_name == EVENT_LINK or _link_preview_text(message, update)):
            text = _link_preview_text(message, update)

        if event_name == EVENT_IMAGE or photo_url:
            message_type = MessageType.PHOTO
            text = caption or text or "[Zalo image]"
            if photo_url:
                cached = await self._cache_media_url(photo_url, kind="image")
                if cached:
                    media_urls.append(cached)
                    media_types.append("image/jpeg")
                else:
                    text = f"{text}\nImage URL: {photo_url}".strip()
        elif event_name == EVENT_VOICE or voice_url:
            message_type = MessageType.VOICE
            text = text or "[Zalo voice message]"
            if voice_url:
                cached = await self._cache_media_url(voice_url, kind="audio")
                if cached:
                    media_urls.append(cached)
                    media_types.append("audio/mpeg")
                else:
                    text = f"{text}\nVoice URL: {voice_url}".strip()
        elif event_name == EVENT_STICKER or sticker_url or _first(message, "sticker"):
            message_type = MessageType.STICKER
            sticker = _safe_str(_first(message, "sticker") or "")
            text = text or f"[Zalo sticker{': ' + sticker if sticker else ''}]"
            if sticker_url:
                text = f"{text}\nSticker URL: {sticker_url}".strip()
        elif document_url:
            message_type = MessageType.DOCUMENT
            text = text or "Zalo file/link"
            text = f"{text}\nFile URL: {document_url}".strip()
        elif event_name == EVENT_UNSUPPORTED:
            link_text = _link_preview_text(message, update)
            if not text and not link_text:
                logger.info(
                    "Zalo: unsupported message had no extractable text event_name=%s keys=%s message_keys=%s",
                    _safe_str(event_name),
                    sorted(str(key) for key in update.keys())[:16],
                    sorted(str(key) for key in message.keys())[:16],
                )
            chat_id = self._extract_chat_id(update, message) or ""
            intake_link = self._url_intake_link(str(chat_id))
            intake_instruction = (
                f"Ask the user to open this intake page and paste the URL there: {intake_link}. "
                "After submitting, they should return to Zalo and send a short message like "
                "`da gui link`; the next inbound message will include the submitted URL."
                if intake_link
                else (
                    "Ask the user to send the link as broken plain text, for example "
                    "`drive . google . com / drive / folders / FOLDER_ID` or "
                    "`docs . google . com / spreadsheets / d / SHEET_ID / edit`; "
                    "the adapter can reconstruct that when Zalo delivers it as text."
                )
            )
            text = (
                text
                or link_text
                or (
                    "[Zalo unsupported message: Zalo did not deliver any text, URL, "
                    "or media content to the bot for this event. Reply in the user's "
                    "language. Explain briefly that this exact Zalo message cannot be "
                    "read because Zalo sent it as an unsupported/no-content event. "
                    "Do not ask the user to paste the same normal full URL again, "
                    "because Zalo may keep converting it into an unreadable link "
                    f"preview/card. {intake_instruction} "
                    "For other sources, ask for raw text or a short non-URL identifier "
                    "that can be reconstructed. Do not ask for unrelated website brief "
                    "details until the source is readable.]"
                )
            )

        return text.strip(), message_type, media_urls, media_types

    def _extract_text(self, update: Dict[str, Any], message: Dict[str, Any]) -> str:
        value = _first_from((message, update), *TEXT_PATHS)
        text = str(value or "").strip()
        reconstructed = _reconstruct_obfuscated_google_url(text)
        if reconstructed and reconstructed not in text:
            return f"{text}\n\nReconstructed URL: {reconstructed}"
        return text

    def _extract_chat_id(self, update: Dict[str, Any], message: Dict[str, Any]) -> Optional[str]:
        value = _first_from((message, update), *CHAT_ID_PATHS)
        return str(value) if value not in (None, "") else None

    def _extract_user_id(self, update: Dict[str, Any], message: Dict[str, Any]) -> Optional[str]:
        value = _first_from((message, update), *USER_ID_PATHS)
        return str(value) if value not in (None, "") else None

    def _extract_user_name(self, update: Dict[str, Any], message: Dict[str, Any]) -> Optional[str]:
        value = _first_from((message, update), *USER_NAME_PATHS)
        return str(value) if value not in (None, "") else None

    def _chat_type(self, update: Dict[str, Any], message: Dict[str, Any]) -> str:
        raw = str(
            _first(message, "chat.chat_type", "chat.type")
            or _first(update, "chat.chat_type", "chat.type", "message.chat.chat_type")
            or "PRIVATE"
        ).strip().upper()
        if raw in {"GROUP", "GROUP_CHAT", "GROUPCHAT"}:
            return "group"
        if raw in {"CHANNEL"}:
            return "channel"
        return "dm"

    def _is_self_message(self, message: Dict[str, Any]) -> bool:
        sender_id = _first(message, "from.id", "sender.id", "user.id")
        if sender_id and self._bot_id and str(sender_id) == str(self._bot_id):
            return True
        is_bot = _first(message, "from.is_bot", "sender.is_bot", "user.is_bot")
        if isinstance(is_bot, bool):
            return is_bot
        return _truthy(is_bot)

    async def _cache_media_url(self, url: str, *, kind: str) -> Optional[str]:
        if not _looks_public_http_url(url):
            return None
        try:
            if kind == "image":
                return await cache_image_from_url(url)
            if kind == "audio":
                return await cache_audio_from_url(url)
        except Exception as exc:
            logger.debug("Zalo: failed to cache %s media URL: %s", kind, exc)
        return None

    def _url_intake_link(self, chat_id: str) -> str:
        if not self.url_intake_public_base or not chat_id:
            return ""
        return f"{self.url_intake_public_base}/i?chat_id={quote(str(chat_id), safe='')}"

    def _pop_url_intake_text(self, chat_id: str) -> str:
        if not self.url_intake_pending_file or not chat_id:
            return ""
        path = self.url_intake_pending_file
        try:
            if not os.path.exists(path):
                return ""
            lock_path = f"{path}.lock"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(lock_path, "a+", encoding="utf-8") as lock_handle:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                except Exception:
                    fcntl = None  # type: ignore[assignment]
                try:
                    with open(path, "r", encoding="utf-8") as pending_handle:
                        pending = json.load(pending_handle)
                    if not isinstance(pending, dict):
                        return ""
                    entries = pending.pop(str(chat_id), [])
                    with open(path, "w", encoding="utf-8") as pending_handle:
                        json.dump(pending, pending_handle, ensure_ascii=False, indent=2)
                finally:
                    try:
                        if fcntl is not None:  # type: ignore[name-defined]
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
            if not isinstance(entries, list) or not entries:
                return ""
            lines = ["[Zalo URL intake submissions for this chat]"]
            for idx, entry in enumerate(entries[-10:], 1):
                if not isinstance(entry, dict):
                    continue
                url = _safe_str(entry.get("url"), limit=4096)
                note = _safe_str(entry.get("note"), limit=1000)
                if not url:
                    continue
                lines.append(f"{idx}. URL: {url}")
                if note:
                    lines.append(f"   Note: {note}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as exc:
            logger.warning("Zalo: failed to read URL intake submissions: %s", exc)
            return ""

    def _extract_message_id(self, payload: Any) -> Optional[str]:
        if isinstance(payload, dict):
            value = _first(
                payload,
                "message_id",
                "msg_id",
                "id",
                "result.message_id",
                "result.msg_id",
                "result.id",
                "message.message_id",
                "message.id",
                "message.mid",
            )
            return str(value) if value not in (None, "") else None
        return None

    def _dedup_key(self, update: Dict[str, Any]) -> str:
        message = self._message_payload(update)
        event_name = self._event_name(update, message) or "unknown"
        chat_id = self._extract_chat_id(update, message) or "unknown"
        msg_id = self._extract_message_id(update)
        if msg_id:
            return f"msg:{event_name}:{chat_id}:{msg_id}"
        update_id = _first(update, "update_id", "id", "event_id")
        if update_id not in (None, ""):
            return f"update:{event_name}:{chat_id}:{update_id}"
        return f"hash:{hashlib.sha256(repr(update).encode()).hexdigest()[:24]}"

    def _remember(self, key: str) -> None:
        if self._seen.maxlen is not None and len(self._seen) >= self._seen.maxlen:
            old = self._seen.popleft()
            self._seen_set.discard(old)
        self._seen.append(key)
        self._seen_set.add(key)

    def _config_bool(self, extra_key: str, env_key: str, default: bool) -> bool:
        raw = os.getenv(env_key)
        if raw is None:
            extra = getattr(self.config, "extra", {}) or {}
            raw = extra.get(extra_key)
        if raw is None:
            return default
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
            return default
        return bool(raw)


def check_requirements() -> bool:
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token"))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    if not os.getenv("ZALO_BOT_TOKEN"):
        return None
    seeded: Dict[str, Any] = {"bot_token": os.environ["ZALO_BOT_TOKEN"]}
    if os.getenv("ZALO_HOME_CHANNEL"):
        seeded["home_channel"] = {
            "chat_id": os.environ["ZALO_HOME_CHANNEL"],
            "name": os.getenv("ZALO_HOME_CHANNEL_NAME", "Zalo Home"),
        }
    if os.getenv("ZALO_API_BASE"):
        seeded["api_base"] = os.environ["ZALO_API_BASE"].strip()
    if os.getenv("ZALO_ALLOWED_USERS"):
        seeded["allowed_users"] = sorted(_csv(os.environ["ZALO_ALLOWED_USERS"]))
    if os.getenv("ZALO_ALLOW_ALL_USERS"):
        seeded["allow_all_users"] = _truthy(os.environ["ZALO_ALLOW_ALL_USERS"])
    if os.getenv("ZALO_DM_ONLY"):
        seeded["dm_only"] = _truthy(os.environ["ZALO_DM_ONLY"])
    if os.getenv("ZALO_POLL_TIMEOUT_SECONDS"):
        try:
            seeded["poll_timeout_seconds"] = int(os.environ["ZALO_POLL_TIMEOUT_SECONDS"])
        except ValueError:
            pass
    if os.getenv("ZALO_POLL_INTERVAL_SECONDS"):
        try:
            seeded["poll_interval_seconds"] = float(os.environ["ZALO_POLL_INTERVAL_SECONDS"])
        except ValueError:
            pass
    if os.getenv("ZALO_CONNECTION_MODE"):
        seeded["connection_mode"] = os.environ["ZALO_CONNECTION_MODE"].strip().lower()
    if os.getenv("ZALO_PARSE_MODE"):
        parse_mode = os.environ["ZALO_PARSE_MODE"].strip().lower()
        if parse_mode in {"markdown", "html", ""}:
            seeded["parse_mode"] = parse_mode
    if os.getenv("ZALO_SUPPRESS_NOISY_STATUS"):
        seeded["suppress_noisy_status"] = _truthy(os.environ["ZALO_SUPPRESS_NOISY_STATUS"])
    webhook_url = os.getenv("ZALO_WEBHOOK_URL") or os.getenv("ZALO_WEBHOOK_PUBLIC_URL")
    if webhook_url:
        seeded["webhook_url"] = webhook_url.strip()
    if os.getenv("ZALO_WEBHOOK_SECRET"):
        seeded["webhook_secret"] = os.environ["ZALO_WEBHOOK_SECRET"].strip()
    if os.getenv("ZALO_WEBHOOK_PATH"):
        seeded["webhook_path"] = os.environ["ZALO_WEBHOOK_PATH"].strip()
    if os.getenv("ZALO_WEBHOOK_HOST"):
        seeded["webhook_host"] = os.environ["ZALO_WEBHOOK_HOST"].strip()
    if os.getenv("ZALO_WEBHOOK_PORT"):
        try:
            seeded["webhook_port"] = int(os.environ["ZALO_WEBHOOK_PORT"])
        except ValueError:
            pass
    if os.getenv("ZALO_WEBHOOK_AUTO_REGISTER"):
        seeded["webhook_auto_register"] = _truthy(os.environ["ZALO_WEBHOOK_AUTO_REGISTER"])
    if os.getenv("ZALO_DELETE_WEBHOOK_ON_POLLING_START"):
        seeded["delete_webhook_on_polling_start"] = _truthy(
            os.environ["ZALO_DELETE_WEBHOOK_ON_POLLING_START"]
        )
    if os.getenv("ZALO_DELETE_WEBHOOK_ON_DISCONNECT"):
        seeded["delete_webhook_on_disconnect"] = _truthy(
            os.environ["ZALO_DELETE_WEBHOOK_ON_DISCONNECT"]
        )
    if os.getenv("ZALO_URL_INTAKE_PUBLIC_BASE"):
        seeded["url_intake_public_base"] = os.environ["ZALO_URL_INTAKE_PUBLIC_BASE"].strip()
    if os.getenv("ZALO_URL_INTAKE_PENDING_FILE"):
        seeded["url_intake_pending_file"] = os.environ["ZALO_URL_INTAKE_PENDING_FILE"].strip()
    return seeded


def _apply_yaml_config(yaml_cfg: dict, zalo_cfg: dict) -> dict | None:
    """Translate non-secret Zalo config.yaml keys into PlatformConfig.extra.

    Mirrors the plugin-owned YAML bridge used by Telegram and other platform
    adapters. Credentials stay env-only: keep ZALO_BOT_TOKEN and
    ZALO_WEBHOOK_SECRET out of config.yaml so profiles can be shared safely.
    Environment variables still take precedence later in _apply_env_overrides.
    """
    extras: Dict[str, Any] = {}

    passthrough_keys = (
        "api_base",
        "allow_all_users",
        "dm_only",
        "private_only",
        "poll_timeout_seconds",
        "poll_interval_seconds",
        "connection_mode",
        "parse_mode",
        "suppress_noisy_status",
        "webhook_url",
        "webhook_public_url",
        "webhook_path",
        "webhook_host",
        "webhook_port",
        "webhook_auto_register",
        "delete_webhook_on_polling_start",
        "delete_webhook_on_disconnect",
        "url_intake_public_base",
        "url_intake_pending_file",
    )
    for key in passthrough_keys:
        if key in zalo_cfg and zalo_cfg[key] is not None:
            extras[key] = zalo_cfg[key]

    allowed_users: set[str] = set()
    for key in ("allowed_users", "allow_from", "allowed_chats"):
        allowed_users.update(_yaml_string_list(zalo_cfg.get(key)))
    if allowed_users:
        extras["allowed_users"] = sorted(allowed_users)

    # Intentionally do not bridge token/webhook_secret from YAML. The adapter
    # still reads those from env through _env_enablement, and explicit secrets
    # should not become part of profile config files by accident.
    return extras or None


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    extra = getattr(pconfig, "extra", {}) or {}
    token = os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token", "")
    if not token or not chat_id:
        return {"error": "Zalo standalone send: missing token or chat_id"}

    async with httpx.AsyncClient(timeout=20) as client:
        url = f"{ZALO_API_BASE}/bot{token}/sendMessage"
        try:
            parse_mode = (
                os.getenv("ZALO_PARSE_MODE")
                or extra.get("parse_mode")
                or DEFAULT_PARSE_MODE
            ).strip().lower()
            if parse_mode not in {"markdown", "html"}:
                parse_mode = ""
            message_ids: list[str] = []
            raw_response: Any = None
            for chunk in BasePlatformAdapter.truncate_message(
                message or " ",
                ZALO_MAX_MESSAGE_LENGTH,
            ) or [" "]:
                payload: Dict[str, Any] = {"chat_id": str(chat_id), "text": chunk or " "}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                response = await client.post(url, json=payload)
                response.raise_for_status()
                raw_response = response.json()
                if isinstance(raw_response, dict) and raw_response.get("ok") is False:
                    return {
                        "error": str(raw_response.get("description") or raw_response),
                        "raw_response": raw_response,
                    }
                msg_id = _first(raw_response, "result.message_id", "message_id")
                if msg_id:
                    message_ids.append(str(msg_id))
            return {
                "success": True,
                "platform": "zalo",
                "chat_id": str(chat_id),
                "message_id": message_ids[-1] if message_ids else None,
                "raw_response": raw_response,
            }
        except Exception as exc:
            return {"error": str(exc)}


def register(ctx) -> None:
    ctx.register_platform(
        name="zalo",
        label="Zalo",
        adapter_factory=lambda cfg: ZaloAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["ZALO_BOT_TOKEN"],
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="ZALO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="ZALO_ALLOWED_USERS",
        allow_all_env="ZALO_ALLOW_ALL_USERS",
        max_message_length=ZALO_MAX_MESSAGE_LENGTH,
        emoji="Z",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting with the user through Zalo Bot Platform. "
            "Use the user's chosen language. Keep replies concise because "
            "Zalo text messages are limited to 2000 characters. Do not rely "
            "on Markdown rendering or native slash-command menus. Users often "
            "chat naturally without commands; photos, voice notes, and public "
            "links may be attached when the platform provides accessible URLs."
        ),
    )
