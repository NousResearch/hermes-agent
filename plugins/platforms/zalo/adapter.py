"""Official Zalo Bot Platform adapter for Hermes Agent.

The adapter follows Zalo's documented Bot API contract. Long polling is aimed
at local development; authenticated HTTPS webhooks are the production path.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

try:
    from aiohttp import web
except Exception:  # pragma: no cover - messaging extra is optional
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
DEFAULT_POLL_TIMEOUT_SECONDS = 30
POLL_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0)
MAX_BACKOFF_JITTER_RATIO = 0.25
DEFAULT_PARSE_MODE = "markdown"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 18787
DEFAULT_WEBHOOK_PATH = "/zalo/webhook"
WEBHOOK_BODY_MAX_BYTES = 1_048_576
WEBHOOK_SECRET_MIN_LENGTH = 8
WEBHOOK_SECRET_MAX_LENGTH = 256

# Zalo's published error table currently defines 400, 401, 403, 404, 408,
# and 429. HTTP 5xx responses are included defensively for edge/proxy errors.
RETRYABLE_ERROR_CODES = {403, 408, 429, 500, 502, 503, 504}
FATAL_POLL_ERROR_CODES = {400, 401, 404}

EVENT_TEXT = "message.text.received"
EVENT_IMAGE = "message.image.received"
EVENT_STICKER = "message.sticker.received"
EVENT_VOICE = "message.voice.received"
EVENT_UNSUPPORTED = "message.unsupported.received"
SUPPORTED_EVENTS = {
    EVENT_TEXT,
    EVENT_IMAGE,
    EVENT_STICKER,
    EVENT_VOICE,
    EVENT_UNSUPPORTED,
}


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
    """Return the first non-empty value at one of a few explicit paths."""
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


def _looks_public_http_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_https_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme == "https" and bool(parsed.netloc)


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(minimum, parsed)
    return min(maximum, parsed) if maximum is not None else parsed


@dataclass
class ZaloApiError(Exception):
    method: str
    error_code: Optional[int]
    description: str
    payload: Any = None

    @property
    def retryable(self) -> bool:
        return self.error_code is None or self.error_code in RETRYABLE_ERROR_CODES

    @property
    def fatal_for_polling(self) -> bool:
        return self.error_code in FATAL_POLL_ERROR_CODES

    def __str__(self) -> str:
        code = f" error_code={self.error_code}" if self.error_code is not None else ""
        return f"{self.method} failed{code}: {self.description}"


class ZaloAdapter(BasePlatformAdapter):
    """Zalo Bot API transport using the standard Hermes platform interface."""

    supports_code_blocks = False
    splits_long_messages = True

    def __init__(self, config, **kwargs):
        _ = kwargs
        super().__init__(config=config, platform=Platform("zalo"))
        extra = getattr(config, "extra", {}) or {}

        self.bot_token = str(os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token") or "")
        self.parse_mode = str(extra.get("parse_mode", DEFAULT_PARSE_MODE)).strip().lower()
        if self.parse_mode not in {"", "markdown", "html"}:
            logger.warning("Zalo: unsupported parse_mode=%r; sending plain text", self.parse_mode)
            self.parse_mode = ""

        self.poll_timeout_seconds = _bounded_int(
            extra.get("poll_timeout_seconds", DEFAULT_POLL_TIMEOUT_SECONDS),
            DEFAULT_POLL_TIMEOUT_SECONDS,
            minimum=1,
        )
        self.connection_mode = str(extra.get("connection_mode") or "auto").strip().lower()
        if self.connection_mode not in {"auto", "polling", "webhook"}:
            logger.warning(
                "Zalo: unsupported connection_mode=%r; using auto",
                self.connection_mode,
            )
            self.connection_mode = "auto"

        self.webhook_url = str(extra.get("webhook_url") or "").strip()
        self.webhook_secret = str(os.getenv("ZALO_WEBHOOK_SECRET") or "").strip()
        self.webhook_path = str(extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH).strip()
        if not self.webhook_path.startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self.webhook_host = str(extra.get("webhook_host") or DEFAULT_WEBHOOK_HOST).strip()
        self.webhook_port = _bounded_int(
            extra.get("webhook_port", DEFAULT_WEBHOOK_PORT),
            DEFAULT_WEBHOOK_PORT,
            minimum=1,
            maximum=65535,
        )
        self.group_policy = str(extra.get("group_policy") or "open").strip().lower()

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
        return f"{ZALO_API_BASE}/bot{self.bot_token}"

    @property
    def _uses_webhook(self) -> bool:
        if self.connection_mode == "polling":
            return False
        if self.connection_mode == "webhook":
            return True
        return bool(self.webhook_url or self.webhook_secret)

    def _validate_transport_config(self) -> Optional[str]:
        if not self._uses_webhook:
            return None
        if not self.webhook_url or not self.webhook_secret:
            return "webhook mode requires platforms.zalo.webhook_url and ZALO_WEBHOOK_SECRET"
        if not _looks_https_url(self.webhook_url):
            return "platforms.zalo.webhook_url must be a public HTTPS URL"
        if not WEBHOOK_SECRET_MIN_LENGTH <= len(self.webhook_secret) <= WEBHOOK_SECRET_MAX_LENGTH:
            return "ZALO_WEBHOOK_SECRET must be 8-256 characters"
        return None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        _ = is_reconnect
        if not self.bot_token:
            self._set_fatal_error("config_missing", "ZALO_BOT_TOKEN must be set", retryable=False)
            return False

        config_error = self._validate_transport_config()
        if config_error:
            self._set_fatal_error("config_invalid", config_error, retryable=False)
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
            logger.debug("Zalo: scoped lock unavailable", exc_info=True)

        self._client = httpx.AsyncClient(
            timeout=self.poll_timeout_seconds + 10,
            trust_env=True,
        )
        try:
            response = await self._api("getMe", {})
            result = response.get("result") if isinstance(response, dict) else None
            if isinstance(result, dict) and result.get("id"):
                self._bot_id = str(result["id"])

            if self._uses_webhook:
                if not await self._start_webhook_server():
                    await self.disconnect()
                    return False
                await self._api(
                    "setWebhook",
                    {"url": self.webhook_url, "secret_token": self.webhook_secret},
                )
                logger.info(
                    "Zalo: webhook listening on %s:%d%s (public URL %s)",
                    self.webhook_host,
                    self.webhook_port,
                    self.webhook_path,
                    self.webhook_url,
                )
            else:
                # Zalo documents polling and webhook delivery as mutually
                # exclusive. Selecting polling therefore clears stale webhook
                # state before the first getUpdates request.
                await self._api("deleteWebhook", {})
                self._poll_task = asyncio.create_task(self._poll_loop(), name="zalo-poll")
                logger.info("Zalo: long polling started")
        except Exception as exc:
            retryable = isinstance(exc, ZaloApiError) and exc.retryable
            self._set_fatal_error("connect_failed", f"Zalo startup failed: {exc}", retryable=retryable)
            await self.disconnect()
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Zalo: poll task ended with error", exc_info=True)
            self._poll_task = None

        if self._web_runner is not None:
            try:
                await self._web_runner.cleanup()
            except Exception:
                logger.debug("Zalo: webhook cleanup failed", exc_info=True)
            self._web_runner = None
            self._web_site = None

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                logger.debug("Zalo: HTTP client cleanup failed", exc_info=True)
            self._client = None

        if self._lock_key:
            try:
                from gateway.status import release_scoped_lock

                release_scoped_lock("zalo", self._lock_key)
            except Exception:
                logger.debug("Zalo: scoped lock release failed", exc_info=True)
            self._lock_key = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        _ = reply_to, metadata
        if not chat_id:
            return SendResult(success=False, error="missing chat_id")

        chunks = self.truncate_message(content or " ", ZALO_MAX_MESSAGE_LENGTH) or [" "]
        message_ids: list[str] = []
        raw_response: Any = None
        for chunk in chunks:
            payload: Dict[str, Any] = {"chat_id": str(chat_id), "text": chunk or " "}
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
            if message_id := self._extract_message_id(raw_response):
                message_ids.append(message_id)

        return SendResult(
            success=True,
            message_id=message_ids[-1] if message_ids else None,
            raw_response=raw_response,
            continuation_message_ids=tuple(message_ids[:-1]),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        _ = metadata
        try:
            await self._api("sendChatAction", {"chat_id": str(chat_id), "action": "typing"})
        except Exception:
            # Typing is best-effort and must never fail message processing.
            logger.debug("Zalo: sendChatAction failed", exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"id": str(chat_id), "name": str(chat_id), "type": "unknown"}

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
            raise ZaloApiError(method, None, str(exc)) from exc

        if response.status_code >= 400:
            description = response.text[:500] if response.text else response.reason_phrase
            raise ZaloApiError(
                method,
                response.status_code,
                description,
                {"status_code": response.status_code, "body": response.text},
            )
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise ZaloApiError(method, None, "invalid JSON response") from exc

        if isinstance(data, dict) and data.get("ok") is False:
            raw_code = data.get("error_code")
            try:
                error_code = int(raw_code) if raw_code is not None else None
            except (TypeError, ValueError):
                error_code = None
            raise ZaloApiError(
                method,
                error_code,
                str(data.get("description") or "unknown Zalo API error"),
                data,
            )
        return data

    async def _poll_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                payload = await self._api(
                    "getUpdates",
                    {"timeout": str(self.poll_timeout_seconds)},
                )
                for update in self._extract_updates(payload):
                    await self._handle_update(update)
                backoff_idx = 0
                # Real getUpdates calls block, but yielding here also keeps the
                # loop cancellable under fast test doubles and proxy failures
                # that return immediately.
                await asyncio.sleep(0)
            except ZaloApiError as exc:
                if exc.fatal_for_polling:
                    self._set_fatal_error("api_error", str(exc), retryable=False)
                    self._mark_disconnected()
                    return
                delay = self._poll_backoff_sleep(backoff_idx)
                logger.warning("Zalo: polling failed: %s; retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                backoff_idx = min(backoff_idx + 1, len(POLL_BACKOFF_SECONDS) - 1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delay = self._poll_backoff_sleep(backoff_idx)
                logger.warning("Zalo: polling failed: %s; retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                backoff_idx = min(backoff_idx + 1, len(POLL_BACKOFF_SECONDS) - 1)

    def _poll_backoff_sleep(self, backoff_idx: int) -> float:
        cap = POLL_BACKOFF_SECONDS[min(backoff_idx, len(POLL_BACKOFF_SECONDS) - 1)]
        return cap + (cap * MAX_BACKOFF_JITTER_RATIO * random.random())

    async def _start_webhook_server(self) -> bool:
        if web is None:
            self._set_fatal_error(
                "dependency_missing",
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
                f"Zalo webhook could not listen on {self.webhook_host}:{self.webhook_port}: {exc}",
                retryable=True,
            )
            if self._web_runner is not None:
                await self._web_runner.cleanup()
            self._web_runner = None
            self._web_site = None
            return False
        return True

    async def _handle_webhook_health(self, request: Any) -> Any:
        _ = request
        return web.Response(text="ok") if web is not None else None

    async def _handle_webhook_request(self, request: Any) -> Any:
        if web is None:
            return None

        supplied = request.headers.get("X-Bot-Api-Secret-Token", "")
        # Raw headers are untrusted Unicode. compare_digest raises TypeError for
        # non-ASCII str input, so compare UTF-8 bytes like the sibling adapters.
        if not hmac.compare_digest(supplied.encode(), self.webhook_secret.encode()):
            logger.warning("Zalo: rejected webhook request with invalid secret")
            return web.json_response({"ok": False, "error": "forbidden"}, status=403)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        updates = self._extract_updates(payload)
        for update in updates:
            try:
                await self._handle_update(update)
            except Exception:
                logger.exception("Zalo: webhook update handling failed")
                return web.json_response({"ok": False, "error": "handler_failed"}, status=500)
        return web.json_response({"ok": True, "processed": len(updates)})

    @staticmethod
    def _extract_updates(payload: Any) -> list[Dict[str, Any]]:
        """Normalize only documented Zalo envelopes plus direct test fixtures."""
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            return [result]
        if isinstance(payload.get("message"), dict) and payload.get("event_name"):
            return [payload]
        return []

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        event_name = str(update.get("event_name") or "").strip()
        if event_name not in SUPPORTED_EVENTS:
            logger.debug("Zalo: ignored unknown event %r", event_name)
            return

        dedup_key = self._dedup_key(update)
        if dedup_key in self._seen_set:
            return
        self._remember(dedup_key)

        message = update.get("message")
        if not isinstance(message, dict) or self._is_self_message(message):
            return
        chat_id = self._extract_chat_id(message)
        user_id = self._extract_user_id(message)
        if not chat_id or not user_id:
            logger.debug("Zalo: skipped event without documented chat/from identity")
            return

        chat_type = self._chat_type(message)
        if chat_type == "group" and self.group_policy == "disabled":
            return

        text, message_type, media_urls, media_types = await self._event_content(
            event_name,
            message,
        )
        if not text and not media_urls:
            return

        message_id = str(message.get("message_id") or dedup_key)
        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            user_name=self._extract_user_name(message),
            message_id=message_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else message_type,
            source=source,
            raw_message=update,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(event)

    async def _event_content(
        self,
        event_name: str,
        message: Dict[str, Any],
    ) -> tuple[str, MessageType, list[str], list[str]]:
        text = str(message.get("text") or "").strip()
        media_urls: list[str] = []
        media_types: list[str] = []

        if event_name == EVENT_IMAGE:
            text = str(message.get("caption") or text or "[Zalo image]").strip()
            photo_url = str(message.get("photo") or "").strip()
            if cached := await self._cache_media_url(photo_url, kind="image"):
                media_urls.append(cached)
                media_types.append("image/jpeg")
            elif photo_url:
                text = f"{text}\nImage URL: {photo_url}"
            return text, MessageType.PHOTO, media_urls, media_types

        if event_name == EVENT_VOICE:
            text = text or "[Zalo voice message]"
            voice_url = str(message.get("voice_url") or "").strip()
            if cached := await self._cache_media_url(voice_url, kind="audio"):
                media_urls.append(cached)
                media_types.append("audio/mpeg")
            elif voice_url:
                text = f"{text}\nVoice URL: {voice_url}"
            return text, MessageType.VOICE, media_urls, media_types

        if event_name == EVENT_STICKER:
            sticker = str(message.get("sticker") or "").strip()
            sticker_url = str(message.get("url") or "").strip()
            text = text or f"[Zalo sticker{': ' + sticker if sticker else ''}]"
            if sticker_url:
                text = f"{text}\nSticker URL: {sticker_url}"
            return text, MessageType.STICKER, media_urls, media_types

        if event_name == EVENT_UNSUPPORTED:
            return (
                "[Zalo unsupported message: Zalo intentionally did not provide "
                "the original content. Ask the user to resend it as text, an image, "
                "or a voice message.]",
                MessageType.TEXT,
                media_urls,
                media_types,
            )

        return text, MessageType.TEXT, media_urls, media_types

    @staticmethod
    def _extract_chat_id(message: Dict[str, Any]) -> Optional[str]:
        value = _first(message, "chat.id")
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _extract_user_id(message: Dict[str, Any]) -> Optional[str]:
        value = _first(message, "from.id")
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _extract_user_name(message: Dict[str, Any]) -> Optional[str]:
        value = _first(message, "from.display_name")
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _chat_type(message: Dict[str, Any]) -> str:
        return "group" if str(_first(message, "chat.chat_type") or "PRIVATE").upper() == "GROUP" else "dm"

    def _is_self_message(self, message: Dict[str, Any]) -> bool:
        sender_id = _first(message, "from.id")
        if sender_id and self._bot_id and str(sender_id) == self._bot_id:
            return True
        return message.get("from", {}).get("is_bot") is True if isinstance(message.get("from"), dict) else False

    async def _cache_media_url(self, url: str, *, kind: str) -> Optional[str]:
        if not _looks_public_http_url(url):
            return None
        try:
            if kind == "image":
                return await cache_image_from_url(url)
            return await cache_audio_from_url(url)
        except Exception:
            logger.debug("Zalo: failed to cache inbound %s", kind, exc_info=True)
            return None

    @staticmethod
    def _extract_message_id(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        value = _first(payload, "result.message_id", "message.message_id", "message_id")
        return str(value) if value not in (None, "") else None

    def _dedup_key(self, update: Dict[str, Any]) -> str:
        raw_message = update.get("message")
        message: Dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        if message_id := message.get("message_id"):
            return f"message:{message_id}"
        canonical = json.dumps(update, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"hash:{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"

    def _remember(self, key: str) -> None:
        if self._seen.maxlen is not None and len(self._seen) >= self._seen.maxlen:
            self._seen_set.discard(self._seen.popleft())
        self._seen.append(key)
        self._seen_set.add(key)


def check_requirements() -> bool:
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token"))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    token = os.getenv("ZALO_BOT_TOKEN")
    if not token:
        return None
    seeded: Dict[str, Any] = {"bot_token": token}
    if home_channel := os.getenv("ZALO_HOME_CHANNEL"):
        seeded["home_channel"] = {
            "chat_id": home_channel,
            "name": os.getenv("ZALO_HOME_CHANNEL_NAME", "Zalo Home"),
        }
    return seeded


def _apply_yaml_config(yaml_cfg: dict, zalo_cfg: dict) -> dict | None:
    """Bridge Zalo YAML into plugin extras and central gateway auth hooks."""
    _ = yaml_cfg
    extras: Dict[str, Any] = {}
    for key in (
        "connection_mode",
        "parse_mode",
        "poll_timeout_seconds",
        "webhook_url",
        "webhook_path",
        "webhook_host",
        "webhook_port",
    ):
        if key in zalo_cfg and zalo_cfg[key] is not None:
            extras[key] = zalo_cfg[key]

    # The gateway is the single authorization authority. Bridge YAML
    # allowlists to the env contract registered below, preserving explicit env
    # precedence exactly like the mature platform plugins.
    allowed_users: set[str] = set()
    for key in ("allow_from", "group_allow_from"):
        allowed_users.update(_yaml_string_list(zalo_cfg.get(key)))
    if allowed_users and not os.getenv("ZALO_ALLOWED_USERS"):
        os.environ["ZALO_ALLOWED_USERS"] = ",".join(sorted(allowed_users))
    if "allow_all_users" in zalo_cfg and not os.getenv("ZALO_ALLOW_ALL_USERS"):
        os.environ["ZALO_ALLOW_ALL_USERS"] = "true" if _truthy(zalo_cfg["allow_all_users"]) else "false"

    home = zalo_cfg.get("home_channel")
    if isinstance(home, dict) and home.get("chat_id") and not os.getenv("ZALO_HOME_CHANNEL"):
        os.environ["ZALO_HOME_CHANNEL"] = str(home["chat_id"])
        if home.get("name"):
            os.environ["ZALO_HOME_CHANNEL_NAME"] = str(home["name"])

    # Bot token and webhook secret intentionally remain env-only credentials.
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
    _ = thread_id, media_files, force_document
    extra = getattr(pconfig, "extra", {}) or {}
    token = os.getenv("ZALO_BOT_TOKEN") or extra.get("bot_token", "")
    if not token or not chat_id:
        return {"error": "Zalo standalone send: missing token or chat_id"}

    parse_mode = str(extra.get("parse_mode", DEFAULT_PARSE_MODE)).strip().lower()
    if parse_mode not in {"markdown", "html"}:
        parse_mode = ""
    message_ids: list[str] = []
    raw_response: Any = None
    try:
        async with httpx.AsyncClient(timeout=20, trust_env=True) as client:
            for chunk in BasePlatformAdapter.truncate_message(
                message or " ",
                ZALO_MAX_MESSAGE_LENGTH,
            ) or [" "]:
                payload: Dict[str, Any] = {"chat_id": str(chat_id), "text": chunk or " "}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                response = await client.post(
                    f"{ZALO_API_BASE}/bot{token}/sendMessage",
                    json=payload,
                )
                response.raise_for_status()
                raw_response = response.json()
                if isinstance(raw_response, dict) and raw_response.get("ok") is False:
                    return {
                        "error": str(raw_response.get("description") or raw_response),
                        "raw_response": raw_response,
                    }
                if message_id := _first(raw_response, "result.message_id"):
                    message_ids.append(str(message_id))
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "success": True,
        "platform": "zalo",
        "chat_id": str(chat_id),
        "message_id": message_ids[-1] if message_ids else None,
        "raw_response": raw_response,
    }


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
            "Use the user's chosen language and keep replies concise because "
            "Zalo text messages are limited to 2000 characters. Zalo supports "
            "Markdown or HTML text, images, voice messages, stickers, and typing indicators."
        ),
    )
