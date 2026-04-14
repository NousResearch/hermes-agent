"""Nextcloud Talk gateway adapter.

Connects Hermes to a self-hosted Nextcloud Talk server as a webhook-based
bot using the official Talk Bot API.  The bot is registered on the
Nextcloud side with ``occ talk:bot:install`` and receives HMAC-signed
webhook callbacks for new chat messages; Hermes replies through the Talk
Bot message endpoint with the same signing scheme.

Because Talk conversation tokens identify rooms across every Nextcloud
instance that installs the bot, the adapter also caches the originating
backend URL per conversation so a single Hermes instance can serve Talk
rooms on multiple Nextcloud servers without per-server configuration.

Environment variables:
    NEXTCLOUD_TALK_SECRET          Bot shared secret (required)
    NEXTCLOUD_TALK_BASE_URL        Fallback Nextcloud base URL for
                                   outbound sends when no inbound webhook
                                   has established one yet
    NEXTCLOUD_TALK_WEBHOOK_HOST    Bind host for the webhook listener
                                   (default: 0.0.0.0)
    NEXTCLOUD_TALK_WEBHOOK_PORT    Bind port (default: 8645)
    NEXTCLOUD_TALK_WEBHOOK_PATH    Webhook path (default: /nextcloud-talk)
    NEXTCLOUD_TALK_CHAT_TYPE       Default chat type ("group" or "dm",
                                   default: group)
    NEXTCLOUD_TALK_HOME_CHANNEL    Talk conversation token for scheduled
                                   cron deliveries
    NEXTCLOUD_TALK_ALLOWED_USERS   Comma-separated Talk actor IDs allowed
                                   to interact (e.g. "users/alice")
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    from aiohttp import ClientSession, ClientTimeout, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    ClientSession = Any  # type: ignore[assignment]
    ClientTimeout = Any  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8645
DEFAULT_PATH = "/nextcloud-talk"
MAX_MESSAGE_LENGTH = 32_000

# Webhook security limits
_MAX_WEBHOOK_BODY_BYTES = 1 * 1024 * 1024  # 1 MB
_WEBHOOK_BODY_READ_TIMEOUT = 10  # seconds
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 120  # requests per window per IP

# Cache limits to prevent unbounded memory growth
_MAX_CHAT_CACHE_ENTRIES = 1024

# Talk conversation tokens are alphanumeric; reject path separators and
# URL-unsafe characters to prevent path traversal and injection.
_VALID_CHAT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def check_nextcloud_talk_requirements() -> bool:
    """Return True when the adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


def _normalize_base_url(value: str) -> str:
    """Normalize a configured Nextcloud base URL."""
    return (value or "").strip().rstrip("/")


def _validate_backend_url(url: str) -> bool:
    """Return True if *url* looks like a valid Nextcloud backend URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _sanitize_chat_id(chat_id: str) -> Optional[str]:
    """Return *chat_id* if it is a valid Talk conversation token, else None."""
    if _VALID_CHAT_ID_RE.match(chat_id):
        return chat_id
    return None


def _decode_talk_message(raw_content: Any) -> str:
    """Decode Talk's JSON-encoded ``object.content`` payload to plain text."""
    if raw_content is None:
        return ""
    if not isinstance(raw_content, str):
        return str(raw_content)

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content

    if isinstance(parsed, dict):
        message = parsed.get("message")
        if isinstance(message, str):
            return message
    return raw_content


def _sign_payload(secret_value: str, random_header: str, body: bytes) -> str:
    """Return the Talk bot HMAC signature for *body*."""
    mac = hmac.new(secret_value.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(random_header.encode("utf-8"))
    mac.update(body)
    return mac.hexdigest()


class _RateLimiter:
    """Simple sliding-window rate limiter keyed by remote IP."""

    _PRUNE_THRESHOLD = 4096  # prune stale keys when dict exceeds this size

    def __init__(self, window: int = _RATE_LIMIT_WINDOW, max_requests: int = _RATE_LIMIT_MAX):
        self._window = window
        self._max = max_requests
        self._hits: Dict[str, collections.deque] = {}

    def _prune_stale(self, now: float) -> None:
        """Remove keys whose windows have fully expired."""
        stale = [k for k, dq in self._hits.items() if not dq or dq[-1] < now - self._window]
        for k in stale:
            del self._hits[k]

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        # Periodic housekeeping to prevent unbounded growth from unique IPs
        if len(self._hits) > self._PRUNE_THRESHOLD:
            self._prune_stale(now)
        dq = self._hits.get(key)
        if dq is None:
            dq = collections.deque()
            self._hits[key] = dq
        # Expire old entries for this key
        while dq and dq[0] < now - self._window:
            dq.popleft()
        if len(dq) >= self._max:
            return False
        dq.append(now)
        return True


class NextcloudTalkAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for webhook-based Nextcloud Talk bots."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEXTCLOUD_TALK)
        self._secret: str = config.token or os.getenv("NEXTCLOUD_TALK_SECRET", "")
        self._base_url: str = _normalize_base_url(
            config.extra.get("base_url") or os.getenv("NEXTCLOUD_TALK_BASE_URL", "")
        )
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        raw_path = str(config.extra.get("path", DEFAULT_PATH) or DEFAULT_PATH).strip()
        self._path: str = raw_path if raw_path.startswith("/") else f"/{raw_path}"
        self._default_chat_type: str = str(
            config.extra.get("chat_type")
            or os.getenv("NEXTCLOUD_TALK_CHAT_TYPE", "group")
        ).strip().lower() or "group"
        if self._default_chat_type not in {"dm", "group"}:
            self._default_chat_type = "group"

        self._runner = None
        self._session: Optional[ClientSession] = None
        self._chat_backends: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._chat_info_cache: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self._rate_limiter = _RateLimiter()

    async def connect(self) -> bool:
        """Start the webhook HTTP server for Nextcloud Talk callbacks."""
        if not self._secret:
            logger.error("[nextcloud_talk] NEXTCLOUD_TALK_SECRET is required")
            return False

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_webhook)

        import socket as _socket

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error(
                "[nextcloud_talk] port %d already in use. Set NEXTCLOUD_TALK_WEBHOOK_PORT or platforms.nextcloud_talk.extra.port",
                self._port,
            )
            return False
        except (ConnectionRefusedError, OSError):
            pass

        self._session = ClientSession(timeout=ClientTimeout(total=30))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[nextcloud_talk] listening on %s:%d%s",
            self._host,
            self._port,
            self._path,
        )
        return True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[nextcloud_talk] disconnected")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_chat_info(self, chat_id: str, info: Dict[str, Any]) -> None:
        """Store chat info with LRU eviction."""
        self._chat_info_cache[chat_id] = info
        self._chat_info_cache.move_to_end(chat_id)
        while len(self._chat_info_cache) > _MAX_CHAT_CACHE_ENTRIES:
            self._chat_info_cache.popitem(last=False)

    def _cache_backend(self, chat_id: str, backend: str) -> None:
        """Store backend URL with LRU eviction."""
        self._chat_backends[chat_id] = backend
        self._chat_backends.move_to_end(chat_id)
        while len(self._chat_backends) > _MAX_CHAT_CACHE_ENTRIES:
            self._chat_backends.popitem(last=False)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a signed bot message into a Nextcloud Talk conversation."""
        if not content:
            return SendResult(success=True)

        if not self._secret:
            return SendResult(success=False, error="Nextcloud Talk not configured: missing secret")

        safe_chat_id = _sanitize_chat_id(str(chat_id))
        if not safe_chat_id:
            return SendResult(success=False, error=f"Invalid Talk conversation token: {chat_id!r}")

        backend = _normalize_base_url(
            (metadata or {}).get("backend_url")
            or self._chat_backends.get(safe_chat_id, "")
            or self._base_url
        )
        if not backend:
            return SendResult(
                success=False,
                error=(
                    "Nextcloud Talk backend URL is unknown for this conversation. "
                    "Set NEXTCLOUD_TALK_BASE_URL or wait for an inbound webhook to establish the backend."
                ),
            )
        if not _validate_backend_url(backend):
            return SendResult(success=False, error=f"Invalid backend URL scheme: {backend!r}")

        chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH)
        last_result = SendResult(success=True)
        session = self._session or ClientSession(timeout=ClientTimeout(total=30))
        session_is_temporary = self._session is None

        try:
            for index, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {"message": chunk}
                if index == 0 and reply_to is not None:
                    try:
                        payload["replyTo"] = int(reply_to)
                    except (TypeError, ValueError):
                        logger.debug("[nextcloud_talk] ignoring non-numeric reply_to=%r", reply_to)

                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                random_header = secrets.token_hex(32)
                # NC Talk Bot API signs: HMAC(secret, random + message_text), NOT random + json_body
                signature = _sign_payload(self._secret, random_header, chunk.encode("utf-8"))
                url = f"{backend}/ocs/v2.php/apps/spreed/api/v1/bot/{safe_chat_id}/message"
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "OCS-APIRequest": "true",
                    "X-Nextcloud-Talk-Bot-Random": random_header,
                    "X-Nextcloud-Talk-Bot-Signature": signature,
                }

                try:
                    async with session.post(url, data=body, headers=headers) as resp:
                        resp_text = await resp.text()
                        if resp.status not in (200, 201):
                            return SendResult(
                                success=False,
                                error=f"Nextcloud Talk API error ({resp.status}): {resp_text}",
                                retryable=resp.status >= 500,
                            )
                        raw_response: Any = resp_text
                        try:
                            raw_response = json.loads(resp_text) if resp_text else {}
                        except json.JSONDecodeError:
                            pass
                        last_result = SendResult(success=True, raw_response=raw_response)
                except Exception as exc:
                    return SendResult(
                        success=False,
                        error=f"Nextcloud Talk send failed: {exc}",
                        retryable=True,
                    )
        finally:
            if session_is_temporary:
                await session.close()

        return last_result

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Nextcloud Talk bots do not currently expose a typing API."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return self._chat_info_cache.get(
            str(chat_id),
            {"name": str(chat_id), "type": self._default_chat_type, "chat_id": str(chat_id)},
        )

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "nextcloud_talk"})

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        remote_ip = getattr(request, "remote", None) or "unknown"

        # Rate limiting
        if not self._rate_limiter.allow(remote_ip):
            logger.warning("[nextcloud_talk] rate limit exceeded for %s", remote_ip)
            return web.Response(status=429, text="Too Many Requests")

        # Content-Type guard — Talk always sends application/json
        content_type = (request.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            logger.warning("[nextcloud_talk] rejected Content-Type %r from %s", content_type, remote_ip)
            return web.Response(status=415, text="Unsupported Media Type")

        # Body size guard — reject early via Content-Length when available
        content_length = request.content_length
        if content_length is not None and content_length > _MAX_WEBHOOK_BODY_BYTES:
            logger.warning("[nextcloud_talk] body too large (%d bytes) from %s", content_length, remote_ip)
            return web.Response(status=413, text="Request body too large")

        # Read body with timeout to prevent slow-loris
        try:
            body = await asyncio.wait_for(request.read(), timeout=_WEBHOOK_BODY_READ_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[nextcloud_talk] body read timed out from %s", remote_ip)
            return web.Response(status=408, text="Request Timeout")
        except Exception as exc:
            logger.warning("[nextcloud_talk] body read failed from %s: %s", remote_ip, exc)
            return web.json_response({"error": "failed to read body"}, status=400)

        if len(body) > _MAX_WEBHOOK_BODY_BYTES:
            logger.warning("[nextcloud_talk] body exceeds limit (%d bytes) from %s", len(body), remote_ip)
            return web.Response(status=413, text="Request body too large")

        # Signature verification (timing-safe)
        signature = (request.headers.get("X-Nextcloud-Talk-Signature") or "").strip().lower()
        random_header = (request.headers.get("X-Nextcloud-Talk-Random") or "").strip()
        backend = _normalize_base_url(request.headers.get("X-Nextcloud-Talk-Backend", ""))

        if not signature or not random_header:
            return web.json_response({"error": "missing signature headers"}, status=401)

        expected = _sign_payload(self._secret, random_header, body)
        if not hmac.compare_digest(expected, signature):
            logger.warning("[nextcloud_talk] invalid signature from %s", remote_ip)
            return web.json_response({"error": "invalid signature"}, status=401)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.json_response({"error": "invalid json body"}, status=400)

        hook_type = str(payload.get("type") or "")
        if hook_type != "Create":
            logger.debug("[nextcloud_talk] ignoring unsupported hook type %s", hook_type)
            return web.json_response({"status": "ignored", "type": hook_type}, status=202)

        actor = payload.get("actor") or {}
        actor_type = str(actor.get("type") or "")
        actor_id = str(actor.get("id") or "")
        if actor_type.lower() == "application" or actor_id.startswith("bots/"):
            return web.json_response({"status": "ignored", "reason": "self_message"}, status=202)

        obj = payload.get("object") or {}
        target = payload.get("target") or {}
        raw_chat_id = str(target.get("id") or "")
        chat_id = _sanitize_chat_id(raw_chat_id)
        if not chat_id:
            logger.warning("[nextcloud_talk] invalid conversation token %r from %s", raw_chat_id, remote_ip)
            return web.json_response({"error": "invalid target.id"}, status=400)

        text = _decode_talk_message(obj.get("content"))
        source = self.build_source(
            chat_id=chat_id,
            chat_name=target.get("name") or chat_id,
            chat_type=self._default_chat_type,
            user_id=actor_id or None,
            user_name=actor.get("name"),
        )

        in_reply_to = obj.get("inReplyTo") or {}
        parent_object = in_reply_to.get("object") or {}
        parent_actor = in_reply_to.get("actor") or {}
        reply_to_text = _decode_talk_message(parent_object.get("content"))
        if reply_to_text and parent_actor.get("name"):
            reply_to_text = f"{parent_actor.get('name')}: {reply_to_text}"

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=str(obj.get("id") or "") or None,
            reply_to_message_id=str(parent_object.get("id") or "") or None,
            reply_to_text=reply_to_text or None,
        )

        self._cache_chat_info(chat_id, {
            "name": target.get("name") or chat_id,
            "type": self._default_chat_type,
            "chat_id": chat_id,
        })
        if backend and _validate_backend_url(backend):
            self._cache_backend(chat_id, backend)

        await self.handle_message(event)
        return web.json_response({"status": "accepted"}, status=202)
