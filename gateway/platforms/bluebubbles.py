"""BlueBubbles iMessage platform adapter.

Uses the local BlueBubbles macOS server for outbound REST sends and inbound
webhooks.  Supports text messaging, media attachments (images, voice, video,
documents), tapback reactions, typing indicators, and read receipts.

Architecture based on PR #5869 (benjaminsehl) with inbound attachment
downloading from PR #4588 (YuhangLin).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.reactions import (
    TapbackAction,
    TapbackDirection,
    TapbackOperation,
    TapbackStatus,
    TapbackType,
    TapbackValidationError,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes_async,
    cache_audio_from_bytes_async,
    cache_document_from_bytes_async,
)
from .media_cache import ext_for_mime
from .bluebubbles_attachments import (
    AttachmentReadiness,
    materialize_attachments,
    schedule_pending_attachment_retry,
)
from gateway.platforms.helpers import compile_mention_patterns, strip_markdown


# Historical BlueBubbles mime→ext maps, preserved verbatim as overrides for
# the shared dispatch in gateway.platforms.media_cache. Both maps are
# CLOSED: unlisted mimes fall back to .jpg / .mp3 (never mimetypes).
_BLUEBUBBLES_IMAGE_EXT_OVERRIDES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".jpg",  # preserves historical bluebubbles mapping
    "image/heif": ".jpg",  # preserves historical bluebubbles mapping
    "image/tiff": ".jpg",  # preserves historical bluebubbles mapping
}
_BLUEBUBBLES_AUDIO_EXT_OVERRIDES = {
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-caf": ".mp3",  # preserves historical bluebubbles mapping
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",  # preserves historical bluebubbles mapping (shared table says .aac)
}

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read with the default-profile startup fallback.

    Secondary profiles construct their adapters under a profile secret
    scope -- the scope is authoritative and a scoped miss returns ``default``
    (no cross-profile borrow from ``os.environ``, which may hold another
    profile's value). The DEFAULT profile's adapter constructs and sends
    *unscoped* under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash this path; there ``os.environ`` is that
    profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and
    ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEBHOOK_HOST = "127.0.0.1"
# BlueBubbles webhook events are small JSON/form payloads; attachments come
# through the REST API, not the webhook. 1 MiB is generous headroom while
# keeping oversized/chunked bodies from being buffered unbounded.
_WEBHOOK_MAX_BODY_BYTES = 1_048_576
DEFAULT_WEBHOOK_PORT = 8645
DEFAULT_WEBHOOK_PATH = "/bluebubbles-webhook"
MAX_TEXT_LENGTH = 4000

# BlueBubbles/iMessage does not expose a stable bot mention identity like
# Slack (<@U...>), Telegram (@botname), or Matrix (MXID). When users opt into
# group mention gating without custom aliases, use conservative Hermes wake
# words so `require_mention: true` is a one-line enablement path.
DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]

# Tapback reaction codes (BlueBubbles associatedMessageType values)
_TAPBACK_ADDED = {
    2000: "love", 2001: "like", 2002: "dislike",
    2003: "laugh", 2004: "emphasize", 2005: "question",
}
_TAPBACK_REMOVED = {
    3000: "love", 3001: "like", 3002: "dislike",
    3003: "laugh", 3004: "emphasize", 3005: "question",
}
_TAPBACK_REACTIONS = frozenset(_TAPBACK_ADDED.values())
_TAPBACK_EMOJI = {
    "❤️": "love",
    "👍": "like",
    "👎": "dislike",
    "😂": "laugh",
    "‼️": "emphasize",
    "❓": "question",
}

# Webhook event types that carry user messages
_MESSAGE_EVENTS = {"new-message", "message", "updated-message"}
_MESSAGE_DEDUP_SIZE = 2048  # LRU cap for completed message identity caches
_MESSAGE_REVISION_WAIT_SECONDS = 1.0
_MESSAGE_CHAT_LOOKUP_TIMEOUT_SECONDS = 1.0
_PROVISIONAL_MESSAGE_WAIT_SECONDS = 1.0
_STOP_TYPING_HELPER_REFRESH_SECONDS = 0.25
_HELPER_NEGATIVE_REFRESH_TTL_SECONDS = 1.0
_READ_RECEIPT_RETRY_SECONDS = 1.0
_ATTACHMENT_RETRY_DELAY_SECONDS = 0.25
_ATTACHMENT_RETRY_ATTEMPTS = 5
_REFERENCED_MESSAGE_LOOKUP_LIMIT = 100
_REPLY_ATTACHMENT_LIMIT = 8
_BLUEBUBBLES_MESSAGE_QUERY_LIMIT = 1_000

_InboundMessageKey = tuple[str, str, str]
_ProvisionalMessageKey = tuple[str, str, str]
_RevisionMediaContext = tuple[List[str], List[str], MessageType, Dict[str, Any], int]


def _is_phone_handle(target: str) -> bool:
    """Return true only for a complete international-style phone handle."""
    value = target.strip()
    return "@" not in value and bool(re.fullmatch(r"\+[1-9]\d{6,14}", value))

# Log redaction patterns
_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

_GUID_CACHE_SIZE = 500  # LRU cap for resolved chat-GUID lookups


def _redact(text: str) -> str:
    """Redact phone numbers and emails from log output."""
    text = _PHONE_RE.sub("[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_bluebubbles_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def _normalize_server_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, flags=re.I):
        value = f"http://{value}"
    return value.rstrip("/")





# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BlueBubblesAdapter(BasePlatformAdapter):
    platform = Platform.BLUEBUBBLES
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_TEXT_LENGTH
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BLUEBUBBLES)
        extra = config.extra or {}
        self.server_url = _normalize_server_url(
            extra.get("server_url") or os.getenv("BLUEBUBBLES_SERVER_URL", "")
        )
        self.password = extra.get("password") or _get_scoped_secret("BLUEBUBBLES_PASSWORD", "")
        self.webhook_host = (
            extra.get("webhook_host")
            or os.getenv("BLUEBUBBLES_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        )
        self.webhook_port = int(
            extra.get("webhook_port")
            or os.getenv("BLUEBUBBLES_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        self.webhook_path = (
            extra.get("webhook_path")
            or os.getenv("BLUEBUBBLES_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
        )
        if not str(self.webhook_path).startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self.send_read_receipts = bool(extra.get("send_read_receipts", True))
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("BLUEBUBBLES_REQUIRE_MENTION")
        self.require_mention = str(_require_mention).strip().lower() in {"true", "1", "yes", "on"}
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("BLUEBUBBLES_MENTION_PATTERNS")
        )
        raw_participant_names = extra.get("participant_names")
        self._participant_names = {
            handle: name.strip()
            for handle, name in raw_participant_names.items()
            if isinstance(handle, str)
            and isinstance(name, str)
            and name.strip()
        } if isinstance(raw_participant_names, dict) else {}
        self.client: Optional[httpx.AsyncClient] = None
        self._runner = None
        self._private_api_enabled: Optional[bool] = None
        self._helper_connected: bool = False
        self._helper_last_refresh_at: float = 0.0

        self._guid_cache: OrderedDict[str, str] = OrderedDict()
        self._seen_message_guids: OrderedDict[str, None] = OrderedDict()
        self._terminal_message_identities: OrderedDict[str, None] = OrderedDict()
        self._pending_message_identities: set[str] = set()
        self._active_message_dispatches: Dict[
            _InboundMessageKey, tuple[int, Optional[str], asyncio.Task]
        ] = {}
        self._tapback_event_states: OrderedDict[
            tuple[str, str, str, int, str, str], tuple[TapbackOperation, int]
        ] = OrderedDict()
        self._tapback_event_serial = 0
        self._pending_message_revisions: Dict[
            _InboundMessageKey,
            tuple[MessageEvent, Optional[str], int, tuple[str, ...]],
        ] = {}
        self._pending_message_revision_tasks: Dict[_InboundMessageKey, asyncio.Task] = {}
        self._provisional_messages: OrderedDict[
            _ProvisionalMessageKey, tuple[Dict[str, Any], Dict[str, Any]]
        ] = OrderedDict()
        self._provisional_message_tasks: Dict[
            _ProvisionalMessageKey, asyncio.Task
        ] = {}
        self._active_attachment_revisions: Dict[_InboundMessageKey, int] = {}
        self._active_attachment_leases: set[tuple[_InboundMessageKey, int]] = set()
        self._active_attachment_identity_leases: Dict[
            tuple[_InboundMessageKey, int], str
        ] = {}
        self._pending_attachment_tasks: Dict[_InboundMessageKey, asyncio.Task] = {}
        self._message_revision_serials: OrderedDict[
            _InboundMessageKey, int
        ] = OrderedDict()
        self._message_revision_orders: OrderedDict[
            str, tuple[int, Optional[float]]
        ] = OrderedDict()
        self._message_revision_text: OrderedDict[_InboundMessageKey, str] = OrderedDict()
        self._message_revision_media: OrderedDict[
            _InboundMessageKey, _RevisionMediaContext
        ] = OrderedDict()
        self._accepted_group_captions: OrderedDict[
            _InboundMessageKey, str
        ] = OrderedDict()
        self._sent_message_keys: OrderedDict[str, str] = OrderedDict()
        self._pending_read_receipts: Dict[str, set[str]] = {}
        self._read_receipt_tasks: Dict[str, asyncio.Task] = {}
        self._sent_read_receipts: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._read_receipts_closed = False
        self._helper_refresh_lock = asyncio.Lock()
        self._typing_transition_locks: Dict[str, asyncio.Lock] = {}
        self._typing_started: set[str] = set()
        self._typing_pending_stops: set[str] = set()
        self._typing_stopped: OrderedDict[str, None] = OrderedDict()
        self._typing_stop_tasks: Dict[str, asyncio.Task] = {}
        self._typing_shutdown = False
        self._inbound_dedup_lock = asyncio.Lock()
        self._outbound_dedup_lock = asyncio.Lock()
        self._read_receipt_lock = asyncio.Lock()

        try:
            self._message_revision_wait_seconds = max(
                0.0,
                float(
                    extra.get(
                        "message_revision_wait_seconds",
                        _MESSAGE_REVISION_WAIT_SECONDS,
                    )
                ),
            )
        except (TypeError, ValueError):
            self._message_revision_wait_seconds = _MESSAGE_REVISION_WAIT_SECONDS
        try:
            self._attachment_retry_delay_seconds = max(
                0.0,
                float(
                    extra.get(
                        "attachment_retry_delay_seconds",
                        _ATTACHMENT_RETRY_DELAY_SECONDS,
                    )
                ),
            )
        except (TypeError, ValueError):
            self._attachment_retry_delay_seconds = _ATTACHMENT_RETRY_DELAY_SECONDS
        try:
            self._message_retry_max_attempts = max(
                1, int(extra.get("message_retry_max_attempts", 3))
            )
        except (TypeError, ValueError):
            self._message_retry_max_attempts = 3
        try:
            self._message_retry_base_delay_seconds = max(
                0.0, float(extra.get("message_retry_base_delay_seconds", 0.5))
            )
        except (TypeError, ValueError):
            self._message_retry_base_delay_seconds = 0.5

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _api_url(self, path: str) -> str:
        sep = "&" if "?" in path else "?"
        return f"{self.server_url}{path}{sep}password={quote(self.password, safe='')}"


    @staticmethod
    def _compile_mention_patterns(raw: Any) -> List[re.Pattern]:
        """Compile group-mention wake words from config/env.

        ``raw`` is a list (from config or env JSON), a string (raw env var:
        JSON list, or comma/newline-separated), or None (use Hermes defaults).
        """
        return compile_mention_patterns(
            raw,
            log_prefix="bluebubbles",
            defaults=DEFAULT_MENTION_PATTERNS,
            logger_=logger,
        )

    def _message_matches_mention_patterns(self, text: str) -> bool:
        if not text or not self._mention_patterns:
            return False
        return any(pattern.search(text) for pattern in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip a leading BlueBubbles wake word before dispatch.

        Custom mention patterns are regular expressions, so stripping only a
        leading match avoids deleting ordinary words later in the prompt.
        """
        if not text:
            return text
        for pattern in self._mention_patterns:
            match = pattern.match(text.lstrip())
            if match:
                cleaned = text.lstrip()[match.end():].lstrip(" ,:-")
                return cleaned or text
        return text

    async def _api_get(self, path: str) -> Dict[str, Any]:
        assert self.client is not None
        res = await self.client.get(self._api_url(path))
        res.raise_for_status()
        return res.json()

    async def _api_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.client is not None
        res = await self.client.post(self._api_url(path), json=payload)
        res.raise_for_status()
        return res.json()

    async def _refresh_helper_state(self) -> bool:
        """Refresh stale Private API/helper reachability from the live server."""
        if self._private_api_enabled and self._helper_connected:
            return True
        if not self.client:
            return False
        async with self._helper_refresh_lock:
            if self._private_api_enabled and self._helper_connected:
                return True
            loop = asyncio.get_running_loop()
            if (
                self._helper_last_refresh_at > 0
                and loop.time() - self._helper_last_refresh_at
                < _HELPER_NEGATIVE_REFRESH_TTL_SECONDS
            ):
                return False
            try:
                info = await self._api_get("/api/v1/server/info")
                server_data = (info or {}).get("data", {})
                self._private_api_enabled = bool(server_data.get("private_api"))
                self._helper_connected = bool(server_data.get("helper_connected"))
            except Exception as exc:
                logger.debug("[bluebubbles] helper state refresh failed: %s", exc)
                self._helper_last_refresh_at = loop.time()
                return False
            self._helper_last_refresh_at = loop.time()
            return bool(self._private_api_enabled and self._helper_connected)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect_outbound(self) -> bool:
        """Open only the REST client needed for outbound delivery.

        Standalone senders (cron and send_message outside the live gateway) must
        not bind, register, or unregister the gateway's inbound webhook.
        """
        self._typing_shutdown = False
        self._read_receipts_closed = False
        if not self.server_url or not self.password:
            logger.error(
                "[bluebubbles] BLUEBUBBLES_SERVER_URL and BLUEBUBBLES_PASSWORD are required"
            )
            return False

        # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            await self._api_get("/api/v1/ping")
            info = await self._api_get("/api/v1/server/info")
            server_data = (info or {}).get("data", {})
            self._private_api_enabled = bool(server_data.get("private_api"))
            self._helper_connected = bool(server_data.get("helper_connected"))
            logger.info(
                "[bluebubbles] connected to %s (private_api=%s, helper=%s)",
                self.server_url,
                self._private_api_enabled,
                self._helper_connected,
            )
            return True
        except Exception as exc:
            logger.error(
                "[bluebubbles] cannot reach server at %s: %s", self.server_url, exc
            )
            await self.disconnect_outbound()
            return False

    async def disconnect_outbound(self) -> None:
        """Close an outbound-only REST client without touching webhooks."""
        self._typing_shutdown = True
        await self._reconcile_typing_stops()
        await self._cancel_read_receipts()
        if self.client:
            await self.client.aclose()
            self.client = None
        self._clear_typing_lifecycle_state()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not await self.connect_outbound():
            return False
        from aiohttp import web

        # Explicit body cap: BlueBubbles webhook events are small JSON (or
        # form-encoded) payloads. client_max_size makes aiohttp enforce the
        # cap on every read path — including chunked requests that carry no
        # Content-Length (same pattern as webhook.py / raft, #58536/#58902).
        app = web.Application(client_max_size=_WEBHOOK_MAX_BODY_BYTES)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        app.router.add_post(self.webhook_path, self._handle_webhook)
        # The webhook auth value is carried in the query string because the
        # BlueBubbles webhook API cannot send custom headers. Do not let
        # aiohttp access logs write that request target to agent.log.
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[bluebubbles] webhook listening on http://%s:%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
        )

        # Register webhook with BlueBubbles server
        # This is required for the server to know where to send events
        await self._register_webhook()

        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        self._typing_shutdown = True
        await self._reconcile_typing_stops()
        # Unregister webhook before cleaning up
        await self._unregister_webhook()

        await self.disconnect_outbound()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._clear_typing_lifecycle_state()
        self._mark_disconnected()

    async def cancel_background_tasks(self) -> None:
        self._typing_shutdown = True
        await super().cancel_background_tasks()
        await self._cancel_read_receipts()
        await self._reconcile_typing_stops()
        self._clear_typing_lifecycle_state()
        async with self._inbound_dedup_lock:
            self._pending_message_revisions.clear()
            self._pending_message_revision_tasks.clear()
            self._provisional_messages.clear()
            self._provisional_message_tasks.clear()
            self._pending_message_identities.clear()
            self._active_message_dispatches.clear()
            self._active_attachment_revisions.clear()
            self._active_attachment_leases.clear()
            self._active_attachment_identity_leases.clear()
            self._pending_attachment_tasks.clear()
            self._message_revision_media.clear()

    @property
    def _webhook_url(self) -> str:
        """Compute the external webhook URL for BlueBubbles registration."""
        host = self.webhook_host
        if host in {"0.0.0.0", "127.0.0.1", "localhost", "::"}:
            host = "localhost"
        return f"http://{host}:{self.webhook_port}{self.webhook_path}"

    @property
    def _webhook_register_url(self) -> str:
        """Webhook URL registered with BlueBubbles, including the password as
        a query param so inbound webhook POSTs carry credentials.

        BlueBubbles posts events to the exact URL registered via
        ``/api/v1/webhook``. Its webhook registration API does not support
        custom headers, so embedding the password in the URL is the only
        way to authenticate inbound webhooks without disabling auth.
        """
        base = self._webhook_url
        if self.password:
            return f"{base}?password={quote(self.password, safe='')}"
        return base

    @property
    def _webhook_register_url_for_log(self) -> str:
        """Webhook registration URL safe for logs."""
        base = self._webhook_url
        if self.password:
            return f"{base}?password=***"
        return base

    async def _find_registered_webhooks(self, url: str) -> list:
        """Return list of BB webhook entries matching *url*."""
        try:
            res = await self._api_get("/api/v1/webhook")
            data = res.get("data")
            if isinstance(data, list):
                return [wh for wh in data if wh.get("url") == url]
        except Exception:
            pass
        return []

    async def _register_webhook(self) -> bool:
        """Register this webhook URL with the BlueBubbles server.

        BlueBubbles requires webhooks to be registered via API before
        it will send events.  Checks for an existing registration first
        to avoid duplicates (e.g. after a crash without clean shutdown).
        """
        if not self.client:
            return False

        webhook_url = self._webhook_register_url

        # Crash resilience — reuse an existing registration if present
        existing = await self._find_registered_webhooks(webhook_url)
        if existing:
            logger.info(
                "[bluebubbles] webhook already registered: %s",
                self._webhook_register_url_for_log,
            )
            return True

        payload = {
            "url": webhook_url,
            "events": ["new-message", "updated-message"],
        }

        try:
            res = await self._api_post("/api/v1/webhook", payload)
            status = res.get("status", 0)
            if 200 <= status < 300:
                logger.info(
                    "[bluebubbles] webhook registered with server: %s",
                    self._webhook_register_url_for_log,
                )
                return True
            else:
                logger.warning(
                    "[bluebubbles] webhook registration returned status %s: %s",
                    status,
                    res.get("message"),
                )
                return False
        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to register webhook with server: %s",
                exc,
            )
            return False

    async def _unregister_webhook(self) -> bool:
        """Unregister this webhook URL from the BlueBubbles server.

        Removes *all* matching registrations to clean up any duplicates
        left by prior crashes.
        """
        if not self.client:
            return False

        webhook_url = self._webhook_register_url
        removed = False

        try:
            for wh in await self._find_registered_webhooks(webhook_url):
                wh_id = wh.get("id")
                if wh_id:
                    res = await self.client.delete(
                        self._api_url(f"/api/v1/webhook/{wh_id}")
                    )
                    res.raise_for_status()
                    removed = True
            if removed:
                logger.info(
                    "[bluebubbles] webhook unregistered: %s",
                    self._webhook_register_url_for_log,
                )
        except Exception as exc:
            logger.debug(
                "[bluebubbles] failed to unregister webhook (non-critical): %s",
                exc,
            )
        return removed

    # ------------------------------------------------------------------
    # Chat GUID resolution
    # ------------------------------------------------------------------

    async def _resolve_chat_guid(self, target: str) -> Optional[str]:
        """Resolve an email/phone to a BlueBubbles chat GUID.

        If *target* already contains a semicolon (raw GUID format like
        ``iMessage;-;user@example.com``), it is returned as-is.  Otherwise
        the adapter queries the BlueBubbles chat list and matches strictly
        on ``chatIdentifier`` / ``identifier``.

        Participant membership is intentionally NOT used as a fallback:
        the same contact can appear in a 1:1 DM and in any number of group
        chats, so a participant match would let an outbound DM reply leak
        into a group thread (see #24157). When no exact chat identity
        matches, return ``None`` and let the caller create a fresh DM
        explicitly via ``_create_chat_for_handle``.
        """
        target = (target or "").strip()
        if not target:
            return None
        # Already a raw GUID
        if ";" in target:
            return target
        if target in self._guid_cache:
            self._guid_cache.move_to_end(target)
            return self._guid_cache[target]
        try:
            payload = await self._api_post(
                "/api/v1/chat/query",
                {"limit": 100, "offset": 0},
            )
            for chat in payload.get("data", []) or []:
                guid = chat.get("guid") or chat.get("chatGuid")
                identifier = chat.get("chatIdentifier") or chat.get("identifier")
                if identifier == target:
                    if guid:
                        self._guid_cache[target] = guid
                        while len(self._guid_cache) > _GUID_CACHE_SIZE:
                            self._guid_cache.popitem(last=False)
                    return guid
        except Exception:
            pass
        return None

    async def _create_chat_for_handle(
        self,
        address: str,
        message: str,
        temp_guid: Optional[str] = None,
    ) -> SendResult:
        """Create a new chat by sending the first message to *address*."""
        payload = {
            "addresses": [address],
            "message": message,
            "tempGuid": temp_guid or f"temp-{uuid.uuid4().hex}",
        }
        try:
            res = await self._api_post("/api/v1/chat/new", payload)
            data = res.get("data") or {}
            msg_id = data.get("guid") or data.get("messageGuid") or "ok"
            return SendResult(success=True, message_id=str(msg_id), raw_response=res)
        except Exception as exc:
            return SendResult(success=False, error=str(exc) or type(exc).__name__)

    # ------------------------------------------------------------------
    # Text sending
    # ------------------------------------------------------------------

    @staticmethod
    def truncate_message(content: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
        # Use the base splitter but skip pagination indicators — iMessage
        # bubbles flow naturally without "(1/3)" suffixes.
        chunks = BasePlatformAdapter.truncate_message(content, max_length)
        return [re.sub(r"\s*\(\d+/\d+\)$", "", c) for c in chunks]

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        text = self.format_message(content)
        if not text:
            return SendResult(success=False, error="BlueBubbles send requires text")
        origin_id = reply_to or (
            metadata.get("reply_to_message_id") if metadata else None
        )
        internal_notice = bool(metadata and metadata.get("internal_notice") is True)
        if not origin_id:
            return await self._send_formatted_text(
                chat_id, text, reply_to, internal_notice=internal_notice
            )

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        dedup_key = f"{chat_id}\0{origin_id}\0{content_hash}"
        async with self._outbound_dedup_lock:
            prior_message_id = self._sent_message_keys.get(dedup_key)
            if prior_message_id is not None:
                self._sent_message_keys.move_to_end(dedup_key)
                logger.debug("[bluebubbles] suppressing duplicate outbound reply")
                return SendResult(
                    success=True,
                    message_id=prior_message_id,
                    raw_response={"deduplicated": True},
                )
            result = await self._send_formatted_text(
                chat_id,
                text,
                reply_to,
                internal_notice=internal_notice,
                dedup_key=dedup_key,
            )
            if result.success and result.message_id:
                self._sent_message_keys[dedup_key] = result.message_id
                self._sent_message_keys.move_to_end(dedup_key)
                while len(self._sent_message_keys) > _MESSAGE_DEDUP_SIZE:
                    self._sent_message_keys.popitem(last=False)
            return result

    async def _send_formatted_text(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str],
        *,
        internal_notice: bool,
        dedup_key: Optional[str] = None,
    ) -> SendResult:
        # Keep a complete reply in one iMessage bubble whenever it fits. Only
        # split when the platform's message-length limit requires it.
        chunks = (
            [text]
            if len(text) <= self.MAX_MESSAGE_LENGTH
            else self.truncate_message(text, max_length=self.MAX_MESSAGE_LENGTH)
        )
        private_reply_available = bool(
            reply_to and await self._refresh_helper_state()
        )
        last = SendResult(success=True)
        for chunk_index, chunk in enumerate(chunks):
            if dedup_key:
                temp_guid = "temp-" + hashlib.sha256(
                    f"{dedup_key}\0{chunk_index}".encode("utf-8")
                ).hexdigest()
            else:
                temp_guid = f"temp-{uuid.uuid4().hex}"
            guid = await self._resolve_chat_guid(chat_id)
            if not guid:
                if internal_notice and _is_phone_handle(chat_id):
                    logger.info(
                        "[bluebubbles] suppressed internal notice to unresolved phone target"
                    )
                    return SendResult(
                        success=True,
                        raw_response={"suppressed": "internal_sms_notice"},
                    )
                # If the target looks like an address, try creating a new chat
                if self._private_api_enabled and (
                    "@" in chat_id or _is_phone_handle(chat_id)
                ):
                    created = await self._create_chat_for_handle(
                        chat_id, chunk, temp_guid=temp_guid
                    )
                    if not created.success:
                        return created
                    last = created
                    if chunk_index == len(chunks) - 1:
                        return created
                    continue
                return SendResult(
                    success=False,
                    error=f"BlueBubbles chat not found for target: {chat_id}",
                )
            if guid.lower().startswith("sms;") and internal_notice:
                logger.info("[bluebubbles] suppressed internal notice over SMS")
                return SendResult(
                    success=True,
                    raw_response={"suppressed": "internal_sms_notice"},
                )
            payload: Dict[str, Any] = {
                "chatGuid": guid,
                "tempGuid": temp_guid,
                "message": chunk,
            }
            if reply_to and private_reply_available:
                payload["method"] = "private-api"
                payload["selectedMessageGuid"] = reply_to
                payload["partIndex"] = 0
            try:
                res = await self._api_post("/api/v1/message/text", payload)
                data = res.get("data") or {}
                msg_id = data.get("guid") or data.get("messageGuid") or "ok"
                last = SendResult(
                    success=True, message_id=str(msg_id), raw_response=res
                )
            except Exception as exc:
                return SendResult(success=False, error=str(exc) or type(exc).__name__)
        return last

    # ------------------------------------------------------------------
    # Media sending (outbound)
    # ------------------------------------------------------------------

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        is_audio_message: bool = False,
    ) -> SendResult:
        """Send a file attachment via BlueBubbles multipart upload."""
        if not self.client:
            return SendResult(success=False, error="Not connected")
        if not await asyncio.to_thread(os.path.isfile, file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        guid = await self._resolve_chat_guid(chat_id)
        if not guid:
            return SendResult(success=False, error=f"Chat not found: {chat_id}")

        fname = filename or os.path.basename(file_path)
        try:
            # httpx's async multipart iterator reads file-like objects through
            # a synchronous chunk generator. Read the file off the event-loop
            # thread before handing bytes to the client.
            payload = await asyncio.to_thread(Path(file_path).read_bytes)
            files = {"attachment": (fname, payload, "application/octet-stream")}
            data: Dict[str, str] = {
                "chatGuid": guid,
                "name": fname,
                "tempGuid": uuid.uuid4().hex,
            }
            if is_audio_message:
                data["isAudioMessage"] = "true"
            res = await self.client.post(
                self._api_url("/api/v1/message/attachment"),
                files=files,
                data=data,
                timeout=120,
            )
            res.raise_for_status()
            result = res.json()

            if caption:
                await self.send(chat_id, caption)

            if result.get("status") == 200:
                rdata = result.get("data") or {}
                msg_id = rdata.get("guid") if isinstance(rdata, dict) else None
                return SendResult(
                    success=True, message_id=msg_id, raw_response=result
                )
            return SendResult(
                success=False,
                error=result.get("message", "Attachment upload failed"),
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url

            local_path = await cache_image_from_url(image_url)
            return await self._send_attachment(chat_id, local_path, caption=caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(chat_id, image_path, caption=caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(
            chat_id, audio_path, caption=caption, is_audio_message=True
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(
            chat_id, file_path, filename=file_name, caption=caption
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self.send_image(
            chat_id, animation_url, caption, reply_to, metadata
        )

    # ------------------------------------------------------------------
    # Typing indicators
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        key = str(chat_id)
        if self._typing_shutdown:
            return

        self._typing_pending_stops.discard(key)
        self._typing_stopped.pop(key, None)
        retry_task = self._typing_stop_tasks.pop(key, None)
        if retry_task is not None and retry_task is not asyncio.current_task():
            retry_task.cancel()

        lock = self._typing_transition_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # A stop request can arrive while helper discovery or GUID lookup
            # is in flight. Re-check at each process boundary so cancellation
            # cannot resurrect typing after cleanup has begun.
            if self._typing_shutdown or key in self._typing_pending_stops:
                return
            if not await self._refresh_helper_state():
                return
            if self._typing_shutdown or key in self._typing_pending_stops:
                return
            try:
                guid = await self._resolve_chat_guid(chat_id)
                if not guid or self._typing_shutdown or key in self._typing_pending_stops:
                    return
                encoded = quote(guid, safe="")
                # Once the request crosses this boundary its delivery is
                # uncertain even if our task is cancelled. Track it before the
                # await so a later stop always reconciles the remote state.
                self._typing_started.add(key)
                await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def stop_typing(self, chat_id: str) -> None:
        key = str(chat_id)
        if key in self._typing_stopped and key not in self._typing_started:
            return
        self._typing_pending_stops.add(key)
        await self._apply_typing_stop(chat_id, schedule_retry=not self._typing_shutdown)

    async def _apply_typing_stop(
        self,
        chat_id: str,
        *,
        schedule_retry: bool,
    ) -> None:
        """Serialize one remote stop after any in-flight typing start."""
        key = str(chat_id)
        lock = self._typing_transition_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._typing_pending_stops:
                return
            if key in self._typing_stopped and key not in self._typing_started:
                self._typing_pending_stops.discard(key)
                return
            helper_available = False
            try:
                helper_available = await asyncio.wait_for(
                    self._refresh_helper_state(),
                    timeout=_STOP_TYPING_HELPER_REFRESH_SECONDS,
                )
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            if not helper_available:
                if key not in self._typing_started:
                    # No start request crossed the helper boundary, so there
                    # is no remote state to reconcile and no reason to retain
                    # a retry worker for an idle chat.
                    self._typing_pending_stops.discard(key)
                elif schedule_retry:
                    self._schedule_typing_stop_retry(chat_id)
                return
            try:
                guid = await self._resolve_chat_guid(chat_id)
                if not guid:
                    if schedule_retry:
                        self._schedule_typing_stop_retry(chat_id)
                    return
                client = self.client
                if client is None:
                    return
                encoded = quote(guid, safe="")
                await client.delete(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if schedule_retry:
                    self._schedule_typing_stop_retry(chat_id)
                return
            self._typing_started.discard(key)
            self._typing_pending_stops.discard(key)
            self._typing_stopped[key] = None
            self._typing_stopped.move_to_end(key)
            while len(self._typing_stopped) > _MESSAGE_DEDUP_SIZE:
                self._typing_stopped.popitem(last=False)

    def _schedule_typing_stop_retry(self, chat_id: str) -> None:
        """Keep one retry worker per chat until the cold helper is reachable."""
        key = str(chat_id)
        if self._typing_shutdown or not self.client:
            return
        existing = self._typing_stop_tasks.get(key)
        if existing is not None and not existing.done():
            return

        async def retry() -> None:
            current = asyncio.current_task()
            try:
                while (
                    not self._typing_shutdown
                    and self.client is not None
                    and key in self._typing_pending_stops
                ):
                    await asyncio.sleep(_HELPER_NEGATIVE_REFRESH_TTL_SECONDS)
                    await self._apply_typing_stop(chat_id, schedule_retry=False)
            except asyncio.CancelledError:
                pass
            finally:
                if self._typing_stop_tasks.get(key) is current:
                    self._typing_stop_tasks.pop(key, None)

        task = asyncio.create_task(retry())
        self._typing_stop_tasks[key] = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _reconcile_typing_stops(self) -> None:
        """Issue one final stop for every locally active or pending chat."""
        chats = set(self._typing_started) | set(self._typing_pending_stops)
        for chat_id in chats:
            self._typing_pending_stops.add(chat_id)
            try:
                await self._apply_typing_stop(chat_id, schedule_retry=False)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    def _clear_typing_lifecycle_state(self) -> None:
        for task in self._typing_stop_tasks.values():
            if not task.done():
                task.cancel()
        self._typing_stop_tasks.clear()
        self._typing_transition_locks.clear()
        self._typing_started.clear()
        self._typing_pending_stops.clear()
        self._typing_stopped.clear()

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    def _read_receipt_allowed(
        self,
        chat_id: str,
        *,
        is_group: Optional[bool] = None,
        admitted: bool = False,
    ) -> bool:
        """Apply receipt privacy policy at the final delivery boundary."""
        if not self.send_read_receipts or self._read_receipts_closed or not chat_id:
            return False
        group = bool(is_group) or ";+;" in chat_id
        return not group or admitted

    async def mark_read(self, chat_id: str) -> bool:
        """Mark a direct conversation read; group callers must use admission."""
        return await self._deliver_read_receipt(
            chat_id,
            is_group=None,
            admitted=False,
        )

    async def _deliver_read_receipt(
        self,
        chat_id: str,
        *,
        is_group: Optional[bool] = None,
        admitted: bool = False,
    ) -> bool:
        if not self._read_receipt_allowed(
            chat_id,
            is_group=is_group,
            admitted=admitted,
        ):
            return False
        if not await self._refresh_helper_state():
            return False
        if self._read_receipts_closed:
            return False
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                if self._read_receipts_closed:
                    return False
                encoded = quote(guid, safe="")
                response = await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/read"), timeout=5
                )
                response.raise_for_status()
                return True
        except Exception:
            pass
        return False

    async def _queue_read_receipt(
        self,
        chat_id: str,
        message_id: str,
        *,
        is_group: bool,
        admitted: bool,
    ) -> bool:
        """Queue one admitted receipt until the Private API helper is ready."""
        if not self._read_receipt_allowed(
            chat_id,
            is_group=is_group,
            admitted=admitted,
        ):
            return False
        message_id = (message_id or "").strip()
        if not message_id:
            return False
        receipt_key = (chat_id, message_id)
        async with self._read_receipt_lock:
            if self._read_receipts_closed or receipt_key in self._sent_read_receipts:
                return False
            pending = self._pending_read_receipts.setdefault(chat_id, set())
            if message_id in pending:
                return False
            pending.add(message_id)
            task = self._read_receipt_tasks.get(chat_id)
            if task is None or task.done():
                task = asyncio.create_task(
                    self._run_read_receipt_worker(chat_id, is_group=is_group)
                )
                self._read_receipt_tasks[chat_id] = task
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
        return True

    async def _run_read_receipt_worker(self, chat_id: str, *, is_group: bool) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                async with self._read_receipt_lock:
                    if self._read_receipts_closed:
                        self._pending_read_receipts.pop(chat_id, None)
                        return
                    pending = self._pending_read_receipts.get(chat_id)
                    if not pending:
                        return
                    attempt_ids = tuple(pending)

                if await self._deliver_read_receipt(
                    chat_id,
                    is_group=is_group,
                    admitted=True,
                ):
                    async with self._read_receipt_lock:
                        pending = self._pending_read_receipts.get(chat_id)
                        if pending is not None:
                            for message_id in attempt_ids:
                                pending.discard(message_id)
                                key = (chat_id, message_id)
                                self._sent_read_receipts[key] = None
                                self._sent_read_receipts.move_to_end(key)
                            while len(self._sent_read_receipts) > _MESSAGE_DEDUP_SIZE:
                                self._sent_read_receipts.popitem(last=False)
                            if not pending:
                                self._pending_read_receipts.pop(chat_id, None)
                                if self._read_receipt_tasks.get(chat_id) is current_task:
                                    self._read_receipt_tasks.pop(chat_id, None)
                                return
                    continue
                await asyncio.sleep(_READ_RECEIPT_RETRY_SECONDS)
        except asyncio.CancelledError:
            async with self._read_receipt_lock:
                self._pending_read_receipts.pop(chat_id, None)
            raise
        finally:
            async with self._read_receipt_lock:
                if self._read_receipt_tasks.get(chat_id) is current_task:
                    self._read_receipt_tasks.pop(chat_id, None)

    async def _cancel_read_receipts(self) -> None:
        """Cancel queued receipt work and prevent post-teardown sends."""
        self._read_receipts_closed = True
        async with self._read_receipt_lock:
            tasks = [task for task in self._read_receipt_tasks.values() if not task.done()]
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._read_receipt_lock:
            self._pending_read_receipts.clear()
            self._read_receipt_tasks.clear()
            self._sent_read_receipts.clear()

    # ------------------------------------------------------------------
    # Tapback reactions
    # ------------------------------------------------------------------

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Never generate an outbound Tapback from inbound processing state."""
        return

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send one explicitly approved native Tapback to an exact target."""
        reaction = _TAPBACK_EMOJI.get((emoji or "").strip())
        if not reaction or not message_id:
            return {"success": False, "error": "Exact native Tapback and message_id required"}
        result = await self.send_reaction(
            chat_id,
            message_id,
            reaction,
            source_event_id=f"outbound:add:{chat_id}:{message_id}:{reaction}",
        )
        return {
            "success": result.success,
            "message_id": result.message_id,
            **({"error": result.error} if result.error else {}),
        }

    async def remove_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Remove the same exact native Tapback identity from its exact target."""
        reaction = _TAPBACK_EMOJI.get((emoji or "").strip())
        if not reaction or not message_id:
            return {"success": False, "error": "Exact native Tapback and message_id required"}
        result = await self.send_reaction(
            chat_id,
            message_id,
            f"-{reaction}",
            source_event_id=f"outbound:remove:{chat_id}:{message_id}:{reaction}",
        )
        return {
            "success": result.success,
            "message_id": result.message_id,
            **({"error": result.error} if result.error else {}),
        }

    async def send_reaction(
        self,
        chat_id: str,
        message_guid: str,
        reaction: str,
        part_index: int = 0,
        *,
        source_event_id: Optional[str] = None,
    ) -> SendResult:
        """Add or remove a native Tapback on an exact message and chat."""
        normalized = (reaction or "").strip().lower()
        base_reaction = normalized.removeprefix("-")
        if base_reaction not in _TAPBACK_REACTIONS or normalized.startswith("--"):
            return SendResult(
                success=False,
                error=f"Unsupported BlueBubbles reaction: {reaction}",
            )
        message_guid = (message_guid or "").strip()
        if not message_guid:
            return SendResult(success=False, error="BlueBubbles reaction requires message GUID")
        if isinstance(part_index, bool) or not isinstance(part_index, int) or part_index < 0:
            return SendResult(
                success=False,
                error="BlueBubbles reaction part index must be a non-negative integer",
            )
        if not self.client:
            return SendResult(success=False, error="BlueBubbles is not connected")
        if not await self._refresh_helper_state():
            return SendResult(success=False, error="Private API helper not connected")
        guid = await self._resolve_chat_guid(chat_id)
        if not guid:
            return SendResult(success=False, error=f"Chat not found: {chat_id}")
        try:
            operation = TapbackOperation(
                platform="bluebubbles",
                chat_id=guid,
                target_message_id=message_guid,
                sender_id="self",
                reaction=TapbackType(base_reaction),
                action=(
                    TapbackAction.REMOVE
                    if normalized.startswith("-")
                    else TapbackAction.ADD
                ),
                direction=TapbackDirection.OUTBOUND,
                source_event_id=source_event_id or message_guid,
                part_index=part_index,
            ).transition_to(TapbackStatus.VALIDATED)
            target = await self._query_exact_message(guid, message_guid)
            if target is None:
                return SendResult(
                    success=False,
                    error=f"Message {message_guid} does not belong to chat {guid}",
                )
            res = await self._api_post(
                "/api/v1/message/react",
                {
                    "chatGuid": operation.chat_id,
                    "selectedMessageGuid": operation.target_message_id,
                    "reaction": (
                        f"-{operation.reaction.value}"
                        if operation.action is TapbackAction.REMOVE
                        else operation.reaction.value
                    ),
                    "partIndex": operation.part_index,
                },
            )
            data = res.get("data") or {}
            message_id = data.get("guid") or data.get("messageGuid")
            return SendResult(
                success=True,
                message_id=str(message_id) if message_id else None,
                raw_response=res,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        is_group = ";+;" in (chat_id or "")
        info: Dict[str, Any] = {
            "name": chat_id,
            "type": "group" if is_group else "dm",
        }
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                res = await self._api_get(
                    f"/api/v1/chat/{encoded}?with=participants"
                )
                data = (res or {}).get("data", {})
                display_name = (
                    data.get("displayName")
                    or data.get("chatIdentifier")
                    or chat_id
                )
                participants = []
                for p in data.get("participants", []) or []:
                    addr = (p.get("address") or "").strip()
                    if addr:
                        participants.append(addr)
                info["name"] = display_name
                if participants:
                    info["participants"] = participants
        except Exception:
            pass
        return info

    def format_message(self, content: str) -> str:
        return strip_markdown(content)

    # ------------------------------------------------------------------
    # Inbound attachment downloading (from #4588)
    # ------------------------------------------------------------------

    async def _download_attachment(
        self, att_guid: str, att_meta: Dict[str, Any]
    ) -> Optional[str]:
        """Download an attachment from BlueBubbles and cache it locally.

        Returns the local file path on success, None on failure.
        """
        if not self.client:
            return None

        try:
            encoded = quote(att_guid, safe="")
            resp = await self.client.get(
                self._api_url(f"/api/v1/attachment/{encoded}/download"),
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.content

            mime = (att_meta.get("mimeType") or "").lower()
            transfer_name = att_meta.get("transferName", "")

            if mime.startswith("image/"):
                ext = ext_for_mime(
                    mime,
                    overrides=_BLUEBUBBLES_IMAGE_EXT_OVERRIDES,
                    # Historical map was closed: any unlisted image mime
                    # fell back to .jpg without consulting mimetypes.
                    use_defaults=False,
                    use_mimetypes=False,
                    fallback=".jpg",
                ) or ".jpg"
                return await cache_image_from_bytes_async(data, ext)

            if mime.startswith("audio/"):
                ext = ext_for_mime(
                    mime,
                    overrides=_BLUEBUBBLES_AUDIO_EXT_OVERRIDES,
                    # Historical map was closed: any unlisted audio mime
                    # fell back to .mp3 without consulting mimetypes.
                    use_defaults=False,
                    use_mimetypes=False,
                    fallback=".mp3",
                ) or ".mp3"
                return await cache_audio_from_bytes_async(data, ext)

            # Videos, documents, and everything else
            filename = transfer_name or f"file_{uuid.uuid4().hex[:8]}"
            return await cache_document_from_bytes_async(data, filename)

        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to download attachment %s: %s",
                _redact(att_guid),
                exc,
            )
            return None

    async def _query_exact_message(
        self,
        chat_guid: str,
        message_guid: str,
        *,
        include_attachments: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return one exact message only when it belongs to the requested chat."""
        if not chat_guid or not message_guid:
            return None
        with_relations = ["chats"]
        if include_attachments:
            with_relations.insert(0, "attachments")
        encoded_guid = quote(message_guid, safe="")
        encoded_with = quote(",".join(with_relations), safe="")
        record: Any = None
        try:
            payload = await self._api_get(
                f"/api/v1/message/{encoded_guid}?with={encoded_with}"
            )
            record = payload.get("data")
            if isinstance(record, list):
                record = record[0] if len(record) == 1 else None
        except Exception:
            # Older/custom BlueBubbles-compatible servers may lack the find
            # route. Keep the exact chat-scoped query as a compatibility fallback.
            payload = await self._api_post(
                "/api/v1/message/query",
                {
                    "limit": 1,
                    "offset": 0,
                    "chatGuid": chat_guid,
                    "with": with_relations,
                    "where": [
                        {
                            "statement": "message.guid = :guid",
                            "args": {"guid": message_guid},
                        }
                    ],
                },
            )
            records = payload.get("data") or []
            record = records[0] if isinstance(records, list) and records else None
        if not isinstance(record, dict) or record.get("guid") != message_guid:
            return None
        chat_guids = {
            guid
            for chat in (record.get("chats") or [])
            if isinstance(chat, dict)
            for guid in [self._value(chat.get("guid"), chat.get("chatGuid"))]
            if guid
        }
        if chat_guids != {chat_guid}:
            return None
        return record

    async def _lookup_referenced_message(
        self,
        chat_guid: str,
        message_guid: str,
        *,
        include_attachments: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Resolve one reference from the bounded current-chat message window."""
        if (
            not isinstance(chat_guid, str)
            or not chat_guid
            or chat_guid != chat_guid.strip()
            or not isinstance(message_guid, str)
            or not message_guid
            or message_guid != message_guid.strip()
        ):
            return None
        with_relations = ["chats"]
        if include_attachments:
            with_relations.insert(0, "attachments")
        try:
            payload = await asyncio.wait_for(
                self._api_post(
                    "/api/v1/message/query",
                    {
                        "limit": _REFERENCED_MESSAGE_LOOKUP_LIMIT,
                        "offset": 0,
                        "chatGuid": chat_guid,
                        "with": with_relations,
                    },
                ),
                timeout=_MESSAGE_CHAT_LOOKUP_TIMEOUT_SECONDS,
            )
        except Exception:
            return None
        records = payload.get("data") if isinstance(payload, dict) else None
        if (
            not isinstance(records, list)
            or len(records) > _REFERENCED_MESSAGE_LOOKUP_LIMIT
        ):
            return None
        matches = [
            record
            for record in records
            if isinstance(record, dict) and record.get("guid") == message_guid
        ]
        if len(matches) != 1:
            return None
        record = matches[0]
        chat_guids = {
            guid
            for chat in (record.get("chats") or [])
            if isinstance(chat, dict)
            for guid in [self._value(chat.get("guid"), chat.get("chatGuid"))]
            if guid
        }
        return record if chat_guids == {chat_guid} else None

    async def _verify_inbound_message_membership(
        self,
        message_guid: str,
        candidate_chat_guid: Optional[str],
    ) -> Optional[str]:
        """Resolve exact server-side membership, never trusting webhook chat fields."""
        if not message_guid:
            return None
        if not candidate_chat_guid:
            return await self._resolve_exact_message_chat_guid(message_guid)
        try:
            record = await asyncio.wait_for(
                self._query_exact_message(candidate_chat_guid, message_guid),
                timeout=_MESSAGE_CHAT_LOOKUP_TIMEOUT_SECONDS,
            )
        except Exception:
            return None
        return candidate_chat_guid if record is not None else None

    async def _resolve_exact_message_chat_guid(
        self, message_guid: str
    ) -> Optional[str]:
        """Resolve one message GUID to exactly one authoritative chat."""
        if not self.client or not message_guid:
            return None
        encoded_guid = quote(message_guid, safe="")
        try:
            payload = await asyncio.wait_for(
                self._api_get(f"/api/v1/message/{encoded_guid}?with=chats"),
                timeout=_MESSAGE_CHAT_LOOKUP_TIMEOUT_SECONDS,
            )
        except Exception:
            return None
        record: Any = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(record, list):
            record = record[0] if len(record) == 1 else None
        if not isinstance(record, dict) or self._value(record.get("guid")) != message_guid:
            return None
        chat_guids = {
            guid
            for chat in (record.get("chats") or [])
            if isinstance(chat, dict)
            for guid in [self._value(chat.get("guid"), chat.get("chatGuid"))]
            if guid
        }
        if len(chat_guids) != 1:
            return None
        return next(iter(chat_guids))

    async def _expire_provisional_message(
        self, provisional_key: _ProvisionalMessageKey
    ) -> None:
        """Retry exact membership once, then fail closed."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(_PROVISIONAL_MESSAGE_WAIT_SECONDS)
            async with self._inbound_dedup_lock:
                if self._provisional_message_tasks.get(provisional_key) is not current_task:
                    return
                pending = self._provisional_messages.get(provisional_key)
            if not pending:
                return
            candidate_chat_guid = provisional_key[2] or None
            chat_guid = await self._verify_inbound_message_membership(
                provisional_key[0], candidate_chat_guid
            )
            if not chat_guid:
                async with self._inbound_dedup_lock:
                    if self._provisional_message_tasks.get(provisional_key) is current_task:
                        self._provisional_messages.pop(provisional_key, None)
                return
            async with self._inbound_dedup_lock:
                if self._provisional_message_tasks.get(provisional_key) is not current_task:
                    return
                pending = self._provisional_messages.get(provisional_key)
            if not pending:
                return
            payload, record = pending
            resolved_record = {
                **record,
                "chatGuid": chat_guid,
                "chats": [{"guid": chat_guid}],
            }
            resolved_payload = dict(payload)
            if isinstance(resolved_payload.get("data"), dict):
                resolved_payload["data"] = resolved_record
            elif isinstance(resolved_payload.get("message"), dict):
                resolved_payload["message"] = resolved_record
            else:
                resolved_payload.update(resolved_record)
            await self._handle_webhook(
                None,
                _trusted_payload=resolved_payload,
                _authoritative_chat_guid=chat_guid,
            )
        finally:
            async with self._inbound_dedup_lock:
                if self._provisional_message_tasks.get(provisional_key) is current_task:
                    self._provisional_messages.pop(provisional_key, None)
                    self._provisional_message_tasks.pop(provisional_key, None)

    async def _queue_provisional_message(
        self,
        provisional_key: _ProvisionalMessageKey,
        payload: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """Hold one ambiguous revision without assigning it to the sender's DM."""
        async with self._inbound_dedup_lock:
            self._provisional_messages[provisional_key] = (
                dict(payload),
                dict(record),
            )
            self._provisional_messages.move_to_end(provisional_key)
            while len(self._provisional_messages) > _MESSAGE_DEDUP_SIZE:
                evicted_key, _ = self._provisional_messages.popitem(last=False)
                evicted_task = self._provisional_message_tasks.pop(evicted_key, None)
                if evicted_task and not evicted_task.done():
                    evicted_task.cancel()

    async def _schedule_provisional_retry(
        self, provisional_key: _ProvisionalMessageKey
    ) -> None:
        async with self._inbound_dedup_lock:
            if provisional_key not in self._provisional_messages:
                return
            prior_task = self._provisional_message_tasks.get(provisional_key)
            if prior_task and not prior_task.done():
                return
            task = asyncio.create_task(self._expire_provisional_message(provisional_key))
            self._provisional_message_tasks[provisional_key] = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _take_provisional_message(
        self,
        provisional_key: _ProvisionalMessageKey,
        *,
        allow_unclaimed_chat: bool = False,
    ) -> Optional[Dict[str, Any]]:
        async with self._inbound_dedup_lock:
            if provisional_key not in self._provisional_messages and allow_unclaimed_chat:
                unclaimed_key = (provisional_key[0], provisional_key[1], "")
                if unclaimed_key in self._provisional_messages:
                    provisional_key = unclaimed_key
            pending = self._provisional_messages.pop(provisional_key, None)
            task = self._provisional_message_tasks.pop(provisional_key, None)
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()
            return pending[1] if pending else None

    @staticmethod
    def _merge_provisional_record(
        authoritative: Dict[str, Any], provisional: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill content gaps without replacing authoritative chat membership."""
        merged = dict(authoritative)
        for field in (
            "text",
            "message",
            "body",
            "handle",
            "sender",
            "from",
            "address",
            "threadOriginatorGuid",
            "associatedMessageGuid",
        ):
            if not merged.get(field) and provisional.get(field):
                merged[field] = provisional[field]
        attachments: List[Dict[str, Any]] = []
        attachment_indexes: Dict[str, int] = {}
        for candidate in (
            provisional.get("attachments") or [],
            authoritative.get("attachments") or [],
        ):
            for attachment in candidate:
                if not isinstance(attachment, dict):
                    continue
                attachment_guid = BlueBubblesAdapter._value(attachment.get("guid"))
                if attachment_guid in attachment_indexes:
                    index = attachment_indexes[attachment_guid]
                    attachments[index] = {**attachments[index], **attachment}
                else:
                    if attachment_guid:
                        attachment_indexes[attachment_guid] = len(attachments)
                    attachments.append(dict(attachment))
        if attachments:
            merged["attachments"] = attachments
        return merged

    async def _lookup_reply_context(
        self, chat_guid: str, message_guid: str
    ) -> tuple[Optional[str], bool, List[str], List[str]]:
        """Hydrate one exact replied-to message, failing closed on locality."""
        if not chat_guid or not message_guid:
            return None, False, [], []
        try:
            record = await self._lookup_referenced_message(
                chat_guid, message_guid, include_attachments=True
            )
            if record is None:
                return None, False, [], []

            paths: List[str] = []
            types: List[str] = []
            attachments = record.get("attachments") or []
            for attachment in attachments[:_REPLY_ATTACHMENT_LIMIT]:
                if not isinstance(attachment, dict):
                    continue
                attachment_guid = self._value(attachment.get("guid"))
                if not attachment_guid:
                    continue
                cached = await self._download_attachment(
                    attachment_guid, attachment
                )
                if cached:
                    paths.append(cached)
                    types.append((attachment.get("mimeType") or "").lower())
            return (
                self._value(record.get("text")),
                bool(record.get("isFromMe")),
                paths,
                types,
            )
        except Exception as exc:
            logger.debug("[bluebubbles] reply context lookup failed: %s", exc)
            return None, False, [], []


    # ------------------------------------------------------------------
    # Webhook handling
    # ------------------------------------------------------------------

    def _extract_payload_record(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
        if isinstance(payload.get("message"), dict):
            return payload.get("message")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @staticmethod
    def _tapback_target(raw_guid: Any) -> tuple[Optional[str], int]:
        """Return the exact target GUID and part index from an associated GUID."""
        if not isinstance(raw_guid, str) or not raw_guid.strip():
            return None, 0
        value = raw_guid.strip()
        match = re.fullmatch(r"p:(\d+)/(.+)", value)
        if match:
            return match.group(2), int(match.group(1))
        if value.startswith("bp:") and value[3:]:
            return value[3:], 0
        return value, 0

    @classmethod
    def _reply_target(cls, *candidates: Any) -> tuple[Optional[str], int]:
        """Parse a quoted-message reference without normalizing malformed IDs."""
        for candidate in candidates:
            if candidate is None or candidate == "":
                continue
            if not isinstance(candidate, str) or candidate != candidate.strip():
                return None, 0
            target, part_index = cls._tapback_target(candidate)
            if not target or target != target.strip():
                return None, 0
            return target, part_index
        return None, 0

    async def _dispatch_inbound_tapback(
        self,
        *,
        payload: Dict[str, Any],
        tapback: tuple[str, str],
        target_guid: str,
        part_index: int,
        session_chat_id: str,
        chat_identifier: Optional[str],
        sender: str,
        is_group: bool,
        message_guid: Optional[str],
    ) -> None:
        """Publish one already-authorized, exact-chat Tapback."""
        if not message_guid:
            logger.debug("[bluebubbles] rejected Tapback without source event ID")
            return
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=chat_identifier or sender,
            chat_type="group" if is_group else "dm",
            user_id=sender,
            user_name=self._participant_names.get(sender, sender),
            chat_id_alt=chat_identifier,
        )
        platform_handler = getattr(self, "_platform_event_handler", None)
        if platform_handler is None:
            return

        action, reaction = tapback
        try:
            operation = TapbackOperation(
                platform="bluebubbles",
                chat_id=session_chat_id,
                target_message_id=target_guid,
                sender_id=sender,
                reaction=TapbackType(reaction),
                action=TapbackAction(action),
                direction=TapbackDirection.INBOUND,
                source_event_id=message_guid,
                part_index=part_index,
            )
            operation = operation.transition_to(TapbackStatus.VALIDATED)
            operation = operation.transition_to(TapbackStatus.PENDING)
            operation = operation.transition_to(TapbackStatus.PROCESSING)
        except (TapbackValidationError, ValueError):
            logger.debug("[bluebubbles] rejected malformed Tapback", exc_info=True)
            return
        state_key = operation.state_key
        async with self._inbound_dedup_lock:
            prior_state = self._tapback_event_states.get(state_key)
            if (
                prior_state
                and prior_state[0].action is operation.action
                and prior_state[0].reaction is operation.reaction
            ):
                self._tapback_event_states.move_to_end(state_key)
                return
            self._tapback_event_serial += 1
            state_serial = self._tapback_event_serial
            self._tapback_event_states[state_key] = (operation, state_serial)
            self._tapback_event_states.move_to_end(state_key)
            self._prune_tapback_event_states_locked()
        event_payload = {
            "platform": "bluebubbles",
            "event_type": "reaction",
            "payload": {
                **operation.to_platform_payload(),
                "raw_event": payload,
            },
        }
        try:
            await platform_handler(event_payload, source)
            async with self._inbound_dedup_lock:
                current_state = self._tapback_event_states.get(state_key)
                if current_state and current_state[1] == state_serial:
                    self._tapback_event_states[state_key] = (
                        operation.transition_to(TapbackStatus.APPLIED),
                        state_serial,
                    )
                    self._tapback_event_states.move_to_end(state_key)
                    self._prune_tapback_event_states_locked()
        except Exception:
            async with self._inbound_dedup_lock:
                current_state = self._tapback_event_states.get(state_key)
                if current_state and current_state[1] == state_serial:
                    if prior_state is None:
                        self._tapback_event_states.pop(state_key, None)
                    else:
                        self._tapback_event_states[state_key] = prior_state
                        self._tapback_event_states.move_to_end(state_key)
                    self._prune_tapback_event_states_locked()
            logger.debug(
                "[bluebubbles] platform event dispatch failed",
                exc_info=True,
            )

    def _prune_tapback_event_states_locked(self) -> None:
        """Bound terminal Tapback history without evicting in-flight work."""
        if len(self._tapback_event_states) <= _MESSAGE_DEDUP_SIZE:
            return
        for state_key, (operation, _serial) in list(
            self._tapback_event_states.items()
        ):
            if operation.status is TapbackStatus.PROCESSING:
                continue
            self._tapback_event_states.pop(state_key, None)
            if len(self._tapback_event_states) <= _MESSAGE_DEDUP_SIZE:
                return

    @staticmethod
    def _is_retryable_inbound_error(exc: BaseException) -> bool:
        """Classify failures that are safe to retry before dispatch commits."""
        return bool(
            getattr(exc, "retryable", False)
            or isinstance(
                exc,
                (
                    ConnectionError,
                    TimeoutError,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                ),
            )
        )

    def _remember_terminal_identities_locked(
        self, identities: tuple[str, ...]
    ) -> None:
        for identity in identities:
            self._terminal_message_identities[identity] = None
            self._terminal_message_identities.move_to_end(identity)
        while len(self._terminal_message_identities) > _MESSAGE_DEDUP_SIZE:
            self._terminal_message_identities.popitem(last=False)

    async def _handle_reserved_message(
        self,
        event: MessageEvent,
        message_identity: Optional[str],
        coalesced_identities: tuple[str, ...] = (),
        message_key: Optional[_InboundMessageKey] = None,
        revision_serial: int = 0,
    ) -> None:
        """Run one revision with supersession-safe cancellation and retries.

        A stable message identity stays reserved for the entire bounded retry
        sequence. Newer revisions cancel only an older task for the same exact
        chat/sender/GUID key. Success is recorded only after the operation and
        a final latest-revision check; permanent or exhausted failures use a
        separate terminal cache so they cannot masquerade as completed work.
        """
        identities = tuple(
            dict.fromkeys(
                identity
                for identity in (message_identity, *coalesced_identities)
                if identity
            )
        )
        current_task = asyncio.current_task()
        join_task: Optional[asyncio.Task] = None
        cancel_task: Optional[asyncio.Task] = None
        if message_key and current_task is not None:
            async with self._inbound_dedup_lock:
                if any(
                    identity in self._terminal_message_identities
                    for identity in identities
                ):
                    self._pending_message_identities.difference_update(identities)
                    return
                active = self._active_message_dispatches.get(message_key)
                if active and not active[2].done():
                    if active[0] == revision_serial and active[1] == message_identity:
                        join_task = active[2]
                    elif active[0] < revision_serial:
                        cancel_task = active[2]
                if join_task is None:
                    self._active_message_dispatches[message_key] = (
                        revision_serial,
                        message_identity,
                        current_task,
                    )

        if join_task is not None:
            try:
                await asyncio.shield(join_task)
            except asyncio.CancelledError:
                pass
            return
        if cancel_task is not None:
            cancel_task.cancel()
            try:
                await asyncio.shield(cancel_task)
            except asyncio.CancelledError:
                pass

        try:
            for attempt in range(1, self._message_retry_max_attempts + 1):
                if message_key:
                    async with self._inbound_dedup_lock:
                        if self._message_revision_serials.get(message_key, 0) > revision_serial:
                            self._pending_message_identities.difference_update(identities)
                            return
                try:
                    await self.handle_message(event)
                except asyncio.CancelledError:
                    if identities:
                        async with self._inbound_dedup_lock:
                            self._pending_message_identities.difference_update(identities)
                    if (
                        message_key
                        and self._message_revision_serials.get(message_key, 0)
                        > revision_serial
                    ):
                        return
                    raise
                except Exception as exc:
                    retryable = self._is_retryable_inbound_error(exc)
                    if retryable and attempt < self._message_retry_max_attempts:
                        delay = self._message_retry_base_delay_seconds * (
                            2 ** (attempt - 1)
                        )
                        logger.warning(
                            "[bluebubbles] transient inbound dispatch failure "
                            "(attempt %d/%d, retrying in %.1fs): %s",
                            attempt,
                            self._message_retry_max_attempts,
                            delay,
                            exc,
                        )
                        try:
                            await asyncio.sleep(delay)
                        except asyncio.CancelledError:
                            async with self._inbound_dedup_lock:
                                self._pending_message_identities.difference_update(
                                    identities
                                )
                                superseded = bool(
                                    message_key
                                    and self._message_revision_serials.get(
                                        message_key, 0
                                    )
                                    > revision_serial
                                )
                            if superseded:
                                return
                            raise
                        continue
                    async with self._inbound_dedup_lock:
                        self._pending_message_identities.difference_update(identities)
                        self._remember_terminal_identities_locked(identities)
                    logger.exception(
                        "[bluebubbles] inbound dispatch failed permanently "
                        "after %d attempt(s)",
                        attempt,
                    )
                    return
                break

            async with self._inbound_dedup_lock:
                if (
                    message_key
                    and self._message_revision_serials.get(message_key, 0)
                    > revision_serial
                ):
                    self._pending_message_identities.difference_update(identities)
                    return
                self._pending_message_identities.difference_update(identities)
                for identity in identities:
                    self._seen_message_guids[identity] = None
                    self._seen_message_guids.move_to_end(identity)
                while len(self._seen_message_guids) > _MESSAGE_DEDUP_SIZE:
                    self._seen_message_guids.popitem(last=False)
                if message_key is not None:
                    media_context = self._message_revision_media.get(message_key)
                    if media_context and media_context[4] <= revision_serial:
                        self._message_revision_media.pop(message_key, None)
        finally:
            if message_key and current_task is not None:
                async with self._inbound_dedup_lock:
                    active = self._active_message_dispatches.get(message_key)
                    if active and active[2] is current_task:
                        self._active_message_dispatches.pop(message_key, None)

    async def _flush_message_revision(self, message_key: _InboundMessageKey) -> None:
        """Dispatch the latest completed webhook revision for one iMessage GUID."""
        current_task = asyncio.current_task()
        pending: Optional[
            tuple[MessageEvent, Optional[str], int, tuple[str, ...]]
        ] = None
        try:
            while True:
                await asyncio.sleep(self._message_revision_wait_seconds)
                async with self._inbound_dedup_lock:
                    if self._pending_message_revision_tasks.get(message_key) is not current_task:
                        return
                    if self._active_attachment_revisions.get(message_key, 0) > 0:
                        continue
                    pending = self._pending_message_revisions.pop(message_key, None)
                    # Once dispatch starts this task is no longer a debounce timer.
                    # A late revision may queue a follow-up, but must not cancel an
                    # already-running agent turn.
                    self._pending_message_revision_tasks.pop(message_key, None)
                    break
            if pending:
                await self._handle_reserved_message(
                    pending[0],
                    pending[1],
                    pending[3],
                    message_key,
                    pending[2],
                )
        finally:
            async with self._inbound_dedup_lock:
                if self._pending_message_revision_tasks.get(message_key) is current_task:
                    self._pending_message_revision_tasks.pop(message_key, None)

    @staticmethod
    def _merge_revision_media(
        event: MessageEvent,
        media_urls: List[str],
        media_types: List[str],
        message_type: MessageType,
        metadata: Dict[str, Any],
    ) -> None:
        """Add richer revision fields without replacing the recipient's text."""
        if len(media_urls) <= len(event.media_urls):
            return
        event.media_urls = list(media_urls)
        event.media_types = list(media_types)
        event.message_type = message_type
        event.metadata = {**metadata, **event.metadata}

    def _remember_revision_media_locked(
        self,
        message_key: _InboundMessageKey,
        event: MessageEvent,
        revision_serial: int,
    ) -> None:
        if not event.media_urls:
            return
        self._message_revision_media[message_key] = (
            list(event.media_urls),
            list(event.media_types),
            event.message_type,
            dict(event.metadata),
            revision_serial,
        )
        self._message_revision_media.move_to_end(message_key)
        while len(self._message_revision_media) > _MESSAGE_DEDUP_SIZE:
            self._message_revision_media.popitem(last=False)

    async def _queue_message_revision(
        self,
        message_key: _InboundMessageKey,
        event: MessageEvent,
        message_identity: Optional[str],
        revision_serial: int,
        *,
        attachment_revision: bool,
    ) -> None:
        """Debounce BlueBubbles revisions without losing richer prior fields."""
        async with self._inbound_dedup_lock:
            if attachment_revision:
                self._release_attachment_revision_locked(
                    message_key,
                    revision_serial,
                    preserve_identity=True,
                )

            prior = self._pending_message_revisions.get(message_key)
            if prior and prior[2] > revision_serial:
                prior_event = prior[0]
                if len(event.media_urls) > len(prior_event.media_urls):
                    self._merge_revision_media(
                        prior_event,
                        event.media_urls,
                        event.media_types,
                        event.message_type,
                        event.metadata,
                    )
                    self._remember_revision_media_locked(
                        message_key, event, revision_serial
                    )
                    identities = tuple(
                        dict.fromkeys(
                            identity
                            for identity in (*prior[3], message_identity)
                            if identity
                        )
                    )
                    self._pending_message_revisions[message_key] = (
                        prior_event,
                        prior[1],
                        prior[2],
                        identities,
                    )
                elif message_identity:
                    self._pending_message_identities.discard(message_identity)
                    self._seen_message_guids[message_identity] = None
                    self._seen_message_guids.move_to_end(message_identity)
                    while len(self._seen_message_guids) > _MESSAGE_DEDUP_SIZE:
                        self._seen_message_guids.popitem(last=False)
                return

            coalesced_identities: tuple[str, ...] = ()
            same_context = bool(
                prior
                and prior[0].source.user_id == event.source.user_id
                and prior[0].source.chat_id == event.source.chat_id
            )
            if same_context and prior and len(prior[0].media_urls) > len(event.media_urls):
                prior_event = prior[0]
                self._merge_revision_media(
                    event,
                    prior_event.media_urls,
                    prior_event.media_types,
                    prior_event.message_type,
                    prior_event.metadata,
                )
                coalesced_identities = tuple(
                    dict.fromkeys(
                        identity
                        for identity in (prior[1], *prior[3])
                        if identity
                    )
                )
            else:
                media_context = self._message_revision_media.get(message_key)
                if media_context:
                    self._merge_revision_media(event, *media_context[:4])
                    self._message_revision_media.move_to_end(message_key)

            if event.media_urls and event.text == "(attachment)":
                prior_text = ""
                if prior:
                    prior_event = prior[0]
                    if (
                        prior_event.source.user_id == event.source.user_id
                        and prior_event.source.chat_id == event.source.chat_id
                    ):
                        prior_text = prior_event.text
                if not prior_text:
                    prior_text = self._message_revision_text.get(message_key, "")
                if prior_text and prior_text != "(attachment)":
                    event.text = prior_text

            if event.text and event.text != "(attachment)":
                self._message_revision_text[message_key] = event.text
                self._message_revision_text.move_to_end(message_key)
                while len(self._message_revision_text) > _MESSAGE_DEDUP_SIZE:
                    self._message_revision_text.popitem(last=False)

            self._remember_revision_media_locked(
                message_key, event, revision_serial
            )

            superseded_identities = ()
            if prior and not coalesced_identities:
                superseded_identities = tuple(
                    identity
                    for identity in (prior[1], *prior[3])
                    if identity and identity != message_identity
                )
            for superseded_identity in superseded_identities:
                self._pending_message_identities.discard(superseded_identity)
                self._seen_message_guids[superseded_identity] = None
                self._seen_message_guids.move_to_end(superseded_identity)
            if superseded_identities:
                while len(self._seen_message_guids) > _MESSAGE_DEDUP_SIZE:
                    self._seen_message_guids.popitem(last=False)
            self._pending_message_revisions[message_key] = (
                event,
                message_identity,
                revision_serial,
                coalesced_identities,
            )
            prior_task = self._pending_message_revision_tasks.get(message_key)
            if prior_task and not prior_task.done():
                prior_task.cancel()
            task = asyncio.create_task(self._flush_message_revision(message_key))
            self._pending_message_revision_tasks[message_key] = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def _release_attachment_revision_now(
        self,
        message_key: _InboundMessageKey,
        revision_serial: int,
        *,
        preserve_identity: bool = False,
    ) -> None:
        lease = (message_key, revision_serial)
        if lease not in self._active_attachment_leases:
            return
        self._active_attachment_leases.discard(lease)
        leased_identity = self._active_attachment_identity_leases.pop(lease, None)
        if leased_identity and not preserve_identity:
            self._pending_message_identities.discard(leased_identity)
        active = self._active_attachment_revisions.get(message_key, 0)
        if active <= 1:
            self._active_attachment_revisions.pop(message_key, None)
        else:
            self._active_attachment_revisions[message_key] = active - 1

    def _release_attachment_revision_locked(
        self,
        message_key: _InboundMessageKey,
        revision_serial: int,
        *,
        preserve_identity: bool = False,
    ) -> None:
        self._release_attachment_revision_now(
            message_key,
            revision_serial,
            preserve_identity=preserve_identity,
        )

    async def _release_attachment_revision(
        self,
        message_key: Optional[_InboundMessageKey],
        revision_serial: int,
        *,
        preserve_identity: bool = False,
    ) -> None:
        if not message_key:
            return
        async with self._inbound_dedup_lock:
            self._release_attachment_revision_locked(
                message_key,
                revision_serial,
                preserve_identity=preserve_identity,
            )

    async def _rollback_uncommitted_revision(
        self,
        message_key: Optional[_InboundMessageKey],
        revision_serial: int,
    ) -> None:
        """Release a cancelled webhook lease without superseding prior work."""
        if not message_key:
            return
        async with self._inbound_dedup_lock:
            self._release_attachment_revision_locked(message_key, revision_serial)
            if self._message_revision_serials.get(message_key) == revision_serial:
                self._message_revision_serials[message_key] = max(
                    0, revision_serial - 1
                )

    async def _handle_webhook(
        self,
        request,
        *,
        _trusted_payload=None,
        _authoritative_chat_guid: Optional[str] = None,
    ):
        from aiohttp import web

        if _trusted_payload is None:
            token = (
                request.query.get("password")
                or request.query.get("guid")
                or request.headers.get("x-password")
                or request.headers.get("x-guid")
                or request.headers.get("x-bluebubbles-guid")
            )
            if token != self.password:
                return web.json_response({"error": "unauthorized"}, status=401)
            try:
                raw = await request.read()
                body = raw.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(body)
                except Exception:
                    from urllib.parse import parse_qs

                    form = parse_qs(body)
                    payload_str = (
                        form.get("payload")
                        or form.get("data")
                        or form.get("message")
                        or [""]
                    )[0]
                    payload = json.loads(payload_str) if payload_str else {}
            except Exception as exc:
                logger.error("[bluebubbles] webhook parse error: %s", exc)
                return web.json_response({"error": "invalid payload"}, status=400)
        else:
            payload = _trusted_payload

        event_type = self._value(payload.get("type"), payload.get("event")) or ""
        # Only process message events; silently acknowledge everything else
        if event_type and event_type not in _MESSAGE_EVENTS:
            return web.Response(text="ok")

        record = self._extract_payload_record(payload) or {}
        is_from_me = bool(
            record.get("isFromMe")
            or record.get("fromMe")
            or record.get("is_from_me")
        )
        if is_from_me:
            return web.Response(text="ok")

        # Classify Tapbacks now, but publish them only after the webhook's
        # source/chat admission fields have been validated below.
        assoc_type = record.get("associatedMessageType")
        tapback: Optional[tuple[str, str]] = None
        assoc_code: Optional[int] = None
        if isinstance(assoc_type, int) and not isinstance(assoc_type, bool):
            assoc_code = assoc_type
        elif isinstance(assoc_type, str) and assoc_type.strip().isdigit():
            assoc_code = int(assoc_type.strip())
        if assoc_code is not None:
            if assoc_code in _TAPBACK_ADDED:
                tapback = ("added", _TAPBACK_ADDED[assoc_code])
            elif assoc_code in _TAPBACK_REMOVED:
                tapback = ("removed", _TAPBACK_REMOVED[assoc_code])
        if assoc_type is not None and assoc_code != 0 and tapback is None:
            # A provider payload that explicitly claims an association type is
            # not safe to reinterpret as conversational text. Unknown future
            # reactions and malformed values must not wake the agent (and thus
            # cannot trigger an acknowledgement Tapback or automated reply).
            logger.debug(
                "[bluebubbles] ignoring unsupported associated message type: %r",
                assoc_type,
            )
            return web.Response(text="ok")

        message_guid = self._value(
            record.get("guid"),
            record.get("messageGuid"),
            record.get("id"),
        )
        text = (
            self._value(
                record.get("text"), record.get("message"), record.get("body")
            )
            or ""
        )

        # Resolve the exact message-to-chat membership before authorization or
        # any other inbound side effect. Webhook chat fields are candidates only.
        candidate_chat_guids = {
            guid
            for candidate in (
                record.get("chatGuid"),
                payload.get("chatGuid"),
                record.get("chat_guid"),
                payload.get("chat_guid"),
                *(
                    nested_candidate
                    for chat in (
                        *(record.get("chats") or []),
                        *(payload.get("chats") or []),
                    )
                    if isinstance(chat, dict)
                    for nested_candidate in (chat.get("guid"), chat.get("chatGuid"))
                ),
            )
            for guid in [self._value(candidate)]
            if guid
        }
        if len(candidate_chat_guids) > 1:
            return web.Response(text="ok")
        candidate_chat_guid = (
            next(iter(candidate_chat_guids)) if candidate_chat_guids else None
        )
        preauth_chat_identifier = self._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        preauth_sender = self._value(
            record.get("handle", {}).get("address")
            if isinstance(record.get("handle"), dict)
            else None,
            record.get("sender"),
            record.get("from"),
            record.get("address"),
        )
        if not message_guid:
            return web.Response(text="ok")

        provisional_key = None
        if preauth_sender:
            provisional_key = (
                message_guid,
                str(preauth_sender),
                candidate_chat_guid or "",
            )
        provisional_record: Optional[Dict[str, Any]] = None

        if _authoritative_chat_guid is not None:
            if candidate_chat_guid != _authoritative_chat_guid:
                return web.Response(text="ok")
            preauth_chat_guid = _authoritative_chat_guid
        elif candidate_chat_guid is None:
            if not provisional_key:
                return web.Response(text="ok")
            await self._queue_provisional_message(
                provisional_key,
                payload,
                record,
            )
            try:
                preauth_chat_guid = await self._verify_inbound_message_membership(
                    message_guid,
                    None,
                )
            except BaseException:
                await self._take_provisional_message(provisional_key)
                raise
            if not preauth_chat_guid:
                await self._schedule_provisional_retry(provisional_key)
                return web.Response(text="ok")
            provisional_record = await self._take_provisional_message(provisional_key)
            if provisional_record is None:
                return web.Response(text="ok")
        else:
            preauth_chat_guid = await self._verify_inbound_message_membership(
                message_guid,
                candidate_chat_guid,
            )
            if not preauth_chat_guid:
                if provisional_key:
                    await self._queue_provisional_message(
                        provisional_key,
                        payload,
                        record,
                    )
                    await self._schedule_provisional_retry(provisional_key)
                return web.Response(text="ok")

        if not provisional_key:
            return web.Response(text="ok")
        if candidate_chat_guid and provisional_record is None:
            provisional_record = await self._take_provisional_message(
                provisional_key,
                allow_unclaimed_chat=True,
            )
        elif provisional_record is None:
            provisional_record = await self._take_provisional_message(provisional_key)
        if provisional_record:
            record = self._merge_provisional_record(record, provisional_record)
            preauth_sender = (
                self._value(
                    record.get("handle", {}).get("address")
                    if isinstance(record.get("handle"), dict)
                    else None,
                    record.get("sender"),
                    record.get("from"),
                    record.get("address"),
                )
                or preauth_sender
            )
            text = (
                self._value(
                    record.get("text"),
                    record.get("message"),
                    record.get("body"),
                )
                or ""
            )
        preauth_chat_id = preauth_chat_guid or preauth_chat_identifier
        preauth_is_group = bool(record.get("isGroup")) or (
            ";+;" in (preauth_chat_guid or "")
        )
        authorized = self._is_sender_authorized(
            preauth_sender,
            "group" if preauth_is_group else "dm",
            preauth_chat_id,
        )
        if (tapback and authorized is not True) or (
            not tapback
            and self._authorization_check is not None
            and authorized is not True
        ):
            return web.Response(text="ok")

        if tapback:
            if not preauth_sender or not preauth_chat_id:
                return web.json_response({"error": "missing message fields"}, status=400)
            target_guid, part_index = self._tapback_target(
                record.get("associatedMessageGuid")
            )
            if not target_guid:
                return web.Response(text="ok")
            target = await self._query_exact_message(
                str(preauth_chat_id), target_guid
            )
            if target is None:
                return web.Response(text="ok")
            await self._dispatch_inbound_tapback(
                payload=payload,
                tapback=tapback,
                target_guid=target_guid,
                part_index=part_index,
                session_chat_id=str(preauth_chat_id),
                chat_identifier=preauth_chat_identifier,
                sender=str(preauth_sender),
                is_group=preauth_is_group,
                message_guid=message_guid,
            )
            return web.Response(text="ok")

        # BlueBubbles does not provide a monotonic revision number on every
        # webhook shape. The stable message GUID is therefore the canonical
        # logical identity. An update is newer than the corresponding create
        # event even when webhooks arrive out of order. When both updates carry
        # dateEdited, that provider timestamp orders them; otherwise same-kind
        # events retain arrival order and content-based idempotency below.
        event_rank = 1 if event_type == "updated-message" else 0
        raw_edit_time = record.get("dateEdited")
        edit_time: Optional[float] = None
        if isinstance(raw_edit_time, (int, float)) and not isinstance(
            raw_edit_time, bool
        ):
            edit_time = float(raw_edit_time)
        elif isinstance(raw_edit_time, str):
            try:
                edit_time = float(raw_edit_time.strip())
            except ValueError:
                pass
        async with self._inbound_dedup_lock:
            prior_order = self._message_revision_orders.get(message_guid)
            if prior_order is not None:
                prior_rank, prior_edit_time = prior_order
                if event_rank < prior_rank or (
                    event_rank == prior_rank == 1
                    and edit_time is not None
                    and prior_edit_time is not None
                    and edit_time < prior_edit_time
                ):
                    self._message_revision_orders.move_to_end(message_guid)
                    return web.Response(text="ok")
                if event_rank == prior_rank and edit_time is None:
                    edit_time = prior_edit_time
            self._message_revision_orders[message_guid] = (event_rank, edit_time)
            self._message_revision_orders.move_to_end(message_guid)
            while len(self._message_revision_orders) > _MESSAGE_DEDUP_SIZE:
                self._message_revision_orders.popitem(last=False)

        message_key: Optional[_InboundMessageKey] = None
        revision_chat_id = preauth_chat_guid
        if not revision_chat_id and preauth_chat_identifier:
            revision_chat_id = (
                f"iMessage;-;{preauth_chat_identifier}"
                if not preauth_is_group
                else preauth_chat_identifier
            )
        if message_guid and revision_chat_id and preauth_sender:
            message_key = (
                str(revision_chat_id),
                str(preauth_sender),
                message_guid,
            )

        revision_serial = 0
        if message_key:
            async with self._inbound_dedup_lock:
                pending_attachment_task = self._pending_attachment_tasks.get(message_key)
                if (
                    pending_attachment_task
                    and pending_attachment_task is not asyncio.current_task()
                    and not pending_attachment_task.done()
                ):
                    pending_attachment_task.cancel()
                    self._pending_attachment_tasks.pop(message_key, None)
                revision_serial = self._message_revision_serials.get(message_key, 0) + 1
                self._message_revision_serials[message_key] = revision_serial
                self._message_revision_serials.move_to_end(message_key)
                while len(self._message_revision_serials) > _MESSAGE_DEDUP_SIZE:
                    evicted = False
                    for candidate in list(self._message_revision_serials):
                        if (
                            candidate not in self._pending_message_revisions
                            and candidate not in self._active_attachment_revisions
                        ):
                            self._message_revision_serials.pop(candidate, None)
                            evicted = True
                            break
                    if not evicted:
                        break

        # --- Inbound attachment handling ---
        attachments = record.get("attachments") or []
        media_urls: List[str] = []
        media_types: List[str] = []
        downloaded_attachment_guids: set[str] = set()
        msg_type = MessageType.TEXT
        attachment_revision = bool(message_key and attachments)
        if attachment_revision and message_key:
            async with self._inbound_dedup_lock:
                self._active_attachment_revisions[message_key] = (
                    self._active_attachment_revisions.get(message_key, 0) + 1
                )
                self._active_attachment_leases.add((message_key, revision_serial))
            request_task = asyncio.current_task()
            if request_task is not None:
                request_task.add_done_callback(
                    lambda _task, key=message_key, serial=revision_serial: (
                        self._release_attachment_revision_now(key, serial)
                    )
                )

        try:
            materialized = await materialize_attachments(
                attachments,
                self._download_attachment,
            )
        except BaseException:
            if attachment_revision:
                await self._rollback_uncommitted_revision(
                    message_key, revision_serial
                )
            raise
        if materialized.readiness is AttachmentReadiness.TERMINAL_FAILURE:
            if attachment_revision:
                await self._release_attachment_revision(message_key, revision_serial)
            logger.warning(
                "[bluebubbles] terminal attachment failure for %s",
                _redact(materialized.failed_guid or message_guid or "unknown"),
            )
            return web.Response(text="ok")
        if materialized.readiness is AttachmentReadiness.PENDING:
            if attachment_revision:
                await self._release_attachment_revision(message_key, revision_serial)
            if message_key and preauth_chat_guid:
                await schedule_pending_attachment_retry(
                    self,
                    message_key,
                    revision_serial,
                    payload,
                    record,
                    preauth_chat_guid,
                    attempts=_ATTACHMENT_RETRY_ATTEMPTS,
                )
            return web.Response(text="ok")

        media_urls.extend(materialized.paths)
        media_types.extend(materialized.mime_types)
        downloaded_attachment_guids.update(
            str(att.get("guid"))
            for att in attachments
            if isinstance(att, dict) and att.get("guid")
        )
        for att, mime in zip(attachments, media_types):
            if mime.startswith("image/"):
                msg_type = MessageType.PHOTO
            elif mime.startswith("audio/") or (
                isinstance(att, dict) and (att.get("uti") or "").endswith("caf")
            ):
                msg_type = MessageType.VOICE
            elif mime.startswith("video/"):
                msg_type = MessageType.VIDEO
            else:
                msg_type = MessageType.DOCUMENT

        # With multiple attachments, prefer PHOTO if any images present
        if len(media_urls) > 1:
            mime_prefixes = {(m or "").split("/")[0] for m in media_types}
            if "image" in mime_prefixes:
                msg_type = MessageType.PHOTO

        if not text and media_urls:
            text = "(attachment)"
        # --- End attachment handling ---

        chat_guid = preauth_chat_guid
        chat_identifier = preauth_chat_identifier
        sender = preauth_sender
        if not sender or not (chat_guid or chat_identifier) or (not text and not tapback):
            if attachment_revision:
                await self._release_attachment_revision(message_key, revision_serial)
            return web.json_response({"error": "missing message fields"}, status=400)

        session_chat_id = chat_guid or chat_identifier
        is_group = preauth_is_group
        if is_group and self.require_mention:
            if self._message_matches_mention_patterns(text):
                text = self._clean_mention_text(text)
                if message_key:
                    async with self._inbound_dedup_lock:
                        self._accepted_group_captions[message_key] = text
                        self._accepted_group_captions.move_to_end(message_key)
                        while len(self._accepted_group_captions) > _MESSAGE_DEDUP_SIZE:
                            self._accepted_group_captions.popitem(last=False)
            else:
                inherited_text = ""
                if message_key and media_urls:
                    async with self._inbound_dedup_lock:
                        accepted = self._accepted_group_captions.get(message_key)
                        if accepted:
                            inherited_text = accepted
                            self._accepted_group_captions.move_to_end(message_key)
                        else:
                            pending = self._pending_message_revisions.get(message_key)
                            if pending:
                                pending_event = pending[0]
                                if (
                                    pending_event.source.user_id == sender
                                    and pending_event.source.chat_id == session_chat_id
                                ):
                                    inherited_text = pending_event.text
                if not inherited_text:
                    logger.debug(
                        "[bluebubbles] ignoring group message (require_mention=true, no mention pattern matched)"
                    )
                    if attachment_revision:
                        await self._release_attachment_revision(message_key, revision_serial)
                    return web.Response(text="ok")
                text = inherited_text
        message_identity: Optional[str] = None
        if message_key:
            # Keep revision identity independent of delivery bookkeeping and
            # mutable webhook timestamps. Attachment GUID plus stable visible
            # descriptors identify the logical media; readiness is included so
            # a successful retry is not hidden by an earlier failed download.
            attachment_identity = [
                {
                    "guid": att.get("guid"),
                    "id": att.get("id"),
                    "mimeType": att.get("mimeType"),
                    "uti": att.get("uti"),
                    "transferName": att.get("transferName"),
                    "totalBytes": att.get("totalBytes"),
                    "__hermes_downloaded": att.get("guid")
                    in downloaded_attachment_guids,
                }
                for att in attachments
                if isinstance(att, dict)
            ]
            revision_payload = json.dumps(
                {"text": text, "attachments": attachment_identity},
                sort_keys=True,
                separators=(",", ":"),
            )
            revision_hash = hashlib.sha256(revision_payload.encode("utf-8")).hexdigest()
            message_identity = "\0".join((*message_key, revision_hash))
            async with self._inbound_dedup_lock:
                if message_identity in self._pending_message_identities:
                    # Identity is computed only after attachment materialization,
                    # but reserving a revision serial happens before that work so
                    # an actual newer revision can supersede it. A duplicate must
                    # not leave that provisional serial behind: doing so makes the
                    # already-queued owner look stale and suppresses its dispatch.
                    if self._message_revision_serials.get(message_key) == revision_serial:
                        self._message_revision_serials[message_key] = max(
                            0, revision_serial - 1
                        )
                    if attachment_revision:
                        self._release_attachment_revision_locked(message_key, revision_serial)
                    return web.Response(text="ok")
                if message_identity in self._seen_message_guids:
                    self._seen_message_guids.move_to_end(message_identity)
                    if self._message_revision_serials.get(message_key) == revision_serial:
                        self._message_revision_serials[message_key] = max(
                            0, revision_serial - 1
                        )
                    if attachment_revision:
                        self._release_attachment_revision_locked(message_key, revision_serial)
                    return web.Response(text="ok")
                self._pending_message_identities.add(message_identity)
                if attachment_revision:
                    self._active_attachment_identity_leases[
                        (message_key, revision_serial)
                    ] = message_identity
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=chat_identifier or sender,
            chat_type="group" if is_group else "dm",
            user_id=sender,
            user_name=self._participant_names.get(sender, sender),
            chat_id_alt=chat_identifier,
        )
        reply_to_message_id, _reply_part_index = self._reply_target(
            record.get("threadOriginatorGuid"),
            record.get("associatedMessageGuid"),
        )
        reply_to_text: Optional[str] = None
        reply_to_is_own_message = False
        reply_attachment_count = 0
        if reply_to_message_id:
            try:
                (
                    reply_to_text,
                    reply_to_is_own_message,
                    reply_paths,
                    reply_types,
                ) = await self._lookup_reply_context(
                    str(session_chat_id), reply_to_message_id
                )
            except BaseException:
                if attachment_revision:
                    await self._release_attachment_revision(
                        message_key, revision_serial
                    )
                elif message_identity:
                    async with self._inbound_dedup_lock:
                        self._pending_message_identities.discard(message_identity)
                raise
            if reply_paths:
                media_urls.extend(reply_paths)
                media_types.extend(reply_types)
                reply_attachment_count = len(reply_paths)
        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=payload,
            message_id=message_guid,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            reply_to_is_own_message=reply_to_is_own_message,
            media_urls=media_urls,
            media_types=media_types,
            metadata=(
                {"bluebubbles_reply_attachment_count": reply_attachment_count}
                if reply_attachment_count
                else {}
            ),
        )
        if message_key and self._message_revision_wait_seconds > 0:
            try:
                await self._queue_message_revision(
                    message_key,
                    event,
                    message_identity,
                    revision_serial,
                    attachment_revision=attachment_revision,
                )
            except BaseException:
                if attachment_revision:
                    await self._rollback_uncommitted_revision(
                        message_key, revision_serial
                    )
                elif message_identity:
                    async with self._inbound_dedup_lock:
                        self._pending_message_identities.discard(message_identity)
                        if (
                            self._message_revision_serials.get(message_key)
                            == revision_serial
                        ):
                            self._message_revision_serials[message_key] = max(
                                0, revision_serial - 1
                            )
                raise
        else:
            if attachment_revision:
                await self._release_attachment_revision(
                    message_key,
                    revision_serial,
                    preserve_identity=True,
                )
            task = asyncio.create_task(
                self._handle_reserved_message(event, message_identity)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Queue only after authorization and group mention admission. The queue
        # retries helper cold boots while retaining the same message identity.
        await self._queue_read_receipt(
            str(session_chat_id),
            message_guid,
            is_group=is_group,
            admitted=True,
        )

        return web.Response(text="ok")
