"""BlueBubbles iMessage platform adapter.

Uses the local BlueBubbles macOS server for outbound REST sends and inbound
webhooks.  Supports text messaging, media attachments (images, voice, video,
documents), tapback reactions, typing indicators, and read receipts.

Architecture based on PR #5869 (benjaminsehl) with inbound attachment
downloading from PR #4588 (YuhangLin).
"""

import asyncio
import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

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
from .media_cache import ext_for_mime
from gateway.platforms.helpers import (
    MessageDeduplicator,
    compile_mention_patterns,
    strip_markdown,
)

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
_VALID_REACTIONS = {"love", "like", "dislike", "laugh", "emphasize", "question"}
_REACTION_ALIASES = {
    "loved": "love",
    "liked": "like",
    "disliked": "dislike",
    "laughed": "laugh",
    "emphasized": "emphasize",
    "questioned": "question",
}

# Only new-message (plus the legacy message alias) starts an agent turn.
# BlueBubbles emits updated-message for receipt, delivery, and attachment state
# changes, often with a different chat GUID shape for the same iMessage.
_MESSAGE_EVENTS = {"new-message", "message"}
_WEBHOOK_EVENTS = ["new-message"]

# BlueBubbles sends each webhook once and only logs non-2xx responses. Retry
# attachment downloads inside that one delivery rather than relying on the
# provider to redeliver the webhook.
_ATTACHMENT_RETRY_DELAYS = (0.25, 0.75)

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


def _bool_setting(value: Any, default: bool = False) -> bool:
    """Parse config booleans without treating the string ``false`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
        self.send_read_receipts = _bool_setting(
            extra.get("send_read_receipts"), default=True
        )
        self.typing_indicators = _bool_setting(
            extra.get("typing_indicators"),
            default=getattr(config, "typing_indicator", True),
        )
        self.config.typing_indicator = self.typing_indicators
        try:
            self.typing_refresh_interval = max(
                1.0, float(extra.get("typing_refresh_interval", 4.0))
            )
        except (TypeError, ValueError):
            self.typing_refresh_interval = 4.0
        self.auto_react = _bool_setting(extra.get("auto_react"), default=True)
        configured_reaction = str(extra.get("auto_react_type") or "like").strip().lower()
        configured_reaction = _REACTION_ALIASES.get(
            configured_reaction, configured_reaction
        )
        if configured_reaction not in _VALID_REACTIONS:
            logger.warning(
                "[bluebubbles] invalid auto_react_type %r; using 'like'",
                configured_reaction,
            )
            configured_reaction = "like"
        self.auto_react_type = configured_reaction
        self.split_paragraph_replies = _bool_setting(
            extra.get("split_paragraph_replies"), default=False
        )
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("BLUEBUBBLES_REQUIRE_MENTION")
        self.require_mention = str(_require_mention).strip().lower() in {"true", "1", "yes", "on"}
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("BLUEBUBBLES_MENTION_PATTERNS")
        )
        self.client: Optional[httpx.AsyncClient] = None
        self._runner = None
        self._private_api_enabled: Optional[bool] = None
        self._helper_connected: bool = False
        self._guid_cache: OrderedDict[str, str] = OrderedDict()
        self._message_dedup = MessageDeduplicator(ttl_seconds=300)
        self._inflight_message_ids: Dict[str, asyncio.Future] = {}

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.server_url or not self.password:
            logger.error(
                "[bluebubbles] BLUEBUBBLES_SERVER_URL and BLUEBUBBLES_PASSWORD are required"
            )
            return False
        from aiohttp import web

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
        except Exception as exc:
            logger.error(
                "[bluebubbles] cannot reach server at %s: %s", self.server_url, exc
            )
            if self.client:
                await self.client.aclose()
                self.client = None
            return False

        # Explicit body cap: BlueBubbles webhook events are small JSON (or
        # form-encoded) payloads. client_max_size makes aiohttp enforce the
        # cap on every read path — including chunked requests that carry no
        # Content-Length (same pattern as webhook.py / raft, #58536/#58902).
        app = web.Application(client_max_size=_WEBHOOK_MAX_BODY_BYTES)

        async def health(_request):
            return web.Response(text="ok")

        app.router.add_get("/health", health)
        app.router.add_post(self.webhook_path, self._handle_webhook)
        # The webhook auth value is carried in the query string because the
        # BlueBubbles webhook API cannot send custom headers. Do not let
        # aiohttp access logs write that request target to agent.log.
        self._runner = web.AppRunner(app, access_log=None)
        try:
            await self._runner.setup()
            site = web.TCPSite(
                self._runner, self.webhook_host, self.webhook_port
            )
            await site.start()
        except asyncio.CancelledError:
            await asyncio.shield(self._cleanup_local_resources())
            raise
        except Exception as exc:
            logger.error(
                "[bluebubbles] failed to start webhook listener on %s:%s: %s",
                self.webhook_host,
                self.webhook_port,
                exc,
            )
            await self._cleanup_local_resources()
            return False
        logger.info(
            "[bluebubbles] webhook listening on http://%s:%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
        )

        # Inbound delivery is not healthy until BlueBubbles accepts the
        # registration. Registration/migration is cancellation-safe and server
        # ownership is reconciled before local resources are released.
        try:
            registered = await self._register_webhook()
        except asyncio.CancelledError:
            try:
                await self._unregister_webhook()
            finally:
                await self._cleanup_local_resources()
            raise
        if not registered:
            logger.error("[bluebubbles] webhook registration failed")
            # Do not unregister here: failure does not prove ownership of a
            # same-URL server registration.
            await self._cleanup_local_resources()
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        await self._unregister_webhook()
        await self._cleanup_local_resources()

    async def _cleanup_local_resources(self) -> None:
        """Close local HTTP resources without changing server registrations."""
        client, self.client = self.client, None
        if client:
            try:
                await client.aclose()
            except Exception as exc:
                logger.debug("[bluebubbles] HTTP client cleanup failed: %s", exc)

        runner, self._runner = self._runner, None
        if runner:
            try:
                await runner.cleanup()
            except Exception as exc:
                logger.debug("[bluebubbles] webhook runner cleanup failed: %s", exc)
        self._mark_disconnected()

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

    async def _find_registered_webhooks(
        self, url: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Return same-URL registrations, or ``None`` when lookup fails."""
        try:
            response = await self._api_get("/api/v1/webhook")
            data = response.get("data")
            if isinstance(data, list):
                return [webhook for webhook in data if webhook.get("url") == url]
            logger.warning("[bluebubbles] webhook lookup returned invalid data")
        except Exception as exc:
            logger.warning("[bluebubbles] failed to list registered webhooks: %s", exc)
        return None

    async def _remove_registered_webhooks(
        self, webhooks: List[Dict[str, Any]]
    ) -> bool:
        """Best-effort cleanup after a working registration is available."""
        assert self.client is not None
        removed_all = True
        for webhook in webhooks:
            webhook_id = webhook.get("id")
            if not webhook_id:
                removed_all = False
                continue
            try:
                response = await self.client.delete(
                    self._api_url(f"/api/v1/webhook/{webhook_id}")
                )
                response.raise_for_status()
            except Exception as exc:
                removed_all = False
                logger.warning(
                    "[bluebubbles] failed to remove stale webhook registration: %s",
                    exc,
                )
        return removed_all

    async def _post_webhook_registration(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Let an ambiguous idempotent POST settle before cancellation escapes."""
        post = asyncio.create_task(self._api_post("/api/v1/webhook", payload))
        try:
            return await asyncio.shield(post)
        except asyncio.CancelledError as cancelled:
            try:
                await post
            except Exception as exc:
                logger.warning(
                    "[bluebubbles] registration failed while connect was cancelled: %s",
                    exc,
                )
            raise cancelled

    async def _migrate_webhook_registration(
        self,
        webhook_url: str,
        existing: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """Replace stale same-URL hooks with rollback and ownership safety.

        BlueBubbles registration POST is idempotent by URL and does not update
        an existing row's events. The stale row must therefore be removed
        before the desired event set can be created.
        """
        assert self.client is not None
        if any(webhook.get("id") is None for webhook in existing):
            logger.warning("[bluebubbles] stale webhook has no ID; migration aborted")
            return False

        removed: List[Dict[str, Any]] = []
        for webhook in existing:
            webhook_id = webhook.get("id")
            try:
                response = await self.client.delete(
                    self._api_url(f"/api/v1/webhook/{webhook_id}")
                )
                response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "[bluebubbles] failed to remove stale webhook registration: %s",
                    exc,
                )
                if removed:
                    current = await self._find_registered_webhooks(webhook_url)
                    if current == []:
                        rollback_events = list(
                            removed[0].get("events") or _WEBHOOK_EVENTS
                        )
                        try:
                            await self._api_post(
                                "/api/v1/webhook",
                                {"url": webhook_url, "events": rollback_events},
                            )
                        except Exception as rollback_exc:
                            logger.error(
                                "[bluebubbles] failed to restore webhook after "
                                "partial stale cleanup: %s",
                                rollback_exc,
                            )
                return False
            removed.append(webhook)

        try:
            response = await self._post_webhook_registration(payload)
            status = response.get("status", 0)
            data = response.get("data")
            returned_events = data.get("events") if isinstance(data, dict) else None
            if not isinstance(status, int) or not 200 <= status < 300:
                raise RuntimeError(f"replacement returned status {status}")
            if returned_events is not None and set(returned_events) != set(
                payload["events"]
            ):
                raise RuntimeError("replacement retained the stale event set")
        except Exception as exc:
            logger.warning(
                "[bluebubbles] webhook replacement failed; reconciling state: %s",
                exc,
            )
            current = await self._find_registered_webhooks(webhook_url)
            if current is None:
                logger.error(
                    "[bluebubbles] cannot verify webhook state after replacement "
                    "failure; rollback skipped to avoid changing an unknown owner"
                )
                return False

            expected_events = set(payload["events"])
            exact = [
                webhook
                for webhook in current
                if set(webhook.get("events") or []) == expected_events
            ]
            if exact:
                logger.info(
                    "[bluebubbles] replacement committed despite local failure"
                )
                return True

            if current:
                logger.error(
                    "[bluebubbles] webhook URL is occupied after replacement "
                    "failure; rollback skipped rather than deleting an unowned hook"
                )
                return False

            rollback_events = list(removed[0].get("events") or _WEBHOOK_EVENTS)
            try:
                rollback = await self._api_post(
                    "/api/v1/webhook",
                    {"url": webhook_url, "events": rollback_events},
                )
                rollback_status = rollback.get("status", 0)
                if not isinstance(rollback_status, int) or not 200 <= rollback_status < 300:
                    raise RuntimeError(
                        f"rollback returned status {rollback_status}"
                    )
            except Exception as rollback_exc:
                logger.error(
                    "[bluebubbles] failed to restore prior webhook registration: %s",
                    rollback_exc,
                )
            return False

        logger.info(
            "[bluebubbles] webhook registration migrated: %s",
            self._webhook_register_url_for_log,
        )
        return True

    async def _run_webhook_migration(
        self,
        webhook_url: str,
        existing: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """Finish an ownership-sensitive migration before cancellation escapes."""
        migration = asyncio.create_task(
            self._migrate_webhook_registration(webhook_url, existing, payload)
        )
        try:
            return await asyncio.shield(migration)
        except asyncio.CancelledError as cancelled:
            try:
                await migration
            except Exception as exc:
                logger.warning(
                    "[bluebubbles] migration failed during cancellation: %s", exc
                )
            raise cancelled

    async def _register_webhook(self) -> bool:
        """Ensure one same-URL registration has the exact desired event set."""
        if not self.client:
            return False

        webhook_url = self._webhook_register_url
        existing = await self._find_registered_webhooks(webhook_url)
        if existing is None:
            return False

        expected_events = set(_WEBHOOK_EVENTS)
        exact = [
            webhook
            for webhook in existing
            if set(webhook.get("events") or []) == expected_events
        ]
        if exact:
            logger.info(
                "[bluebubbles] webhook already registered: %s",
                self._webhook_register_url_for_log,
            )
            keep = exact[0]
            stale = [webhook for webhook in existing if webhook is not keep]
            if stale:
                await self._remove_registered_webhooks(stale)
            return True

        payload = {"url": webhook_url, "events": list(_WEBHOOK_EVENTS)}
        if existing:
            return await self._run_webhook_migration(webhook_url, existing, payload)

        try:
            response = await self._post_webhook_registration(payload)
            status = response.get("status", 0)
            if isinstance(status, int) and 200 <= status < 300:
                logger.info(
                    "[bluebubbles] webhook registered with server: %s",
                    self._webhook_register_url_for_log,
                )
                return True
            logger.warning(
                "[bluebubbles] webhook registration returned status %s: %s",
                status,
                response.get("message"),
            )
        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to register webhook with server: %s", exc
            )
        return False

    async def _unregister_webhook(self) -> bool:
        """Keep the fixed-URL registration durable across gateway reconnects.

        BlueBubbles POST is idempotent by URL, so neither its response nor a
        subsequent lookup can prove which concurrent process created the row.
        Deleting it on disconnect could remove another process's registration.
        """
        return False

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
        self, address: str, message: str
    ) -> SendResult:
        """Create a new chat by sending the first message to *address*."""
        payload = {
            "addresses": [address],
            "message": message,
            "tempGuid": f"temp-{datetime.utcnow().timestamp()}",
        }
        try:
            res = await self._api_post("/api/v1/chat/new", payload)
            data = res.get("data") or {}
            msg_id = data.get("guid") or data.get("messageGuid") or "ok"
            return SendResult(success=True, message_id=str(msg_id), raw_response=res)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

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
        # Keep a normal assistant response in one iMessage bubble. Paragraph
        # splitting made one answer look like duplicate replies. It remains an
        # explicit opt-in, while over-limit messages are always chunked.
        if self.split_paragraph_replies:
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
            ]
            chunks: List[str] = []
            for para in paragraphs or [text]:
                if len(para) <= self.MAX_MESSAGE_LENGTH:
                    chunks.append(para)
                else:
                    chunks.extend(
                        self.truncate_message(
                            para, max_length=self.MAX_MESSAGE_LENGTH
                        )
                    )
        elif len(text) <= self.MAX_MESSAGE_LENGTH:
            chunks = [text]
        else:
            chunks = self.truncate_message(
                text, max_length=self.MAX_MESSAGE_LENGTH
            )
        last = SendResult(success=True)
        guid: Optional[str] = None
        for index, chunk in enumerate(chunks):
            if not guid:
                guid = await self._resolve_chat_guid(chat_id)
            if not guid:
                # If the target looks like an address, create the chat with the
                # first chunk, then resolve it before sending any remainder.
                if self._private_api_enabled and (
                    "@" in chat_id or re.match(r"^\+\d+", chat_id)
                ):
                    created = await self._create_chat_for_handle(chat_id, chunk)
                    if not created.success or index == len(chunks) - 1:
                        return created
                    last = created
                    raw = created.raw_response if isinstance(created.raw_response, dict) else {}
                    data = raw.get("data") if isinstance(raw, dict) else {}
                    if isinstance(data, dict):
                        guid = data.get("chatGuid") or data.get("chatGUID")
                        chat = data.get("chat")
                        if not guid and isinstance(chat, dict):
                            guid = chat.get("guid") or chat.get("chatGuid")
                    if guid:
                        self._guid_cache[chat_id] = str(guid)
                    else:
                        guid = await self._resolve_chat_guid(chat_id)
                    if not guid:
                        return SendResult(
                            success=False,
                            error=(
                                "BlueBubbles created the chat but could not resolve it "
                                "to deliver the remaining message chunks"
                            ),
                            message_id=created.message_id,
                            raw_response=created.raw_response,
                        )
                    continue
                return SendResult(
                    success=False,
                    error=f"BlueBubbles chat not found for target: {chat_id}",
                )
            payload: Dict[str, Any] = {
                "chatGuid": guid,
                "tempGuid": f"temp-{datetime.utcnow().timestamp()}",
                "message": chunk,
            }
            if reply_to and self._private_api_enabled and self._helper_connected:
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
                return SendResult(success=False, error=str(exc))
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
        if not os.path.isfile(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        guid = await self._resolve_chat_guid(chat_id)
        if not guid:
            return SendResult(success=False, error=f"Chat not found: {chat_id}")

        fname = filename or os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                files = {"attachment": (fname, f, "application/octet-stream")}
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

    def get_typing_refresh_interval(self) -> float:
        """Use BlueBubbles' cadence in the shared base typing lifecycle."""
        return self.typing_refresh_interval

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self.typing_indicators:
            return
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
        except Exception as exc:
            logger.debug("[bluebubbles] send_typing failed: %s", exc)

    async def stop_typing(self, chat_id: str) -> None:
        if not self.typing_indicators:
            return
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.delete(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
        except Exception as exc:
            logger.debug("[bluebubbles] stop_typing failed: %s", exc)

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    async def mark_read(self, chat_id: str) -> bool:
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return False
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/read"), timeout=5
                )
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Tapback reactions and processing UX
    # ------------------------------------------------------------------

    async def _send_reaction(
        self, chat_id: str, message_id: Optional[str], reaction: str
    ) -> bool:
        if (
            not self._private_api_enabled
            or not self._helper_connected
            or not self.client
            or not chat_id
            or not message_id
        ):
            return False
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if not guid:
                return False
            response = await self._api_post(
                "/api/v1/message/react",
                {
                    "chatGuid": guid,
                    "selectedMessageGuid": message_id,
                    "reaction": reaction,
                    "partIndex": 0,
                },
            )
            status = response.get("status")
            if not isinstance(status, int) or not 200 <= status < 300:
                logger.debug(
                    "[bluebubbles] reaction returned invalid or unsuccessful status %r",
                    status,
                )
                return False
            return True
        except Exception as exc:
            logger.debug("[bluebubbles] reaction failed: %s", exc)
            return False

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Acknowledge processing with one native tapback and no text ack."""
        if not self.auto_react:
            return
        await self._send_reaction(
            getattr(event.source, "chat_id", ""),
            getattr(event, "message_id", None),
            self.auto_react_type,
        )

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
                return cache_image_from_bytes(data, ext)

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
                return cache_audio_from_bytes(data, ext)

            # Videos, documents, and everything else
            filename = transfer_name or f"file_{uuid.uuid4().hex[:8]}"
            return cache_document_from_bytes(data, filename)

        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to download attachment %s: %s",
                _redact(att_guid),
                exc,
            )
            return None

    async def _download_attachment_with_retries(
        self, att_guid: str, att_meta: Dict[str, Any]
    ) -> Optional[str]:
        """Bound retries to the provider's single webhook delivery."""
        cached = await self._download_attachment(att_guid, att_meta)
        if cached:
            return cached
        for delay in _ATTACHMENT_RETRY_DELAYS:
            await asyncio.sleep(delay)
            cached = await self._download_attachment(att_guid, att_meta)
            if cached:
                return cached
        return None

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

    def _claim_inbound_message(
        self, message_id: str
    ) -> tuple[Optional[asyncio.Future], bool]:
        """Reserve a GUID, or return the existing in-flight outcome."""
        if self._message_dedup.contains(message_id):
            return None, False
        existing = self._inflight_message_ids.get(message_id)
        if existing is not None:
            return existing, False
        claim = asyncio.get_running_loop().create_future()
        self._inflight_message_ids[message_id] = claim
        return claim, True

    def _finish_inbound_claim(
        self,
        message_id: Optional[str],
        claim: Optional[asyncio.Future],
        *,
        accepted: bool,
    ) -> None:
        """Commit a successful handoff, or release a failed reservation."""
        if not message_id or claim is None:
            return
        if self._inflight_message_ids.get(message_id) is not claim:
            return
        self._inflight_message_ids.pop(message_id, None)
        if accepted:
            self._message_dedup.is_duplicate(message_id)
        if not claim.done():
            claim.set_result(accepted)

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    async def _handle_webhook(self, request):
        from aiohttp import web

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

        # Skip tapback reactions delivered as messages
        assoc_type = record.get("associatedMessageType")
        if isinstance(assoc_type, int) and assoc_type in {
            **_TAPBACK_ADDED,
            **_TAPBACK_REMOVED,
        }:
            return web.Response(text="ok")

        message_id = self._value(
            record.get("guid"),
            record.get("messageGuid"),
            record.get("id"),
        )
        claim: Optional[asyncio.Future] = None
        if message_id:
            claim, is_owner = self._claim_inbound_message(message_id)
            if claim is None:
                logger.info("[bluebubbles] duplicate inbound message ignored")
                return web.Response(text="ok")
            if not is_owner:
                accepted = await asyncio.shield(claim)
                return web.Response(
                    text="ok" if accepted else "handoff unavailable",
                    status=200 if accepted else 503,
                )

        text = (
            self._value(
                record.get("text"), record.get("message"), record.get("body")
            )
            or ""
        )

        # --- Inbound attachment handling ---
        attachments = record.get("attachments") or []
        if not isinstance(attachments, list):
            self._finish_inbound_claim(message_id, claim, accepted=False)
            return web.json_response({"error": "invalid attachments"}, status=400)
        media_urls: List[str] = []
        media_types: List[str] = []
        msg_type = MessageType.TEXT
        attachment_failed = False

        for att in attachments:
            try:
                if not isinstance(att, dict):
                    attachment_failed = True
                    continue
                att_guid = att.get("guid", "")
                if not att_guid:
                    attachment_failed = True
                    continue
                cached = await self._download_attachment_with_retries(att_guid, att)
                if not cached:
                    attachment_failed = True
                    continue
                mime = (att.get("mimeType") or "").lower()
                media_urls.append(cached)
                media_types.append(mime)
                if mime.startswith("image/"):
                    msg_type = MessageType.PHOTO
                elif mime.startswith("audio/") or (att.get("uti") or "").endswith(
                    "caf"
                ):
                    msg_type = MessageType.VOICE
                elif mime.startswith("video/"):
                    msg_type = MessageType.VIDEO
                else:
                    msg_type = MessageType.DOCUMENT
            except asyncio.CancelledError:
                self._finish_inbound_claim(message_id, claim, accepted=False)
                raise
            except Exception:
                attachment_failed = True
                logger.exception(
                    "[bluebubbles] inbound attachment failed; preserving other content"
                )

        if attachment_failed:
            logger.warning(
                "[bluebubbles] one or more inbound attachments remained unavailable "
                "after bounded retries; preserving recoverable message content"
            )

        # With multiple attachments, prefer PHOTO if any images present
        if len(media_urls) > 1:
            mime_prefixes = {(m or "").split("/")[0] for m in media_types}
            if "image" in mime_prefixes:
                msg_type = MessageType.PHOTO

        if not text and media_urls:
            text = "(attachment)"
        if attachments and not text and not media_urls:
            # BlueBubbles will not redeliver this webhook. Preserve the user
            # turn even when every attachment remains unavailable so the agent
            # can acknowledge the failed media instead of silently losing it.
            text = "(attachment unavailable)"
        # --- End attachment handling ---

        chat_guid = self._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        # Fallback: BlueBubbles v1.9+ webhook payloads omit top-level chatGuid;
        # the chat GUID is nested under data.chats[0].guid instead.
        if not chat_guid:
            _chats = record.get("chats") or []
            if _chats and isinstance(_chats[0], dict):
                chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        chat_identifier = self._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            self._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        if not sender or not (chat_guid or chat_identifier) or not text:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            return web.json_response({"error": "missing message fields"}, status=400)

        session_chat_id = chat_guid or chat_identifier
        is_group = bool(record.get("isGroup")) or (";+;" in (chat_guid or ""))
        if is_group and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                logger.debug(
                    "[bluebubbles] ignoring group message (require_mention=true, no mention pattern matched)"
                )
                self._finish_inbound_claim(message_id, claim, accepted=True)
                return web.Response(text="ok")
            text = self._clean_mention_text(text)
        try:
            source = self.build_source(
                chat_id=session_chat_id,
                chat_name=chat_identifier or sender,
                chat_type="group" if is_group else "dm",
                user_id=sender,
                user_name=sender,
                chat_id_alt=chat_identifier,
            )
            event = MessageEvent(
                text=text,
                message_type=msg_type,
                source=source,
                raw_message=payload,
                message_id=message_id,
                reply_to_message_id=self._value(
                    record.get("threadOriginatorGuid"),
                    record.get("associatedMessageGuid"),
                ),
                media_urls=media_urls,
                media_types=media_types,
            )
            # BasePlatformAdapter.handle_message returns after accepting the
            # handoff and spawning agent work; awaiting it lets us release a
            # failed claim without waiting for the agent turn itself.
            await self.handle_message(event)
        except asyncio.CancelledError:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            raise
        except Exception:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            logger.exception("[bluebubbles] failed to hand off inbound message")
            return web.Response(text="handoff unavailable", status=503)

        self._finish_inbound_claim(message_id, claim, accepted=True)

        # Fire-and-forget read receipt
        if self.send_read_receipts and session_chat_id:
            asyncio.create_task(self.mark_read(session_chat_id))

        return web.Response(text="ok")
