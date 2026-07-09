"""
Blooio (iMessage) platform adapter for Hermes Agent.

Blooio is a hosted iMessage API — you send and receive real iMessages (with
SMS/RCS fallback) through a REST API and inbound webhooks, without running a
Mac yourself. This adapter is the twin of the built-in BlueBubbles iMessage
channel and the Photon Spectrum plugin, but talks to Blooio's v2 REST API
(``https://api.blooio.com/v2/api``) instead.

Design
------

**Inbound = webhooks.** The adapter runs a small ``aiohttp`` webhook server
and registers ``/blooio/webhook`` with Blooio. Each event is HMAC-SHA256
signature-verified (Stripe-style ``X-Blooio-Signature: t=<ts>,v1=<hex>``
over ``"{ts}.{rawBody}"``) using the webhook's signing secret, deduped on
``message_id``, and dispatched to the gateway as a normalized
``MessageEvent``. ``message.received`` becomes an inbound message;
``message.reaction`` on one of the bot's own messages becomes a synthetic
``reaction:added:<emoji>`` event (same pattern as Photon/Feishu); delivery
receipts (``message.delivered`` / ``.read`` / ``.failed``) are ignored.

**Outbound = REST.** ``send`` POSTs to ``/chats/{chatId}/messages`` with the
API key as a bearer token. The ``chatId`` is a bare E.164 phone number, an
email address, or a ``grp_…`` group id — the same value inbound events carry
as ``external_id`` (1:1) or ``group_id`` (groups). Blooio accepts a text
array, so a long reply is chunked and sent as one call. Typing indicators,
read receipts, and tapback/emoji reactions each map to a dedicated Blooio
endpoint.

**Media.** Blooio attachments are HTTPS URLs. Sending a remote image URL is a
straight pass-through; sending a local file requires a publicly reachable
``BLOOIO_PUBLIC_URL`` — the file is registered and served from the same
aiohttp app (``/blooio/media/<token>/<name>``) with a traversal guard, exactly
like the LINE plugin.

**Requires a public HTTPS host.** Inbound webhooks and local-file attachment
URLs both need Blooio to reach this process, so a laptop-only Hermes must be
exposed via Cloudflare Tunnel / Tailscale Funnel / ngrok. Set
``BLOOIO_PUBLIC_URL`` to that hostname.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote as _urlquote

logger = logging.getLogger(__name__)

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
)
from gateway.platforms.helpers import strip_markdown
from gateway.config import Platform


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_BASE_URL = "https://api.blooio.com/v2/api"

DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/blooio/webhook"
DEFAULT_MEDIA_PATH_PREFIX = "/blooio/media"

# Webhook events are small JSON payloads; attachments arrive as URLs, not
# inline bytes, so a 1 MiB cap is generous headroom.
WEBHOOK_BODY_MAX_BYTES = 1_048_576

# iMessage has no documented hard cap, but the protocol gets unhappy past
# ~16 KB. Match the BlueBubbles channel's conservative per-bubble limit.
MAX_TEXT_LENGTH = 4000

# Reject webhook events whose signature timestamp is too old/new — bounds a
# replay window. Blooio signs with seconds since epoch.
_SIGNATURE_TOLERANCE_SECONDS = 5 * 60

# Bounded dedup of processed message ids (webhooks are at-least-once).
_DEDUP_MAX_SIZE = 2000

# iMessage classic tapbacks. Blooio accepts these names (prefixed ``+``/``-``)
# or any emoji. We keep the set so a plugin-side reaction request can validate
# a friendly name before falling back to treating the value as an emoji.
_CLASSIC_TAPBACKS = {
    "love", "like", "dislike", "laugh", "emphasize", "question",
}

# Lifecycle-hook tapbacks (opt-in via BLOOIO_REACTIONS). iMessage renders
# these emoji as native reactions.
_REACTION_WORKING = "👀"
_REACTION_SUCCESS = "👍"
_REACTION_FAILURE = "👎"

# Group-chat mention wake words — identical defaults to the BlueBubbles and
# Photon iMessage channels so all three gate group chats the same way.
DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]

# Log redaction — iMessage identifiers are phone numbers / emails (PII).
_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

# Minimum seconds between typing-indicator calls for the same chat.
_TYPING_COOLDOWN_SECONDS = 5.0


def _redact(text: str) -> str:
    """Mask phone numbers and emails in log output."""
    if not text:
        return text
    text = _PHONE_RE.sub("[redacted]", text)
    text = _EMAIL_RE.sub("[redacted]", text)
    return text


def _csv_set(value: str) -> Set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------

def verify_blooio_signature(
    body: bytes,
    signature_header: str,
    signing_secret: str,
    *,
    tolerance_seconds: int = _SIGNATURE_TOLERANCE_SECONDS,
    now: Optional[float] = None,
) -> bool:
    """Verify a Blooio ``X-Blooio-Signature`` header.

    Blooio signs the *raw* request body Stripe-style: the header is
    ``t=<unix_seconds>,v1=<hex>`` where ``v1`` is the hex HMAC-SHA256 of
    ``"{t}.{raw_body}"`` keyed by the webhook's signing secret
    (``whsec_…``). The timestamp is checked against ``tolerance_seconds`` to
    bound replay, and the digest is compared in constant time.
    """
    if not signature_header or not signing_secret or body is None:
        return False

    parsed: Dict[str, str] = {}
    for part in signature_header.split(","):
        key, _, val = part.strip().partition("=")
        if key and val:
            parsed[key.strip()] = val.strip()

    timestamp = parsed.get("t")
    provided = parsed.get("v1")
    if not timestamp or not provided:
        return False

    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False

    current = time.time() if now is None else now
    if tolerance_seconds and abs(current - ts) > tolerance_seconds:
        return False

    try:
        signed_payload = f"{timestamp}.".encode("utf-8") + body
        expected = hmac.new(
            signing_secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
    except Exception:
        return False
    return hmac.compare_digest(expected, provided)


# ---------------------------------------------------------------------------
# Inbound dedup
# ---------------------------------------------------------------------------

class _MessageDeduplicator:
    """Bounded LRU of processed ids to drop at-least-once webhook retries."""

    def __init__(self, max_size: int = _DEDUP_MAX_SIZE) -> None:
        self._seen: Dict[str, float] = {}
        self._max = max_size

    def is_duplicate(self, key: str) -> bool:
        if not key:
            return False
        if key in self._seen:
            return True
        if len(self._seen) >= self._max:
            cutoff = sorted(self._seen.values())[len(self._seen) // 10 or 1]
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
        self._seen[key] = time.time()
        return False


# ---------------------------------------------------------------------------
# Blooio REST client
# ---------------------------------------------------------------------------

class _BlooioClient:
    """Thin async wrapper over the Blooio v2 REST API (httpx)."""

    def __init__(self, api_key: str, base_url: str, *, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self, method: str, path: str, *, json_body: Optional[dict] = None
    ) -> Dict[str, Any]:
        import httpx

        url = f"{self._base}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.request(
                method, url, headers=self._headers, json=json_body
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Blooio {method} {path} → {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json() or {}
        except Exception:
            return {}

    async def get_me(self) -> Dict[str, Any]:
        return await self._request("GET", "/me")

    async def send_message(self, chat_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/chats/{_urlquote(chat_id, safe='')}/messages", json_body=body
        )

    async def set_typing(self, chat_id: str, active: bool) -> None:
        method = "POST" if active else "DELETE"
        await self._request(method, f"/chats/{_urlquote(chat_id, safe='')}/typing")

    async def mark_read(self, chat_id: str) -> None:
        await self._request("POST", f"/chats/{_urlquote(chat_id, safe='')}/read")

    async def react(
        self, chat_id: str, message_id: str, reaction: str,
        *, direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"reaction": reaction}
        if direction:
            body["direction"] = direction
        return await self._request(
            "POST",
            f"/chats/{_urlquote(chat_id, safe='')}/messages/"
            f"{_urlquote(str(message_id), safe='')}/reactions",
            json_body=body,
        )

    async def create_webhook(
        self, webhook_url: str, webhook_type: str = "all"
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/webhooks",
            json_body={"webhook_url": webhook_url, "webhook_type": webhook_type},
        )

    async def download(self, url: str) -> bytes:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                raise RuntimeError(f"Blooio attachment download {resp.status_code}")
            return resp.content


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BlooioAdapter(BasePlatformAdapter):
    """Blooio iMessage gateway adapter (webhook in, REST out)."""

    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_TEXT_LENGTH
    splits_long_messages = True

    _SENT_IDS_MAX = 1000
    _LAST_INBOUND_CHATS_MAX = 200

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("blooio"))
        extra = getattr(config, "extra", {}) or {}

        # Credentials
        self.api_key = os.getenv("BLOOIO_API_KEY") or extra.get("api_key", "")
        self.signing_secret = (
            os.getenv("BLOOIO_WEBHOOK_SECRET") or extra.get("webhook_secret", "")
        )
        self.api_base_url = (
            os.getenv("BLOOIO_API_BASE_URL")
            or extra.get("api_base_url")
            or DEFAULT_API_BASE_URL
        )
        # Optional explicit from-number. When unset we reply from whichever of
        # our numbers received the inbound message (tracked per chat), which is
        # the correct behavior for multi-number pools.
        self.from_number = os.getenv("BLOOIO_FROM_NUMBER") or extra.get("from_number", "")

        # Webhook server
        self.webhook_host = os.getenv("BLOOIO_HOST") or extra.get("host", DEFAULT_WEBHOOK_HOST)
        try:
            self.webhook_port = int(
                os.getenv("BLOOIO_PORT") or extra.get("port", DEFAULT_WEBHOOK_PORT)
            )
        except (TypeError, ValueError):
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self.webhook_path = extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        if not str(self.webhook_path).startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"

        self.public_base_url = (
            os.getenv("BLOOIO_PUBLIC_URL") or extra.get("public_url", "") or ""
        ).rstrip("/")
        self.auto_register_webhook = _truthy(
            os.getenv("BLOOIO_AUTO_REGISTER_WEBHOOK"),
            bool(extra.get("auto_register_webhook", False)),
        )

        # Allowlist gating (phones/emails for DMs, grp_ ids for groups)
        self.allow_all = _truthy(
            os.getenv("BLOOIO_ALLOW_ALL_USERS"), bool(extra.get("allow_all_users", False))
        )
        self.allowed_users = _csv_set(os.getenv("BLOOIO_ALLOWED_USERS", "")) | set(
            extra.get("allowed_users", [])
        )
        self.allowed_groups = _csv_set(os.getenv("BLOOIO_ALLOWED_GROUPS", "")) | set(
            extra.get("allowed_groups", [])
        )

        # Group-mention gating (parity with BlueBubbles / Photon). DMs never gated.
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("BLOOIO_REQUIRE_MENTION")
        self.require_mention = _truthy(_require_mention)
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("BLOOIO_MENTION_PATTERNS")
        )

        # Read receipts + reactions are opt-in.
        self.send_read_receipts = _truthy(
            os.getenv("BLOOIO_SEND_READ_RECEIPTS"),
            bool(extra.get("send_read_receipts", False)),
        )

        # Runtime state
        self._client: Optional[_BlooioClient] = None
        self._app = None
        self._runner = None
        self._site = None
        self._dedup = _MessageDeduplicator()
        self._sent_message_ids: Dict[str, float] = {}
        self._last_inbound_by_chat: Dict[str, str] = {}
        # Which of our numbers received the last inbound per chat → reply-from.
        self._reply_from_by_chat: Dict[str, str] = {}
        self._typing_last_sent: Dict[str, float] = {}

        # Media serving state (local files → public HTTPS for Blooio to fetch)
        self._media_tokens: Dict[str, Tuple[str, float]] = {}
        self._media_temp_paths: Set[str] = set()
        self._media_ttl = 1800

    # -- Group-mention gating -------------------------------------------------

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> "list[re.Pattern]":
        if raw is None:
            patterns: List[Any] = list(DEFAULT_MENTION_PATTERNS)
        elif isinstance(raw, str):
            text = raw.strip()
            try:
                loaded = json.loads(text) if text else []
            except Exception:
                loaded = None
            patterns = loaded if isinstance(loaded, list) else [
                part.strip()
                for line in text.splitlines()
                for part in line.split(",")
            ]
        elif isinstance(raw, list):
            patterns = raw
        else:
            patterns = [raw]

        compiled: "list[re.Pattern]" = []
        for pattern in patterns:
            token = str(pattern).strip()
            if not token:
                continue
            try:
                compiled.append(re.compile(token, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[blooio] invalid mention pattern %r: %s", token, exc)
        return compiled

    def _matches_mention(self, text: str) -> bool:
        if not text or not self._mention_patterns:
            return False
        return any(p.search(text) for p in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        if not text:
            return text
        for pattern in self._mention_patterns:
            match = pattern.match(text.lstrip())
            if match:
                cleaned = text.lstrip()[match.end():].lstrip(" ,:-")
                return cleaned or text
        return text

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.api_key:
            self._set_fatal_error(
                "config_missing",
                "BLOOIO_API_KEY must be set. Create an API key in the Blooio dashboard.",
                retryable=False,
            )
            return False

        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for the Blooio adapter — `pip install aiohttp`",
                retryable=False,
            )
            return False

        self._client = _BlooioClient(self.api_key, self.api_base_url)

        # Spin up the webhook server.
        from aiohttp import web

        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_post(self.webhook_path, self._handle_webhook)
        self._app.router.add_get(f"{self.webhook_path}/health", self._handle_health)
        self._app.router.add_get(
            f"{DEFAULT_MEDIA_PATH_PREFIX}/{{token}}/{{filename}}", self._handle_media
        )

        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind Blooio webhook on {self.webhook_host}:{self.webhook_port}: {exc}",
                retryable=True,
            )
            return False

        if not self.signing_secret:
            logger.warning(
                "[blooio] BLOOIO_WEBHOOK_SECRET not set — inbound webhook "
                "signatures will NOT be verified. Set the signing secret from "
                "your Blooio webhook for production."
            )

        # Optional convenience: register our public webhook URL with Blooio and
        # capture the returned signing secret (shown once) for verification.
        if self.auto_register_webhook and not is_reconnect:
            await self._maybe_register_webhook()

        self._mark_connected()
        logger.info(
            "[blooio] webhook listening on %s:%s%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
            f" (public: {self.public_base_url})" if self.public_base_url else "",
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        self._app = None
        for path in list(self._media_temp_paths):
            try:
                os.unlink(path)
            except OSError:
                pass
        self._media_temp_paths.clear()
        self._media_tokens.clear()

    async def _maybe_register_webhook(self) -> None:
        if not self.public_base_url:
            logger.warning(
                "[blooio] auto-register requested but BLOOIO_PUBLIC_URL is unset — skipping"
            )
            return
        webhook_url = f"{self.public_base_url}{self.webhook_path}"
        try:
            data = await self._client.create_webhook(webhook_url, "all")
        except Exception as exc:
            logger.warning("[blooio] webhook auto-registration failed: %s", exc)
            return
        secret = data.get("signing_secret") or (data.get("webhook") or {}).get(
            "signing_secret"
        )
        if secret:
            self.signing_secret = secret
            logger.info(
                "[blooio] registered webhook %s and captured signing secret",
                webhook_url,
            )
        else:
            logger.info("[blooio] registered webhook %s", webhook_url)

    # -- Webhook handlers -----------------------------------------------------

    async def _handle_health(self, request) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok", "platform": "blooio"})

    async def _handle_webhook(self, request) -> Any:
        from aiohttp import web

        try:
            body = await request.read()
        except Exception as exc:
            logger.debug("[blooio] webhook read failed: %s", exc)
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")

        # Verify signature when a secret is configured.
        if self.signing_secret:
            signature = request.headers.get("X-Blooio-Signature", "")
            if not verify_blooio_signature(body, signature, self.signing_secret):
                return web.Response(status=401, text="invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.Response(status=400, text="bad json")

        # Blooio may deliver one event or a batch under "events".
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            events = [payload]

        for event in events:
            if not isinstance(event, dict):
                continue
            try:
                await self._dispatch_event(event)
            except Exception:
                logger.exception("[blooio] dispatch_event failed")

        return web.Response(status=200, text="ok")

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("event") or ""

        if event_type == "message.received":
            await self._handle_inbound_message(event)
        elif event_type == "message.reaction":
            await self._handle_inbound_reaction(event)
        else:
            # Delivery receipts (sent/delivered/read/failed) and everything
            # else are not agent-actionable.
            logger.debug("[blooio] ignoring event type %r", event_type)

    # ---- inbound message ----------------------------------------------------

    def _resolve_chat(self, event: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return ``(chat_id, chat_type, user_id)`` for an inbound event.

        1:1 chats key on ``external_id`` (the other party's phone/email);
        group chats key on ``group_id`` with ``sender`` as the participant.
        """
        is_group = bool(event.get("is_group"))
        if is_group:
            chat_id = event.get("group_id") or event.get("external_id") or ""
            user_id = event.get("sender") or ""
            return chat_id, "group", user_id
        external = event.get("external_id") or ""
        return external, "dm", external

    def _is_allowed(self, chat_type: str, chat_id: str, user_id: str) -> bool:
        if self.allow_all:
            return True
        if chat_type == "group":
            return bool(chat_id) and chat_id in self.allowed_groups
        return bool(user_id) and user_id in self.allowed_users

    async def _handle_inbound_message(self, event: Dict[str, Any]) -> None:
        message_id = event.get("message_id") or ""
        if message_id and self._dedup.is_duplicate(message_id):
            logger.debug("[blooio] dropping duplicate message %s", message_id)
            return

        chat_id, chat_type, user_id = self._resolve_chat(event)
        if not chat_id:
            logger.warning("[blooio] inbound message missing chat id")
            return

        if not self._is_allowed(chat_type, chat_id, user_id):
            logger.info(
                "[blooio] rejecting unauthorized %s message from %s",
                chat_type, _redact(user_id or chat_id),
            )
            return

        # Remember which of our numbers received this, and the last inbound
        # message id, so replies + agent-facing reactions can default sensibly.
        internal_id = event.get("internal_id")
        if internal_id:
            self._reply_from_by_chat[self._trim(self._reply_from_by_chat, chat_id)] = internal_id
        self._record_last_inbound(chat_id, message_id)

        text = event.get("text") or ""
        media_urls: List[str] = []
        media_types: List[str] = []
        mtype = MessageType.TEXT

        for attachment in event.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            url = attachment.get("url")
            if not url:
                continue
            cached, cached_type = await self._cache_attachment(
                url, attachment.get("name")
            )
            if cached:
                media_urls.append(cached)
                media_types.append(cached_type)
                if mtype == MessageType.TEXT:
                    mtype = _mime_message_type(cached_type)
            else:
                marker = f"[Blooio attachment: {attachment.get('name') or url}]"
                text = f"{text}\n{marker}".strip() if text else marker

        if not text and media_urls:
            text = "(attachment)"

        # Group-mention gating (DMs never gated).
        if chat_type == "group" and self.require_mention:
            if not self._matches_mention(text):
                logger.debug(
                    "[blooio] ignoring group message (require_mention, no match)"
                )
                return
            text = self._clean_mention_text(text)

        # Best-effort read receipt.
        if self.send_read_receipts and self._client:
            asyncio.create_task(self._safe_mark_read(chat_id))

        source = self.build_source(
            chat_id=chat_id,
            chat_name=event.get("group_name") or chat_id,
            chat_type=chat_type,
            user_id=user_id or chat_id,
            user_name=user_id or None,
        )
        await self.handle_message(
            MessageEvent(
                text=text,
                message_type=mtype,
                source=source,
                message_id=message_id,
                raw_message=event,
                media_urls=media_urls,
                media_types=media_types,
                timestamp=_event_timestamp(event),
            )
        )

    async def _handle_inbound_reaction(self, event: Dict[str, Any]) -> None:
        """Route a tapback/emoji reaction — only on messages the bot sent.

        Reactions on human↔human messages aren't addressed to us. A reaction
        that targets one of our own sent messages becomes a synthetic
        ``reaction:added:<value>`` text event (same shape as Photon/Feishu).
        """
        message_id = event.get("message_id") or ""
        action = event.get("action") or "added"
        reaction_id = f"{message_id}:{event.get('reaction')}:{action}:{event.get('timestamp')}"
        if self._dedup.is_duplicate(reaction_id):
            return

        # Only surface reactions on OUR outbound messages.
        if not message_id or message_id not in self._sent_message_ids:
            logger.debug("[blooio] ignoring reaction on a message we didn't send")
            return

        # No allowlist gate here: a reaction on one of OUR messages is
        # implicitly addressed to the bot (the sent-id check above is the
        # real gate), same as the Photon iMessage plugin.
        chat_id, chat_type, user_id = self._resolve_chat(event)
        if not chat_id:
            return

        reaction = event.get("reaction") or ""
        source = self.build_source(
            chat_id=chat_id,
            chat_name=event.get("group_name") or chat_id,
            chat_type=chat_type,
            user_id=user_id or chat_id,
            user_name=user_id or None,
        )
        await self.handle_message(
            MessageEvent(
                text=f"reaction:{action}:{reaction}",
                message_type=MessageType.TEXT,
                source=source,
                message_id=message_id,
                reply_to_message_id=message_id,
                reply_to_text=event.get("original_text") or None,
                reply_to_is_own_message=True,
                raw_message=event,
                timestamp=_event_timestamp(event),
            )
        )

    async def _safe_mark_read(self, chat_id: str) -> None:
        try:
            await self._client.mark_read(chat_id)
        except Exception as exc:
            logger.debug("[blooio] mark_read failed: %s", exc)

    async def _cache_attachment(
        self, url: str, name: Optional[str]
    ) -> Tuple[Optional[str], str]:
        """Download a Blooio attachment URL and cache it as a local path."""
        if not self._client:
            return None, ""
        try:
            raw = await self._client.download(url)
        except Exception as exc:
            logger.warning("[blooio] attachment download failed: %s", exc)
            return None, ""
        mime, _ = mimetypes.guess_type(name or url)
        mime = (mime or "").lower()
        suffix = Path(name).suffix if name else ""
        try:
            if mime.startswith("image/"):
                return cache_image_from_bytes(raw, suffix or ".jpg"), mime
            if mime.startswith("audio/"):
                return cache_audio_from_bytes(raw, suffix or ".m4a"), mime
            if mime.startswith("video/"):
                return cache_video_from_bytes(raw, suffix or ".mp4"), mime
            return cache_document_from_bytes(raw, name or "attachment"), (
                mime or "application/octet-stream"
            )
        except Exception as exc:
            logger.warning("[blooio] failed to cache attachment: %s", exc)
            return None, ""

    # -- Sent-id + last-inbound tracking --------------------------------------

    def _record_sent_message(self, message_id: Optional[str]) -> None:
        if not message_id:
            return
        sent = self._sent_message_ids
        sent.pop(message_id, None)
        sent[message_id] = time.time()
        if len(sent) > self._SENT_IDS_MAX:
            for old in list(sent.keys())[: len(sent) - self._SENT_IDS_MAX]:
                del sent[old]

    def _record_last_inbound(self, chat_id: str, message_id: str) -> None:
        if not chat_id or not message_id:
            return
        last = self._last_inbound_by_chat
        last.pop(chat_id, None)
        last[chat_id] = message_id
        if len(last) > self._LAST_INBOUND_CHATS_MAX:
            for old in list(last.keys())[: len(last) - self._LAST_INBOUND_CHATS_MAX]:
                del last[old]

    @staticmethod
    def _trim(mapping: Dict[str, str], key: str, max_size: int = 200) -> str:
        """Bounded insertion for the reply-from map; returns the key."""
        mapping.pop(key, None)
        if len(mapping) >= max_size:
            for old in list(mapping.keys())[: len(mapping) - max_size + 1]:
                del mapping[old]
        return key

    # -- Outbound: text -------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Blooio adapter not connected")

        text = self.format_message(content)
        chunks = self.truncate_message(text, max_length=self.MAX_MESSAGE_LENGTH)
        if not chunks:
            return SendResult(success=True, message_id=None)

        body: Dict[str, Any] = {
            "text": chunks if len(chunks) > 1 else chunks[0],
        }
        from_number = self._reply_from_by_chat.get(chat_id) or self.from_number
        if from_number:
            body["from_number"] = from_number
        if reply_to:
            body["reply_to"] = {"message_id": reply_to}

        try:
            data = await self._client.send_message(chat_id, body)
        except Exception as exc:
            logger.error("[blooio] send failed: %s", _redact(str(exc)))
            return SendResult(success=False, error=str(exc))

        message_id = data.get("message_id")
        message_ids = data.get("message_ids") or []
        for mid in message_ids:
            self._record_sent_message(mid)
        if message_id:
            self._record_sent_message(message_id)
        elif message_ids:
            message_id = message_ids[-1]
        return SendResult(success=True, message_id=message_id, raw_response=data)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        now = time.time()
        if now - self._typing_last_sent.get(chat_id, 0.0) < _TYPING_COOLDOWN_SECONDS:
            return
        self._typing_last_sent[chat_id] = now
        if self._client:
            try:
                await self._client.set_typing(chat_id, True)
            except Exception as exc:
                logger.debug("[blooio] send_typing failed: %s", exc)

    async def stop_typing(self, chat_id: str) -> None:
        self._typing_last_sent.pop(chat_id, None)
        if self._client:
            try:
                await self._client.set_typing(chat_id, False)
            except Exception as exc:
                logger.debug("[blooio] stop_typing failed: %s", exc)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "group" if str(chat_id).startswith("grp_") else "dm"
        return {"name": chat_id, "type": chat_type, "id": chat_id}

    def format_message(self, content: str) -> str:
        # iMessage renders plain text — strip Markdown the client can't show.
        return strip_markdown(content)

    # -- Outbound: media ------------------------------------------------------

    async def _send_attachment_urls(
        self,
        chat_id: str,
        urls: List[str],
        caption: Optional[str] = None,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Blooio adapter not connected")
        body: Dict[str, Any] = {"attachments": urls}
        if caption:
            body["text"] = self.format_message(caption)
        from_number = self._reply_from_by_chat.get(chat_id) or self.from_number
        if from_number:
            body["from_number"] = from_number
        try:
            data = await self._client.send_message(chat_id, body)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        message_id = data.get("message_id") or (data.get("message_ids") or [None])[-1]
        self._record_sent_message(message_id)
        return SendResult(success=True, message_id=message_id, raw_response=data)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # Blooio ingests attachment URLs directly — pass a remote URL through.
        if _is_http_url(image_url):
            return await self._send_attachment_urls(chat_id, [image_url], caption)
        return await self.send_image_file(chat_id, image_url, caption)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        if _is_http_url(video_path):
            return await self._send_attachment_urls(chat_id, [video_path], caption)
        return await self._send_local_file(chat_id, video_path, caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        if _is_http_url(audio_path):
            return await self._send_attachment_urls(chat_id, [audio_path], caption)
        return await self._send_local_file(chat_id, audio_path, caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        if _is_http_url(file_path):
            return await self._send_attachment_urls(chat_id, [file_path], caption)
        return await self._send_local_file(chat_id, file_path, caption, name=file_name)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # iMessage renders GIFs inline as ordinary image attachments.
        return await self.send_image(chat_id, animation_url, caption, reply_to, metadata)

    async def _send_local_file(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        *,
        name: Optional[str] = None,
    ) -> SendResult:
        """Serve a local file over the public URL, then send it as an attachment."""
        safe_path = self.validate_media_delivery_path(str(path))
        if not safe_path:
            return SendResult(success=False, error=f"unsafe or missing path: {path}")
        if not self.public_base_url:
            return SendResult(
                success=False,
                error="BLOOIO_PUBLIC_URL must be set to send local files "
                "(Blooio fetches attachments from a public HTTPS URL)",
            )
        token = self._register_media(safe_path)
        url = self._media_url(token, name or Path(safe_path).name)
        if not url.lower().startswith("https://"):
            return SendResult(
                success=False, error=f"Blooio attachment URL must be HTTPS: {url}"
            )
        return await self._send_attachment_urls(chat_id, [url], caption)

    def _register_media(self, file_path: str) -> str:
        now = time.time()
        for token in list(self._media_tokens.keys()):
            _, exp = self._media_tokens[token]
            if now > exp:
                self._media_tokens.pop(token, None)
        resolved = str(Path(file_path).resolve())
        token = secrets.token_urlsafe(32)
        self._media_tokens[token] = (resolved, now + self._media_ttl)
        return token

    def _media_url(self, token: str, filename: str) -> str:
        base = self.public_base_url
        safe_name = _urlquote(filename, safe="")
        return f"{base}{DEFAULT_MEDIA_PATH_PREFIX}/{token}/{safe_name}"

    async def _handle_media(self, request) -> Any:
        from aiohttp import web

        token = request.match_info["token"]
        entry = self._media_tokens.get(token)
        if not entry:
            return web.Response(status=404, text="not found")
        file_path, expires_at = entry
        if time.time() > expires_at:
            self._media_tokens.pop(token, None)
            return web.Response(status=410, text="gone")
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return web.Response(status=404, text="not found")

        try:
            from hermes_constants import get_hermes_home

            hermes_home = Path(get_hermes_home()).resolve()
        except Exception:
            hermes_home = Path.home().joinpath(".hermes").resolve()
        allowed_roots = {
            Path(tempfile.gettempdir()).resolve(),
            Path("/tmp").resolve(),
            hermes_home,
        }
        resolved = path.resolve()
        if not any(_is_relative_to(resolved, r) for r in allowed_roots):
            logger.warning("[blooio] refusing to serve outside allowed roots")
            return web.Response(status=403, text="forbidden")
        content_type, _ = mimetypes.guess_type(str(path))
        return web.FileResponse(
            path, headers={"Content-Type": content_type or "application/octet-stream"}
        )

    # -- Reactions ------------------------------------------------------------

    def _reactions_enabled(self) -> bool:
        return _truthy(os.getenv("BLOOIO_REACTIONS"), False)

    @staticmethod
    def _normalize_reaction(value: str) -> str:
        """Return a Blooio reaction token (classic tapback name or emoji).

        Values already prefixed with ``+``/``-`` are respected. Bare classic
        tapback names and bare emoji default to ``+`` (add).
        """
        value = (value or "").strip()
        if value[:1] in {"+", "-"}:
            return value
        return f"+{value}"

    async def _react(
        self, chat_id: str, message_id: str, reaction: str,
        *, direction: Optional[str] = None,
    ) -> bool:
        if not self._client:
            return False
        try:
            await self._client.react(
                chat_id, message_id, self._normalize_reaction(reaction),
                direction=direction,
            )
            return True
        except Exception as exc:
            logger.debug("[blooio] react failed: %s", exc)
            return False

    async def _unreact(self, chat_id: str, message_id: str, reaction: str) -> bool:
        if not self._client:
            return False
        value = (reaction or "").strip().lstrip("+")
        try:
            await self._client.react(chat_id, message_id, f"-{value}")
            return True
        except Exception as exc:
            logger.debug("[blooio] unreact failed: %s", exc)
            return False

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Agent-facing tapback/emoji reaction on a message in ``chat_id``.

        Without ``message_id`` we target the chat's most recent inbound
        message, falling back to Blooio's relative index ``-1`` (the last
        message in the chat). ``emoji`` may be a classic tapback name
        (``love``, ``like``, …) or any emoji, with an optional ``+``/``-``.
        """
        target = message_id or self._last_inbound_by_chat.get(chat_id)
        direction = None
        if not target:
            target = "-1"  # Blooio relative index → last message in the chat
            direction = "inbound"
        ok = await self._react(chat_id, target, emoji, direction=direction)
        if not ok:
            return {"success": False, "error": "reaction failed (see debug log)"}
        return {"success": True, "message_id": target}

    async def remove_reaction(
        self, chat_id: str, emoji: str = "", message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        target = message_id or self._last_inbound_by_chat.get(chat_id) or "-1"
        ok = await self._unreact(chat_id, target, emoji or _REACTION_WORKING)
        if not ok:
            return {"success": False, "error": "unreact failed (see debug log)"}
        return {"success": True, "message_id": target}

    async def on_processing_start(self, event: MessageEvent) -> None:
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._react(chat_id, message_id, _REACTION_WORKING)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not chat_id or not message_id:
            return
        await self._unreact(chat_id, message_id, _REACTION_WORKING)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._react(chat_id, message_id, _REACTION_SUCCESS)
        elif outcome == ProcessingOutcome.FAILURE:
            await self._react(chat_id, message_id, _REACTION_FAILURE)
        # CANCELLED: leave unreacted.


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _is_http_url(value: str) -> bool:
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def _mime_message_type(mime: str) -> MessageType:
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    return MessageType.DOCUMENT


def _event_timestamp(event: Dict[str, Any]):
    from datetime import datetime, timezone

    raw = event.get("timestamp")
    if isinstance(raw, (int, float)):
        try:
            # Blooio timestamps are milliseconds since epoch.
            return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    return datetime.now(tz=timezone.utc)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except (AttributeError, ValueError):
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require an API key plus aiohttp + httpx at runtime."""
    if not os.getenv("BLOOIO_API_KEY"):
        return False
    try:
        import aiohttp  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("BLOOIO_API_KEY") or extra.get("api_key"))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Auto-seed PlatformConfig.extra from an env-only Blooio setup."""
    if not os.getenv("BLOOIO_API_KEY"):
        return None
    seeded: Dict[str, Any] = {}
    if os.getenv("BLOOIO_PORT"):
        try:
            seeded["port"] = int(os.environ["BLOOIO_PORT"])
        except ValueError:
            pass
    if os.getenv("BLOOIO_HOST"):
        seeded["host"] = os.environ["BLOOIO_HOST"]
    if os.getenv("BLOOIO_PUBLIC_URL"):
        seeded["public_url"] = os.environ["BLOOIO_PUBLIC_URL"]
    if os.getenv("BLOOIO_HOME_CHANNEL"):
        seeded["home_channel"] = {
            "chat_id": os.environ["BLOOIO_HOME_CHANNEL"],
            "name": os.getenv("BLOOIO_HOME_CHANNEL_NAME", "Home"),
        }
    return seeded or {}


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process send for cron jobs detached from the gateway.

    Sends text via the Blooio REST API using the API key. Remote media URLs
    are forwarded as attachments; local files can't be served without the
    gateway's webhook server bound, so we note them as a text hint instead.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    api_key = os.getenv("BLOOIO_API_KEY") or extra.get("api_key", "")
    if not api_key or not chat_id:
        return {"error": "Blooio standalone send: missing API key or chat_id"}

    base_url = (
        os.getenv("BLOOIO_API_BASE_URL")
        or extra.get("api_base_url")
        or DEFAULT_API_BASE_URL
    )
    client = _BlooioClient(api_key, base_url)

    body: Dict[str, Any] = {}
    text = strip_markdown(message or "")
    remote_urls: List[str] = []
    local_hint = 0
    for item in media_files or []:
        path = item[0] if isinstance(item, (list, tuple)) else item
        if _is_http_url(str(path)):
            remote_urls.append(str(path))
        else:
            local_hint += 1
    if local_hint:
        hint = f"[{local_hint} attachment(s) generated; not deliverable from cron]"
        text = f"{text}\n{hint}".strip() if text else hint
    if text:
        body["text"] = text[:MAX_TEXT_LENGTH]
    if remote_urls:
        body["attachments"] = remote_urls
    from_number = os.getenv("BLOOIO_FROM_NUMBER") or extra.get("from_number")
    if from_number:
        body["from_number"] = from_number
    if not body:
        return {"error": "Blooio standalone send: nothing to send"}

    try:
        data = await client.send_message(chat_id, body)
    except Exception as exc:
        return {"error": str(exc)}
    message_id = data.get("message_id") or (data.get("message_ids") or [None])[-1]
    return {"success": True, "message_id": message_id}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup blooio``."""
    print()
    print("Blooio (iMessage) setup")
    print("-----------------------")
    print("Create an API key in the Blooio dashboard: https://app.blooio.com/")
    print("Blooio delivers inbound messages via webhooks, so Hermes must be")
    print("reachable at a public HTTPS URL (Cloudflare Tunnel, ngrok, etc.).")
    print()

    try:
        from hermes_cli.config import get_env_var, set_env_var
    except ImportError:
        print("hermes_cli.config not available; set BLOOIO_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_var(var) if callable(get_env_var) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt

                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            set_env_var(var, value)

    _prompt("BLOOIO_API_KEY", "Blooio API key", secret=True)
    _prompt("BLOOIO_WEBHOOK_SECRET", "Webhook signing secret (whsec_…)", secret=True)
    _prompt("BLOOIO_PUBLIC_URL", "Public HTTPS base URL (e.g. https://my-tunnel.example.com)")
    _prompt("BLOOIO_ALLOWED_USERS", "Allowed sender phone numbers/emails (comma-separated; blank=skip)")
    print(
        "Done. In the Blooio dashboard, add a webhook pointing to "
        "<your-public-url>/blooio/webhook (type: all) and copy its signing "
        "secret into BLOOIO_WEBHOOK_SECRET."
    )


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="blooio",
        label="iMessage via Blooio",
        adapter_factory=lambda cfg: BlooioAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BLOOIO_API_KEY"],
        install_hint=(
            "Set BLOOIO_API_KEY (and BLOOIO_WEBHOOK_SECRET). Blooio needs a "
            "public HTTPS URL — expose Hermes via Cloudflare Tunnel/ngrok and "
            "set BLOOIO_PUBLIC_URL. Run: hermes setup blooio"
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="BLOOIO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="BLOOIO_ALLOWED_USERS",
        allow_all_env="BLOOIO_ALLOW_ALL_USERS",
        max_message_length=MAX_TEXT_LENGTH,
        emoji="💬",
        # iMessage identifiers are E.164 phone numbers / emails → PII-sensitive,
        # matching the BlueBubbles and Photon iMessage channels.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating over iMessage via Blooio. Treat replies "
            "like normal text messages — short, friendly, and conversational. "
            "iMessage does NOT render Markdown, so avoid ** or # markup; bare "
            "URLs are fine and get a rich preview. Recipient identifiers are "
            "E.164 phone numbers or emails — never expose them unless asked. "
            "You can react to messages with tapbacks (love, like, dislike, "
            "laugh, emphasize, question) or emoji."
        ),
    )
