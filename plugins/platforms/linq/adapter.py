"""
Linq iMessage platform adapter for Hermes Agent.

Linq (https://linqapp.com) delivers real iMessage "blue bubbles" over a hosted
partner API — no Mac and no BlueBubbles server required.  This adapter plugs
Linq into the Hermes gateway as a first-class messaging platform, alongside
the bundled BlueBubbles and Photon iMessage channels.

Inbound:
    Linq Blue v3 POSTs signed JSON webhooks to a URL you register in the Linq
    dashboard.  The adapter runs an aiohttp server on ``LINQ_WEBHOOK_PORT``,
    verifies the ``X-Webhook-Signature`` HMAC (``hex(hmac_sha256(secret,
    "{ts}.{body}"))``), rejects deliveries with a timestamp drift > 5 minutes,
    dedupes on ``message.id``, and dispatches a normalized ``MessageEvent`` to
    the gateway runner via ``BasePlatformAdapter.handle_message``.

Outbound:
    Linq exposes a public REST API, so — unlike Photon — there is no Node
    sidecar.  Every ``send`` / ``send_image`` / ``send_typing`` call is a
    direct ``httpx`` request to ``https://api.linqapp.com/api/partner/v3``.

Configuration precedence is env var → ``config.yaml`` (``platforms.linq.extra``)
→ ``~/.hermes/auth.json``, matching every other Hermes platform.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx ships with Hermes
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in Hermes core (hermes-agent[messaging])
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from . import auth as linq_auth
from . import signing
from .linq_api import DEFAULT_API_BASE, LinqClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants

_DEFAULT_WEBHOOK_PORT = 8790
_DEFAULT_WEBHOOK_PATH = "/linq/webhook"
_DEFAULT_WEBHOOK_BIND = "0.0.0.0"

# iMessage practical message size is ~16 KB; keep a conservative cap that
# matches the BlueBubbles and Photon iMessage channels.
_MAX_MESSAGE_LENGTH = 8000

# Dedup parameters — Linq retries deliveries, so we WILL see a message.id more
# than once.  Keep a generous buffer pruned by age.
_DEDUP_MAX_SIZE = 4000
_DEDUP_WINDOW_SECONDS = 48 * 3600

# Inbound media MIME prefixes we download to a local path for vision tooling.
_IMAGE_PREFIXES = ("image/",)


# ---------------------------------------------------------------------------
# Module-level helpers (also used by check_fn / validate_config / standalone)

def check_requirements() -> bool:
    """Return True when the adapter's Python deps are importable."""
    return HTTPX_AVAILABLE and AIOHTTP_AVAILABLE


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    token = (
        os.getenv(linq_auth.ENV_TOKEN)
        or extra.get("api_token")
        or linq_auth.load_token()
    )
    return bool(token)


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env so env-only setups appear in status."""
    token = linq_auth.load_token()
    if not token:
        return None
    seeded: Dict[str, Any] = {
        "api_token": token,
        "webhook_port": signing.coerce_port(os.getenv("LINQ_WEBHOOK_PORT"), _DEFAULT_WEBHOOK_PORT),
        "webhook_path": os.getenv("LINQ_WEBHOOK_PATH") or _DEFAULT_WEBHOOK_PATH,
    }
    phone = linq_auth.load_from_phone()
    if phone:
        seeded["from_phone"] = phone
    return seeded


# ---------------------------------------------------------------------------
# Adapter

class LinqAdapter(BasePlatformAdapter):
    """Inbound: signed webhook on aiohttp. Outbound: direct Linq REST API."""

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("linq"))
        extra = config.extra or {}

        # Credentials (env wins, then config.extra, then auth.json).
        self._token: str = (
            os.getenv(linq_auth.ENV_TOKEN)
            or extra.get("api_token")
            or linq_auth.load_token()
            or ""
        )
        self._api_base: str = (
            os.getenv("LINQ_API_BASE")
            or extra.get("api_base")
            or DEFAULT_API_BASE
        )
        # Optional: only handle inbound addressed to this Linq number (lets a
        # multi-number account run one adapter per number).
        self._from_phone: Optional[str] = (
            os.getenv(linq_auth.ENV_FROM_PHONE)
            or extra.get("from_phone")
            or linq_auth.load_from_phone()
            or None
        )

        # Webhook receiver.
        self._webhook_port = signing.coerce_port(
            extra.get("webhook_port") or os.getenv("LINQ_WEBHOOK_PORT"),
            _DEFAULT_WEBHOOK_PORT,
        )
        self._webhook_path = (
            extra.get("webhook_path")
            or os.getenv("LINQ_WEBHOOK_PATH")
            or _DEFAULT_WEBHOOK_PATH
        )
        self._webhook_bind = (
            extra.get("webhook_bind")
            or os.getenv("LINQ_WEBHOOK_BIND")
            or _DEFAULT_WEBHOOK_BIND
        )
        self._webhook_secret: str = (
            os.getenv("LINQ_WEBHOOK_SECRET")
            or extra.get("webhook_secret")
            or ""
        )

        # Send read receipts + typing indicators on inbound (default on, as
        # with the BlueBubbles channel's send_read_receipts).
        _srr = extra.get("send_read_receipts")
        if _srr is None:
            _srr = os.getenv("LINQ_SEND_READ_RECEIPTS")
        self._send_read_receipts = signing.coerce_bool(_srr, default=True)

        # Group-chat mention gating (parity with Photon / BlueBubbles).
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("LINQ_REQUIRE_MENTION")
        self.require_mention = signing.coerce_bool(_require_mention, default=False)
        self._mention_patterns = signing.compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("LINQ_MENTION_PATTERNS")
        )

        # Runtime state.
        self._runner: "Optional[web.AppRunner]" = None
        self._client: Optional[LinqClient] = None
        self._seen_messages: Dict[str, float] = {}

    # -- Connection lifecycle ---------------------------------------------

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP", "aiohttp not installed. Run: pip install aiohttp", retryable=False
            )
            return False
        if not HTTPX_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP", "httpx not installed. Run: pip install httpx", retryable=False
            )
            return False
        if not self._token:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "LINQ_API_TOKEN is required. Run: hermes linq setup",
                retryable=False,
            )
            return False

        try:
            await self._start_webhook_server()
        except OSError as exc:
            self._set_fatal_error(
                "PORT_IN_USE",
                f"webhook port {self._webhook_port} unavailable: {exc}",
                retryable=True,
            )
            return False

        self._client = LinqClient(self._token, api_base=self._api_base)

        if not self._webhook_secret:
            logger.warning(
                "[linq] LINQ_WEBHOOK_SECRET unset — accepting unsigned webhook "
                "deliveries. Set the signing secret from your Linq dashboard to "
                "enable HMAC verification."
            )
        self._mark_connected()
        logger.info(
            "[linq] connected — webhook at %s:%d%s (outbound via %s)",
            self._webhook_bind, self._webhook_port, self._webhook_path, self._api_base,
        )
        return True

    async def disconnect(self) -> None:
        await self._stop_webhook_server()
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._mark_disconnected()

    # -- Webhook server ----------------------------------------------------

    async def _start_webhook_server(self) -> None:
        app = web.Application()
        app.router.add_post(self._webhook_path, self._handle_webhook)
        app.router.add_get("/healthz", lambda _req: web.Response(text="ok"))
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_bind, self._webhook_port)
        await site.start()

    async def _stop_webhook_server(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        if self._webhook_secret:
            ts = request.headers.get("X-Webhook-Timestamp", "")
            sig = request.headers.get("X-Webhook-Signature", "")
            if not signing.verify_signature(
                body=body,
                timestamp_header=ts,
                signature_header=sig,
                signing_secret=self._webhook_secret,
            ):
                logger.warning("[linq] rejected webhook with bad signature")
                return web.Response(status=401, text="invalid signature")

        import json

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")

        event_type = payload.get("event_type") or payload.get("event")
        if event_type != "message.received":
            # Acknowledge reactions / delivery receipts / future event types
            # with 200 so Linq doesn't retry them.  Log the interesting ones.
            if event_type == "reaction.received":
                data = payload.get("data") or {}
                if not data.get("is_from_me"):
                    reaction = data.get("reaction") or {}
                    logger.debug(
                        "[linq] reaction %s %s from=%s msg=%s",
                        reaction.get("operation"), reaction.get("type"),
                        _redact(data.get("from")), data.get("message_id"),
                    )
            elif event_type == "message.delivery_status":
                data = payload.get("data") or {}
                logger.debug(
                    "[linq] delivery %s msg=%s", data.get("status"), data.get("message_id")
                )
            return web.Response(text="ok")

        data = payload.get("data") or {}
        message = data.get("message") or {}
        msg_id = message.get("id")
        if not msg_id:
            return web.Response(status=400, text="missing message.id")
        if self._is_duplicate(msg_id):
            return web.Response(text="ok (dup)")

        try:
            await self._dispatch_inbound(data)
        except Exception:
            logger.exception("[linq] inbound dispatch failed")
            # 200 anyway — we own the dedup; failing here would make Linq retry
            # the same id we've already recorded.
        return web.Response(text="ok")

    def _is_duplicate(self, msg_id: str) -> bool:
        import time

        now = time.time()
        if len(self._seen_messages) > _DEDUP_MAX_SIZE:
            cutoff = now - _DEDUP_WINDOW_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    async def _dispatch_inbound(self, data: Dict[str, Any]) -> None:
        sender = (data.get("from") or "").strip()
        if not sender:
            logger.warning("[linq] inbound missing sender")
            return
        if data.get("is_from_me"):
            return  # never reply to our own echoes

        # Multi-number filter: ignore traffic addressed to a different Linq
        # number when this adapter is pinned to one.
        if self._from_phone and data.get("recipient_phone") not in (None, self._from_phone):
            logger.debug("[linq] skipping message to %s (not %s)", _redact(data.get("recipient_phone")), _redact(self._from_phone))
            return

        chat_id = str(data.get("chat_id") or "")
        if not chat_id:
            logger.warning("[linq] inbound missing chat_id")
            return

        message = data.get("message") or {}
        parts = message.get("parts") or []
        text = signing.extract_text(parts)
        media = signing.extract_media(parts)

        if not text.strip() and not media:
            return

        chat_type = "group" if signing.is_group_chat(data) else "dm"

        # Group-mention gating (parity with Photon / BlueBubbles). DMs are
        # never gated; group messages are dropped unless they hit a wake word,
        # and the leading wake word is stripped from the ones that do.
        if chat_type == "group" and self.require_mention:
            if not signing.message_matches_mention(text, self._mention_patterns):
                logger.debug("[linq] ignoring group message (require_mention, no match)")
                return
            text = signing.clean_mention_text(text, self._mention_patterns)

        # Read receipt + typing indicator (best-effort, fire-and-forget).
        if self._send_read_receipts and self._client is not None:
            asyncio.create_task(self._client.mark_read(chat_id))
            asyncio.create_task(self._client.start_typing(chat_id))

        # Download image attachments to a local path so the vision tools can
        # read them.  Non-image media is noted inline (the agent at least
        # knows something was sent and where it lives).
        media_urls: List[str] = []
        media_types: List[str] = []
        mtype = MessageType.TEXT
        if media:
            mtype, extra_note = await self._ingest_media(media, media_urls, media_types)
            if extra_note:
                text = f"{text}\n{extra_note}" if text else extra_note

        body_text = text.strip()
        if not body_text and not media_urls:
            return

        # Timestamp.
        ts_str = data.get("received_at") or ""
        try:
            timestamp = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        reply_to = (message.get("reply_to") or {}).get("message_id")

        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender,
            message_id=str(message.get("id")) if message.get("id") else None,
        )
        event = MessageEvent(
            text=body_text,
            message_type=mtype,
            source=source,
            message_id=message.get("id"),
            raw_message=data,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to,
        )
        await self.handle_message(event)

    async def _ingest_media(
        self,
        media: List[Dict[str, Any]],
        media_urls: List[str],
        media_types: List[str],
    ) -> "tuple[MessageType, str]":
        """Download image attachments locally; note the rest inline.

        Returns ``(message_type, extra_text_note)``.
        """
        notes: List[str] = []
        mtype = MessageType.TEXT
        for item in media:
            url = item.get("url")
            mime = (item.get("mime_type") or "").lower()
            if not url:
                continue
            if any(mime.startswith(p) for p in _IMAGE_PREFIXES):
                try:
                    from gateway.platforms.base import cache_image_from_url

                    local_path = await cache_image_from_url(url)
                    media_urls.append(local_path)
                    media_types.append(mime or "image/jpeg")
                    mtype = MessageType.PHOTO
                except Exception as exc:
                    logger.debug("[linq] failed to cache inbound image: %s", exc)
                    notes.append(f"[image attachment: {url}]")
            else:
                label = item.get("filename") or mime or "attachment"
                notes.append(f"[attachment received: {label} — {url}]")
        return mtype, "\n".join(notes)

    # -- Outbound ----------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="Linq adapter not connected")
        text = content or ""
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[linq] truncating outbound from %d to %d chars", len(text), self.MAX_MESSAGE_LENGTH
            )
            text = text[: self.MAX_MESSAGE_LENGTH]
        try:
            result = await self._client.send_message(chat_id, text=text, reply_to_message_id=reply_to)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=_is_retryable(exc))
        return SendResult(success=True, message_id=result.get("message_id"))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # Linq accepts a public media URL directly — no local caching or
        # sidecar needed (the key simplification over Photon's outbound path).
        if self._client is None:
            return SendResult(success=False, error="Linq adapter not connected")
        try:
            result = await self._client.send_message(
                chat_id, text=caption or None, media_url=image_url, reply_to_message_id=reply_to
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=_is_retryable(exc))
        return SendResult(success=True, message_id=result.get("message_id"))

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

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        if self._client is None:
            return
        try:
            await self._client.start_typing(chat_id)
        except Exception as exc:
            logger.debug("[linq] send_typing failed: %s", exc)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Linq chat ids are opaque tokens; surface what we have.

        Linq's inbound payloads identify peers by phone number but address
        chats by ``chat_id``, so we can't cheaply resolve a display name here.
        """
        chat_type = "group" if signing.is_group_chat({"chat_id": chat_id}) else "dm"
        return {"name": chat_id, "type": chat_type, "id": chat_id}


# ---------------------------------------------------------------------------
# Helpers

def _redact(value: Any) -> str:
    """Mask all but the last 2 digits of a phone-like identifier for logs."""
    s = str(value or "")
    if len(s) <= 2:
        return "***"
    return "***" + s[-2:]


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("connecterror", "connectionerror", "connecttimeout"))


# ---------------------------------------------------------------------------
# Standalone (out-of-process) send for cron deliveries when the gateway is not
# co-resident.  Opens an ephemeral Linq client, sends, and closes.

async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,  # noqa: ARG001 - Linq has no threads
    media_files: Optional[list] = None,
    force_document: bool = False,  # noqa: ARG001 - iMessage auto-detects file kind
) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not installed"}
    extra = (pconfig.extra or {})
    token = os.getenv(linq_auth.ENV_TOKEN) or extra.get("api_token") or linq_auth.load_token()
    if not token:
        return {"error": "Linq standalone send requires LINQ_API_TOKEN"}
    api_base = os.getenv("LINQ_API_BASE") or extra.get("api_base") or DEFAULT_API_BASE

    last_id: Optional[str] = None
    try:
        async with LinqClient(token, api_base=api_base) as client:
            if message:
                result = await client.send_message(chat_id, text=message[:_MAX_MESSAGE_LENGTH])
                last_id = result.get("message_id")
            # Local media files can't be forwarded: Linq sends media by public
            # URL, not multipart upload.  Note the omission rather than failing
            # the whole delivery.
            if media_files:
                logger.warning(
                    "[linq] standalone send skipped %d local media file(s) — "
                    "Linq sends media by URL, not file upload",
                    len(media_files),
                )
        return {"success": True, "message_id": last_id}
    except Exception as exc:
        return {"error": f"Linq standalone send failed: {exc}"}


# ---------------------------------------------------------------------------
# Plugin entry point

def register(ctx) -> None:
    """Called by the Hermes plugin loader at startup."""
    from . import cli as _cli

    ctx.register_platform(
        name="linq",
        label="Linq iMessage",
        adapter_factory=lambda cfg: LinqAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["LINQ_API_TOKEN"],
        install_hint=(
            "Run: hermes linq setup  (stores your Linq API token + from-phone, "
            "then register a webhook URL in your Linq dashboard)."
        ),
        setup_fn=_cli.gateway_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="LINQ_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="LINQ_ALLOWED_USERS",
        allow_all_env="LINQ_ALLOW_ALL_USERS",
        max_message_length=_MAX_MESSAGE_LENGTH,
        emoji="💬",
        # iMessage carries E.164 phone numbers — treat session descriptions as
        # PII-sensitive so they get redacted before reaching the LLM (matches
        # the BlueBubbles and Photon iMessage channels).
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via Linq (iMessage). Treat replies like "
            "regular text messages — short, friendly, no markdown rendering. "
            "Recipient identifiers are E.164 phone numbers; never expose them "
            "in responses unless the user asked. Inbound images are downloaded "
            "locally and readable with your vision tools."
        ),
    )

    # Register CLI subcommands — `hermes linq ...`
    ctx.register_cli_command(
        name="linq",
        help="Set up and manage the Linq iMessage integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )
