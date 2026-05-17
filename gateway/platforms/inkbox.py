"""Inkbox platform adapter.

Inkbox (https://inkbox.ai) is API-first communication infrastructure that
gives an AI agent a stable email address, phone number, and persistent
contact list scoped to one *agent identity*. This adapter routes the
three Inkbox modalities — inbound email, inbound SMS, and live voice
calls — into a single contact-keyed Hermes session per remote party.

Architecture
------------
On ``connect()`` the adapter:
  1. Brings up the Inkbox edge-mode tunnel attached to the configured
     identity (tunnels are provisioned atomically when the identity is
     created — there is no standalone ``POST /api/v1/tunnels``). The
     public URL is ``https://{agent_handle}.{env}.inkboxwire.com`` and
     the tunnel id is persisted in HERMES_HOME so subsequent runs reuse
     the same tunnel. Data-plane auth uses the SDK client's ``x-api-key``
     directly. Production deployments can bypass tunneling entirely by
     setting ``INKBOX_PUBLIC_URL``.
  2. PATCHes every mailbox + phone number on the configured identity
     so their webhook URLs / call WebSocket URL point at the tunnel.
  3. Starts an aiohttp server with two routes:
        - ``POST /webhook`` — verifies the ``X-Inkbox-Signature`` HMAC
          via the SDK, parses the body into one of three event shapes
          (mail / SMS / call), resolves the remote party to a Contact
          via ``inkbox.contacts.lookup()``, and pushes a
          :class:`MessageEvent` onto the gateway runner.
        - ``WS /phone/media/ws`` — live-call media bridge. Receives
          ``transcript`` events from Inkbox, hands each finalized
          transcript turn to the gateway as a MessageEvent, and pushes
          the agent's streamed response back as ``text`` frames for
          Inkbox-managed TTS playback.

Session keys
------------
Every inbound event is mapped to ``chat_id = contact_id`` so that one
Hermes session spans email + SMS + voice for the same remote party::

    inbound mail   → chat_id=contact_id, thread_id=f"email:{tid}"
    inbound SMS    → chat_id=contact_id, thread_id=None
    inbound call   → chat_id=contact_id, thread_id=f"call:{call_id}"
    outbound call  → chat_id=contact_id, thread_id=None  (joins the
                     contact's main session so the agent inherits the
                     conversation that decided to place the call)

When ``inkbox.contacts.lookup()`` returns 0 or >1 contacts the adapter
falls back to the raw email address / phone number as ``chat_id``, so
unknown senders still get a session — just not a contact-merged one.

Outbound
--------
``send()`` is mode-aware via ``metadata['mode']``:
  - ``email`` → ``identity.send_email(to=..., subject=..., body_text=...)``
  - ``sms``   → ``identity.send_text(to=..., text=...)``
  - ``voice`` → push a ``text`` frame onto the contact's active call
    WebSocket so Inkbox-managed TTS speaks it to the caller.

When the agent streams (gateway calls ``edit_message()`` repeatedly),
voice deltas are forwarded to the WS as incremental ``text`` events;
email and SMS edits are no-ops (the platforms have no native edit
semantics).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket as _socket
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

try:
    from aiohttp import WSMsgType, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    WSMsgType = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from inkbox import Inkbox, verify_webhook

    INKBOX_AVAILABLE = True
except ImportError:
    Inkbox = None  # type: ignore[assignment]
    verify_webhook = None  # type: ignore[assignment]
    INKBOX_AVAILABLE = False

try:
    from inkbox.tunnels.client import (
        TunnelListener,
        connect as inkbox_tunnel_connect,
    )

    INKBOX_TUNNEL_AVAILABLE = True
except ImportError:
    TunnelListener = None  # type: ignore[assignment]
    inkbox_tunnel_connect = None  # type: ignore[assignment]
    INKBOX_TUNNEL_AVAILABLE = False

from gateway.config import INKBOX_BASE_URL_DEFAULT, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.helpers import redact_phone

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_WEBHOOK_PATH = "/webhook"
DEFAULT_WS_PATH = "/phone/media/ws"
CONTACT_CACHE_TTL_SECONDS = 300
WEBHOOK_DEDUP_TTL_SECONDS = 300
SMS_MAX_LENGTH = 1600  # Inkbox SMS hard cap
SMS_TEXT_BATCH_DELAY_SECONDS = 0.0
SMS_TEXT_BATCH_MAX_MESSAGES = 8
SMS_TEXT_BATCH_MAX_CHARS = 4000

SMS_CONTROL_WORDS = frozenset({
    "start",
    "stop",
    "unstop",
    "help",
    "cancel",
    "end",
    "quit",
    "yes",
    "subscribe",
    "info",
    "unsubscribe",
})

# Stable ``error`` codes returned by the Inkbox SMS send endpoint. Sourced
# from the live server (apps/api_server/subapps/phone/send_text_service.py).
SMS_SENDER_PROVISIONING_ERROR_CODES = frozenset({
    "sender_sms_pending",
    "sender_sms_assignment_failed",
    "sender_not_registered",
    "sender_registration_required",
    "messaging_profile_disabled",
    "toll_free_sms_unsupported",
})
SMS_CONSENT_ERROR_CODES = frozenset({
    "recipient_not_opted_in",
    "recipient_opted_out",
    "recipient_blocked",
})
SMS_RATE_LIMIT_ERROR_CODES = frozenset({
    "carrier_rate_limit",
    "sender_rate_limited",
})
SMS_TRANSIENT_ERROR_CODES = frozenset({
    "carrier_unavailable",
})
SMS_PERMANENT_ERROR_CODES = frozenset({
    "invalid_phone_number",
    "message_too_long",
    "carrier_rejected",
})

# Hermes emits a few classes of admin/system notice via adapter.send() —
# session-reset banners ("◐ ..."), runtime info blocks ("◆ Model: ..."),
# the home-channel prompt ("📬 No home channel..."), update/restart notes
# ("🔄 ..."), check/x-mark status pings ("✓ ..." / "✗ ..."), and the
# warning prefix ("⚠️ ...").  These are CLI/terminal-style chatter that
# leaks into the user's actual mailbox or SMS thread on Inkbox.  Drop
# them at adapter.send() so they never get delivered as real messages.
_ADMIN_NOTICE_PREFIXES: Tuple[str, ...] = (
    "◐", "◆", "📬", "🔄", "✓", "✗", "⚠️", "⚠", "⚡", "💡", "⏳",
    # Tool-call narration glyphs — Hermes emits these as interim
    # "I'm running X right now" updates while streaming. They have no
    # place in a real SMS or email thread.
    "💻",  # terminal / bash
    "🔎",  # grep / search_files
    "📖",  # read
    "📝",  # write / edit
    "📚",  # skill load
    "📋",  # todo / task planning
    "🐍",  # exec / python
    "🌐",  # web fetch
    "🧠",  # thinking / reasoning
    "🛠",  # tool generic
    "🔧",  # tool generic alt
    # Save/cache/persistence glyph — covers the background self-improvement
    # review banner ("💾 Self-improvement review: User profile updated · …"),
    # prompt-cache + cached-context status pings, trajectory-compressor
    # "Metrics saved to …" notices, etc.
    "💾",
)

# Substrings that mark CLI/TUI runtime chatter even when the leading glyph is
# absent (some Hermes notices fold across sentences, e.g. the busy/queue tip
# arrives mid-paragraph after the ⚡ banner).  Match any of these → suppress.
_ADMIN_NOTICE_SUBSTRINGS: Tuple[str, ...] = (
    "Interrupting current task",
    "First-time tip",
    "/busy queue",
    "/busy steer",
    "/busy status",
    "Session automatically reset",
    "No home channel is set",
    "Still working",
    "min elapsed — iteration",
    "Cronjob Response:",
    # Belt-and-suspenders: the self-improvement banner is sometimes
    # forwarded with the leading glyph stripped upstream of us.
    "Self-improvement review:",
)


def _is_hermes_admin_notice(content: str) -> bool:
    """True when *content* is a Hermes-internal status/admin chatter line.

    Triggered when the message starts with one of the well-known glyphs
    Hermes uses to flag system messages in the CLI/TUI, or when the body
    contains any of ``_ADMIN_NOTICE_SUBSTRINGS``.  These have no business
    landing in a real human's email inbox, SMS thread, or — worst of all
    — being read aloud as TTS over a live phone call.
    """
    head = (content or "").lstrip().lstrip("﻿")
    if head.startswith(_ADMIN_NOTICE_PREFIXES):
        return True
    return any(s in head for s in _ADMIN_NOTICE_SUBSTRINGS)


def _float_setting(extra: Dict[str, Any], key: str, env_name: str, default: float) -> float:
    raw = extra[key] if key in extra else os.getenv(env_name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


def _int_setting(extra: Dict[str, Any], key: str, env_name: str, default: int) -> int:
    raw = extra[key] if key in extra else os.getenv(env_name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _parse_inkbox_timestamp(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _format_inkbox_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _format_sms_delta(first_at: datetime, current_at: datetime) -> str:
    seconds = max(0, int(round((current_at - first_at).total_seconds())))
    return f"+{seconds}s"


def _normalized_sms_control_word(text: str) -> Optional[str]:
    normalized = " ".join((text or "").strip().lower().split())
    return normalized if normalized in SMS_CONTROL_WORDS else None


def _plain_value(value: Any) -> Optional[str]:
    """Return enum-like values as strings without leaking object repr noise."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text or None


def _json_safe_detail(value: Any) -> Any:
    """Keep structured provider error details if they are JSON-safe."""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sms_error_body(detail: Any) -> Dict[str, Any]:
    """Normalize Inkbox API error payloads to the innermost detail dict."""
    if not isinstance(detail, dict):
        return {}
    nested = detail.get("detail")
    if isinstance(nested, dict):
        return nested
    return detail


def _classify_sms_error(
    status_code: Optional[int],
    error_code: Optional[str],
    message: str,
) -> Tuple[str, bool]:
    """Return ``(category, retryable)`` for an Inkbox SMS send failure."""
    code = (error_code or "").lower().strip()
    lower_msg = (message or "").lower()

    if code in SMS_TRANSIENT_ERROR_CODES:
        return ("transient", True)
    if code in SMS_SENDER_PROVISIONING_ERROR_CODES:
        return ("sender_provisioning", False)
    if code in SMS_CONSENT_ERROR_CODES:
        return ("recipient_consent", False)
    if code in SMS_RATE_LIMIT_ERROR_CODES:
        return ("rate_limit", False)
    if code in SMS_PERMANENT_ERROR_CODES:
        return ("permanent", False)

    if status_code in {408, 500, 502, 503, 504}:
        return ("transient", True)
    if status_code == 429:
        return ("rate_limit", False)
    if status_code == 409:
        return ("conflict", False)
    if status_code is not None and 400 <= status_code < 500:
        return ("permanent", False)

    if any(marker in lower_msg for marker in ("timeout", "temporar", "connection")):
        return ("transient", True)
    return ("sdk_error", False)


def _extract_inkbox_sms_error(exc: Exception) -> Dict[str, Any]:
    """Extract structured fields from SDK exceptions without depending on SDK types."""
    status_code = getattr(exc, "status_code", None)
    detail = getattr(exc, "detail", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    body = _sms_error_body(detail)
    error_code = _plain_value(
        body.get("error")
        or body.get("code")
        or getattr(exc, "code", None)
    )
    message = _plain_value(body.get("message"))
    if not message:
        nested = body.get("detail")
        message = nested if isinstance(nested, str) else None
    if not message:
        message = str(exc) or exc.__class__.__name__

    category, retryable = _classify_sms_error(status_code, error_code, message)
    return {
        "status_code": status_code,
        "error_code": error_code,
        "message": message,
        "detail": _json_safe_detail(detail),
        "category": category,
        "retryable": retryable,
    }


def _format_inkbox_sms_error(fields: Dict[str, Any]) -> str:
    status = fields.get("status_code")
    code = fields.get("error_code")
    message = fields.get("message") or "send_text failed"
    prefix = "Inkbox SMS send failed"
    if status:
        prefix += f" (HTTP {status})"
    if code:
        prefix += f" [{code}]"
    return f"{prefix}: {message}"


def _sms_send_failure(exc: Exception, *, to_number: str) -> SendResult:
    fields = _extract_inkbox_sms_error(exc)
    logger.error(
        "[Inkbox] SMS send failed to %s: status=%s code=%s category=%s retryable=%s message=%s",
        redact_phone(to_number),
        fields.get("status_code"),
        fields.get("error_code"),
        fields.get("category"),
        fields.get("retryable"),
        fields.get("message"),
    )
    return SendResult(
        success=False,
        error=_format_inkbox_sms_error(fields),
        raw_response={"platform": "inkbox", "mode": "sms", **fields},
        retryable=bool(fields.get("retryable")),
        fallback_allowed=False,
    )


def _sms_send_failure_dict(exc: Exception) -> Dict[str, Any]:
    fields = _extract_inkbox_sms_error(exc)
    return {
        "success": False,
        "platform": "inkbox",
        "mode": "sms",
        "error": _format_inkbox_sms_error(fields),
        **fields,
    }


def _text_message_metadata(message: Any, *, mode: str) -> Dict[str, Any]:
    """Small, non-body metadata payload for SendResult.raw_response."""
    return {
        "platform": "inkbox",
        "mode": mode,
        "message_id": _plain_value(getattr(message, "id", None)),
        "delivery_status": _plain_value(getattr(message, "delivery_status", None)),
        "status": _plain_value(getattr(message, "status", None)),
        "direction": _plain_value(getattr(message, "direction", None)),
        "type": _plain_value(getattr(message, "type", None)),
    }


def _inkbox_tunnel_state_dir() -> "Path":
    """Dedicated subdir of HERMES_HOME for the SDK's tunnel state.

    The SDK writes generic-named files inside ``state_dir`` —
    ``state.json``, ``private_key.pem``, ``cert_chain.pem`` — so we keep
    them in their own subfolder to avoid colliding with other state files
    Hermes itself owns under HERMES_HOME (e.g. ``state.json`` for sessions).
    """
    from pathlib import Path  # local — keep top-of-module import surface tight
    from hermes_cli.config import get_hermes_home
    return Path(get_hermes_home()) / "inkbox_tunnel"


def _wipe_inkbox_tunnel_state(state_dir: "Path") -> None:
    """Remove the three SDK-owned files inside ``state_dir``.

    Called on every connect so a stale ``tunnel_id`` referencing a tunnel
    that's been removed server-side can never block the next start. The
    directory itself is left in place; the SDK recreates contents during
    ``connect()``.
    """
    for name in ("state.json", "private_key.pem", "cert_chain.pem"):
        path = state_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug(
                "[Inkbox] Couldn't remove stale tunnel state file %s: %s",
                path, exc,
            )


def _slugify_for_tunnel(handle: str) -> str:
    """Convert an agent handle into a valid tunnel-name slug.

    Tunnel names must be 3-63 lowercase letters/digits/hyphens — same rules
    as a DNS label. Mirrors what the deleted ``inkbox_tunnel.derive_tunnel_name``
    helper did, kept inline now that it's a one-liner.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", (handle or "").lower()).strip("-")
    return slug[:63] if slug else "hermes-agent"


def check_inkbox_requirements() -> bool:
    """Return True iff the Python ``inkbox`` SDK and aiohttp are importable.

    The SDK ships its own tunnel runtime under ``inkbox.tunnels.client`` —
    no extra dependency beyond the inkbox extra is needed when running
    against inkboxwire.com. Operators behind their own reverse proxy /
    hosted tunnel can set ``INKBOX_PUBLIC_URL`` and skip the tunnel path
    entirely.
    """
    return INKBOX_AVAILABLE and AIOHTTP_AVAILABLE


class InkboxAdapter(BasePlatformAdapter):
    """Hermes platform adapter for Inkbox (email + SMS + voice)."""

    MAX_MESSAGE_LENGTH = 4096  # email/voice are unbounded; SMS chunked separately in send()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.INKBOX)
        extra = config.extra or {}

        self._api_key = (
            extra.get("api_key") or os.getenv("INKBOX_API_KEY") or ""
        ).strip()
        self._signing_key = (
            extra.get("signing_key") or os.getenv("INKBOX_SIGNING_KEY") or ""
        ).strip()
        self._identity_handle = (
            extra.get("identity") or os.getenv("INKBOX_IDENTITY") or ""
        ).strip()
        self._base_url = (
            extra.get("base_url") or os.getenv("INKBOX_BASE_URL") or INKBOX_BASE_URL_DEFAULT
        ).strip()
        self._host = str(extra.get("host") or os.getenv("INKBOX_HOST") or DEFAULT_HOST)
        self._port = int(
            extra.get("port") or os.getenv("INKBOX_LISTEN_PORT") or DEFAULT_PORT
        )
        self._webhook_path = str(extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH)
        self._ws_path = str(extra.get("ws_path") or DEFAULT_WS_PATH)
        self._public_url_override = (
            extra.get("public_url") or os.getenv("INKBOX_PUBLIC_URL") or ""
        ).strip()
        self._tunnel_name_override = (
            extra.get("tunnel_name") or os.getenv("INKBOX_TUNNEL_NAME") or ""
        ).strip().lower()
        # Gate the start-time guard + per-webhook verify block. Defaults to
        # true so a missing INKBOX_SIGNING_KEY fails loudly instead of
        # silently accepting unsigned traffic from anyone who finds the
        # tunnel URL.
        #
        # Explicit `in extra` check (not `extra.get(...) or ...`) so that
        # a config-level boolean False isn't silently coalesced into the
        # env default — `False or "true"` evaluates to "true".
        if "require_signature" in extra:
            raw_require_signature = extra["require_signature"]
        else:
            raw_require_signature = os.getenv("INKBOX_REQUIRE_SIGNATURE", "true")
        self._require_signature = str(raw_require_signature).lower() not in ("false", "0", "no")
        self._sms_text_batch_delay_seconds = _float_setting(
            extra,
            "sms_text_batch_delay_seconds",
            "INKBOX_SMS_TEXT_BATCH_DELAY_SECONDS",
            SMS_TEXT_BATCH_DELAY_SECONDS,
        )
        self._sms_text_batch_max_messages = _int_setting(
            extra,
            "sms_text_batch_max_messages",
            "INKBOX_SMS_TEXT_BATCH_MAX_MESSAGES",
            SMS_TEXT_BATCH_MAX_MESSAGES,
        )
        self._sms_text_batch_max_chars = _int_setting(
            extra,
            "sms_text_batch_max_chars",
            "INKBOX_SMS_TEXT_BATCH_MAX_CHARS",
            SMS_TEXT_BATCH_MAX_CHARS,
        )

        # Live state.
        self._inkbox: Optional[Any] = None
        self._public_url: Optional[str] = None
        self._public_host: Optional[str] = None
        self._app: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._site: Optional[Any] = None
        self._tunnel: Optional[Any] = None  # inkbox.tunnels.client.TunnelListener
        self._tunnel_runtime_thread: Optional["threading.Thread"] = None
        # contact_id → active call WebSocket. Used by send()/edit_message() to
        # push voice replies to the correct ongoing call.
        self._active_call_ws: Dict[str, Any] = {}
        # Per-WS metadata so the WS handler can rebuild the source on each turn.
        self._call_ws_meta: Dict[int, Dict[str, Any]] = {}
        # ((kind, value) → (contact_id, contact_name, expires_at)).  TTL cache
        # for inkbox.contacts.lookup() — every inbound event resolves the
        # remote party to a Contact, and the same number/email shows up
        # repeatedly within a single conversation.
        self._contact_cache: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str], float]] = {}
        # Webhook dedup by ``X-Inkbox-Request-Id`` (Inkbox retries on timeout).
        self._seen_request_ids: Dict[str, float] = {}
        # session_key → pending inbound SMS fragments waiting for a quiet
        # window. Human SMS users often send fragments/corrections in bursts;
        # batching keeps one thought from becoming several self-interrupting
        # agent turns.
        self._pending_sms_text_batches: Dict[str, Dict[str, Any]] = {}
        self._pending_sms_text_batch_tasks: Dict[str, asyncio.Task] = {}
        # chat_id → metadata of the most-recent inbound email for that chat.
        # Used by send()'s email branch to populate Re: <subject> and the
        # In-Reply-To header so replies thread into the original conversation
        # in the recipient's mail client.
        self._last_inbound_email: Dict[str, Dict[str, str]] = {}
        # chat_id → modality of the most-recent inbound message ('email',
        # 'sms', or 'voice').  Critical when chat_id is a Contact UUID (no
        # `+` or `@` to disambiguate) — without this, send() defaults the
        # mode by chat_id shape and would email an SMS reply.
        self._last_inbound_modality: Dict[str, str] = {}
        # chat_id → unix timestamp at which the contact's call WS most-recently
        # closed.  send() consults this to drop replies generated during the
        # short window after a call ends — when the agent's last in-call turn
        # finishes generating after the WS is gone, the response would
        # otherwise fall through to the email/SMS default and leak the
        # voice-intended text into the user's inbox.
        self._voice_recently_closed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not check_inkbox_requirements():
            logger.warning(
                "[Inkbox] aiohttp or `inkbox` SDK not installed. "
                "Run: pip install 'hermes-agent[inkbox]' or `pip install inkbox aiohttp`",
            )
            return False
        if not self._api_key:
            logger.warning("[Inkbox] INKBOX_API_KEY not set")
            return False
        if not self._identity_handle:
            logger.warning("[Inkbox] INKBOX_IDENTITY not set")
            return False
        if self._require_signature and not self._signing_key:
            logger.warning(
                "[Inkbox] INKBOX_SIGNING_KEY not set and "
                "INKBOX_REQUIRE_SIGNATURE is enabled; refusing to start. "
                "Generate a signing key in https://inkbox.ai/console/signing-keys "
                "or set INKBOX_REQUIRE_SIGNATURE=false for local-only testing.",
            )
            return False

        if not self._acquire_platform_lock(
            scope="inkbox",
            identity=self._identity_handle,
            resource_desc=f"Inkbox identity '{self._identity_handle}'",
        ):
            return False

        # Check the listen port is free before trying to bind.
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error("[Inkbox] Port %d already in use", self._port)
            self._release_platform_lock()
            return False
        except (ConnectionRefusedError, OSError):
            pass

        try:
            self._inkbox = Inkbox(api_key=self._api_key, base_url=self._base_url)
        except Exception as exc:
            logger.error("[Inkbox] Failed to construct SDK client: %s", exc)
            self._release_platform_lock()
            return False

        # Start the local aiohttp server FIRST. The SDK's tunnel runtime opens
        # its data-plane connection during ``tunnel_connect`` and starts
        # forwarding immediately, so the local server has to be accepting
        # before we hand inkboxwire.com a forward-to URL.
        try:
            self._app = web.Application()
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_post(self._webhook_path, self._handle_webhook)
            self._app.router.add_get(self._ws_path, self._handle_call_ws)
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except Exception:
            logger.exception("[Inkbox] Failed to start HTTP server")
            await self._cleanup()
            self._release_platform_lock()
            return False

        # Resolve the public URL: explicit override wins, else open an
        # SDK-managed tunnel against inkboxwire.com.
        if self._public_url_override:
            self._public_url = self._public_url_override.rstrip("/")
            self._public_host = urlparse(self._public_url).netloc
        elif INKBOX_TUNNEL_AVAILABLE:
            if not await self._provision_inkbox_tunnel():
                await self._cleanup()
                self._release_platform_lock()
                return False
        else:
            logger.error(
                "[Inkbox] No public URL configured. Set INKBOX_PUBLIC_URL or "
                "install the inkbox extra: pip install 'hermes-agent[inkbox]'.",
            )
            await self._cleanup()
            self._release_platform_lock()
            return False

        # PATCH the identity's mailboxes + phone numbers to point at this server.
        try:
            await asyncio.to_thread(self._patch_identity_objects)
        except Exception:
            logger.exception("[Inkbox] Failed to patch identity webhook URLs")
            await self._cleanup()
            self._release_platform_lock()
            return False

        self._mark_connected()
        logger.info(
            "[Inkbox] Connected: identity=%s public=%s listen=%s:%d",
            self._identity_handle, self._public_url, self._host, self._port,
        )

        return True

    async def disconnect(self) -> None:
        self._running = False
        await self._cleanup()
        self._release_platform_lock()
        self._mark_disconnected()
        logger.info("[Inkbox] Disconnected")

    async def _cleanup(self) -> None:
        for task in list(self._pending_sms_text_batch_tasks.values()):
            if not task.done():
                task.cancel()
        if self._pending_sms_text_batch_tasks:
            await asyncio.gather(
                *self._pending_sms_text_batch_tasks.values(),
                return_exceptions=True,
            )
        self._pending_sms_text_batch_tasks.clear()
        self._pending_sms_text_batches.clear()

        # Close any live call WS so callers don't hang on a half-open socket.
        for ws in list(self._active_call_ws.values()):
            with suppress(Exception):
                await ws.close()
        self._active_call_ws.clear()
        self._call_ws_meta.clear()

        if self._site is not None:
            with suppress(Exception):
                await self._site.stop()
            self._site = None
        if self._runner is not None:
            with suppress(Exception):
                await self._runner.cleanup()
            self._runner = None
        self._app = None
        if self._tunnel is not None:
            # ``listener.close`` is sync; offload so it doesn't block the loop.
            with suppress(Exception):
                await asyncio.to_thread(self._tunnel.close)
            self._tunnel = None
        if self._tunnel_runtime_thread is not None:
            # close() unblocks wait(); join briefly so logs flush in order.
            with suppress(Exception):
                await asyncio.to_thread(
                    self._tunnel_runtime_thread.join, 5.0,
                )
            self._tunnel_runtime_thread = None
        if self._inkbox is not None:
            with suppress(Exception):
                self._inkbox.close()
            self._inkbox = None

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------

    async def _provision_inkbox_tunnel(self) -> bool:
        """Open an SDK-managed tunnel to inkboxwire.com.

        ``inkbox.tunnels.client.connect`` only does the control-plane
        registration; the runtime thread that actually opens the h2 data
        plane is started lazily inside ``listener.wait()``. We don't want
        to block the gateway event loop on ``wait()``, so we spawn a small
        background thread to drive it — which gives us a live data plane
        plus a place to log any runtime error the listener captures.

        Data-plane auth uses the SDK client's ``x-api-key`` (admin-scoped
        or scoped to this tunnel's owning identity), so there is no
        connect_secret to mint, rotate, or persist on this side.

        SDK-managed state (state.json, private_key.pem, cert_chain.pem)
        lives in a dedicated subdir of HERMES_HOME so its generic filenames
        don't collide with Hermes' own state files. We wipe that subdir on
        every connect so a stale ``tunnel_id`` referencing a tunnel that's
        been removed server-side can never put us in a TunnelRemoved loop.
        """
        import threading
        from inkbox.tunnels.exceptions import TunnelNotProvisioned

        tunnel_name = self._tunnel_name_override or _slugify_for_tunnel(
            self._identity_handle,
        )
        forward_to = f"http://127.0.0.1:{self._port}"
        state_dir = _inkbox_tunnel_state_dir()

        _wipe_inkbox_tunnel_state(state_dir)

        try:
            # ``connect`` is sync (does an HTTPS round-trip + opens the data
            # plane); offload to a thread so the gateway event loop isn't
            # blocked. The returned listener owns its own supervisor threads.
            self._tunnel = await asyncio.to_thread(
                inkbox_tunnel_connect,
                self._inkbox,
                name=tunnel_name,
                forward_to=forward_to,
                state_dir=state_dir,
            )
        except TunnelNotProvisioned:
            # The identity exists but its tunnel does not (1:1 invariant
            # broken upstream, or running against an identity created
            # before the data-model migration landed). Surface a clear
            # message — recovery is to recreate the identity, which
            # atomically provisions the tunnel again.
            logger.error(
                "[Inkbox] No tunnel provisioned for handle %r — recreate "
                "the identity via the Inkbox console or `inkbox identity "
                "create`, then restart the gateway.",
                tunnel_name,
            )
            self._tunnel = None
            return False
        except Exception:
            logger.exception("[Inkbox] Failed to open SDK tunnel")
            self._tunnel = None
            return False

        # Drive the listener's runtime in a daemon thread. ``wait()`` calls
        # ``_start_thread_if_needed()`` internally — that's what actually
        # spawns the data-plane runtime thread. Without this, ``connect()``
        # returns a listener whose runtime never starts and inkboxwire.com
        # gets a "no agent connected" 503 for every inbound webhook.
        def _drive_listener(listener):
            try:
                listener.wait()
            except KeyboardInterrupt:
                pass
            except Exception:
                logger.exception(
                    "[Inkbox] Tunnel runtime exited with error",
                )

        self._tunnel_runtime_thread = threading.Thread(
            target=_drive_listener,
            args=(self._tunnel,),
            name="inkbox-tunnel-wait",
            daemon=True,
        )
        self._tunnel_runtime_thread.start()

        self._public_url = self._tunnel.public_url.rstrip("/")
        self._public_host = self._tunnel.tunnel.public_host
        logger.info(
            "[Inkbox] Tunnel ready: %s → 127.0.0.1:%d",
            self._public_url, self._port,
        )
        return True

    def _patch_identity_objects(self) -> None:
        """Point every mailbox + phone number on the identity at this server."""
        webhook_url = f"{self._public_url}{self._webhook_path}"
        ws_url = f"wss://{self._public_host}{self._ws_path}"

        identity = self._inkbox.get_identity(self._identity_handle)

        # Mailbox: webhook for inbound mail events.
        if identity.mailbox is not None:
            self._inkbox.mailboxes.update(
                identity.mailbox.email_address,
                webhook_url=webhook_url,
            )
            logger.info(
                "[Inkbox] Patched mailbox %s → %s",
                identity.mailbox.email_address, webhook_url,
            )

        # Phone number: text webhook + incoming-call action.
        # ``incoming_call_action="auto_accept"`` tells Inkbox to pick up the
        # call itself and immediately open a WS to ``client_websocket_url``,
        # without round-tripping a webhook first.  Lower setup latency than
        # ``webhook`` mode, and the call context arrives on the WS itself
        # via the ``x-call-context`` header (parsed in ``_handle_call_ws``).
        if identity.phone_number is not None:
            self._inkbox.phone_numbers.update(
                identity.phone_number.id,
                incoming_text_webhook_url=webhook_url,
                incoming_call_webhook_url=webhook_url,
                incoming_call_action="auto_accept",
                client_websocket_url=ws_url,
            )
            logger.info(
                "[Inkbox] Patched phone %s → %s + %s",
                identity.phone_number.number, webhook_url, ws_url,
            )

        # Persist the resolved identity so non-Inkbox sessions (CLI, etc.) can
        # tell the agent which email + phone it can be reached on.  Read by
        # ``prompt_builder.build_inkbox_identity_hint``.
        self._write_identity_state(identity, webhook_url, ws_url)

    def _write_identity_state(self, identity, webhook_url: str, ws_url: str) -> None:
        # Atomic write (tmp + os.replace) so a concurrent reader — e.g.
        # ``prompt_builder.build_inkbox_identity_hint`` running in another
        # process at agent start — can never observe a half-written file.
        # Matches the pattern used by feishu_comment_rules._save_pairing and
        # the google_chat adapter's thread-count store.
        try:
            from hermes_cli.config import get_hermes_home
            state_path = get_hermes_home() / "inkbox_identity_state.json"
            state = {
                "handle": self._identity_handle,
                "email_address": (
                    getattr(identity.mailbox, "email_address", None)
                    if identity.mailbox else None
                ),
                "phone_number": (
                    getattr(identity.phone_number, "number", None)
                    if identity.phone_number else None
                ),
                "phone_number_id": (
                    str(getattr(identity.phone_number, "id", ""))
                    if identity.phone_number else None
                ),
                "public_url": self._public_url,
                "webhook_url": webhook_url,
                "ws_url": ws_url,
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(tmp_path, state_path)
        except Exception as exc:
            logger.debug("[Inkbox] Failed to write identity state file: %s", exc)

    # ------------------------------------------------------------------
    # Outbound: send / edit / get_chat_info
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Dispatch a message via the right Inkbox modality.

        ``metadata['mode']`` selects the channel: ``email`` (default for
        contacts that have an email on file), ``sms``, or ``voice``. For
        voice mode the message is pushed onto the contact's active call
        WebSocket — the caller hears it through Inkbox-managed TTS.

        Hermes admin / status banners (session-reset notices, runtime info,
        home-channel prompt, update notifications) are silently dropped —
        these are CLI chatter that never belongs in a real user's email or
        SMS thread.  See ``_is_hermes_admin_notice`` for the prefix list.
        """
        if _is_hermes_admin_notice(content):
            logger.debug(
                "[Inkbox] Suppressed admin notice for chat %s: %s…",
                chat_id, (content or "")[:60].replace("\n", " "),
            )
            return SendResult(success=True, message_id="suppressed-admin-notice")

        # The [SILENT] marker is the cron scheduler's "I have nothing to
        # say" sentinel and is also instructed to the agent in the
        # post-call synthetic [call_ended] turn (see _handle_call_ws).
        # If the agent emits it through any send() path — including a
        # send_message tool call that picks email/SMS as the channel —
        # drop the send entirely. A bare [SILENT] email is never a
        # message a human wants to receive.
        if (content or "").strip().upper() == "[SILENT]":
            logger.info(
                "[Inkbox] Suppressed [SILENT] sentinel for chat %s",
                chat_id,
            )
            return SendResult(success=True, message_id="suppressed-silent-marker")

        meta = metadata or {}
        mode = (meta.get("mode") or "").lower().strip()

        # End-of-call grace window: when a voice call ends, the agent's last
        # in-flight turn often finishes generating *after* the WS has closed.
        # Without this guard the response falls through to the email/SMS
        # default and leaks the voice-intended text into the user's inbox.
        #
        # Drop when ALL of:
        #   - call WS just closed for this chat within VOICE_GRACE_SECONDS
        #   - no active call WS now (so we cannot ride the WS)
        #   - no fresh non-voice inbound has arrived since the close (which
        #     would have repopulated ``_last_inbound_modality`` and made this
        #     a legitimate SMS/email reply, not stale voice content).
        #
        # Note: we intentionally suppress regardless of whether the caller
        # passed an explicit ``mode``. An explicit ``mode='email'`` from a
        # post-call send_message tool call is exactly the case that bit us
        # (the agent's "reflect on the call" reply leaked out as an email).
        VOICE_GRACE_SECONDS = 60
        closed_at = self._voice_recently_closed.get(str(chat_id))
        if (
            closed_at is not None
            and (time.time() - closed_at) < VOICE_GRACE_SECONDS
            and chat_id not in self._active_call_ws
            and not self._last_inbound_modality.get(str(chat_id))
        ):
            logger.info(
                "[Inkbox] Suppressed post-call voice-leakage for chat %s: %s…",
                chat_id, (content or "")[:60].replace("\n", " "),
            )
            return SendResult(success=True, message_id="suppressed-post-call-leak")
        # Garbage-collect stale entries so the dict doesn't grow unbounded.
        if closed_at is not None and (time.time() - closed_at) > VOICE_GRACE_SECONDS:
            self._voice_recently_closed.pop(str(chat_id), None)

        # Resolve mode if the gateway didn't pass one explicitly.  Order of
        # preference:
        #   1. An open live-call WebSocket on this chat — voice trumps
        #      everything because dropping it would leave the caller hearing
        #      silence while we send an email.
        #   2. The modality of the most-recent inbound from this chat —
        #      SMS-conversations on contact-UUID chat_ids land here (the
        #      chat_id shape doesn't reveal which channel inbound came in
        #      on).
        #   3. SMS if the chat target itself looks like an E.164 number.
        #   4. Email otherwise (contact UUIDs, raw email addresses).
        if not mode and chat_id in self._active_call_ws:
            mode = "voice"
        if not mode:
            mode = self._last_inbound_modality.get(str(chat_id), "")
        if not mode:
            mode = "sms" if str(chat_id).startswith("+") else "email"

        # Voice replies ride the per-call WebSocket the WS handler keeps
        # open for the duration of the call.  No SDK round-trip.
        if mode == "voice":
            ws = self._active_call_ws.get(chat_id)
            if ws is None:
                return SendResult(
                    success=False,
                    error=(
                        f"No active call WebSocket for chat_id={chat_id}. "
                        "Voice replies require an open call."
                    ),
                )
            turn_id = str(meta.get("turn_id") or "")
            try:
                # Two-frame protocol matching the legacy phone-bridge: a
                # delta carrying the text, then a final ``done: true`` frame
                # that flushes Inkbox's TTS and ends the turn.
                await ws.send_str(json.dumps(
                    {"event": "text", "delta": content, "turn_id": turn_id}
                ))
                await ws.send_str(json.dumps(
                    {"event": "text", "done": True, "turn_id": turn_id}
                ))
                return SendResult(success=True)
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)

        if self._inkbox is None:
            return SendResult(success=False, error="Inkbox SDK client not initialized")

        try:
            identity = await asyncio.to_thread(
                self._inkbox.get_identity, self._identity_handle,
            )
        except Exception as exc:
            return SendResult(success=False, error=f"get_identity failed: {exc}")

        if mode == "sms":
            to_number = str(meta.get("to_phone") or chat_id).strip()
            if not to_number.startswith("+"):
                # chat_id is a contact UUID (or unknown shape) — look up the
                # primary phone number on the contact record.
                to_number = await asyncio.to_thread(self._lookup_contact_phone, chat_id)
                if not to_number:
                    return SendResult(
                        success=False,
                        error=f"No phone number on contact {chat_id}",
                    )
            try:
                msg = await asyncio.to_thread(
                    identity.send_text, to=to_number, text=content[:SMS_MAX_LENGTH],
                )
                raw_response = _text_message_metadata(msg, mode="sms")
                logger.info(
                    "[Inkbox] SMS queued to %s: id=%s delivery_status=%s",
                    redact_phone(to_number),
                    raw_response.get("message_id") or "",
                    raw_response.get("delivery_status") or raw_response.get("status") or "",
                )
                return SendResult(
                    success=True,
                    message_id=str(getattr(msg, "id", "")),
                    raw_response=raw_response,
                )
            except Exception as exc:
                return _sms_send_failure(exc, to_number=to_number)

        if mode == "email":
            to_addr = (meta.get("to_email") or "").strip()
            if not to_addr:
                # If the chat_id already looks like an email address, use it
                # directly — this is the unknown-sender path where the
                # contact lookup at ingest returned 0 matches and the raw
                # email became the chat_id.  Only try contacts.get() when the
                # chat_id is a contact UUID we can actually fetch.
                if "@" in str(chat_id):
                    to_addr = str(chat_id).strip()
                else:
                    to_addr = await asyncio.to_thread(self._lookup_contact_email, chat_id)
            if not to_addr:
                return SendResult(
                    success=False,
                    error=f"No email address on contact {chat_id}",
                )

            # Threading: prefer the inbound RFC 5322 Message-ID we stashed in
            # _on_mail_received, fall back to whatever the gateway passed in
            # as ``reply_to``.  Subject defaults to ``Re: <inbound subject>``
            # when replying to a known thread, ``(no subject)`` when sending
            # cold.  Mail clients use both signals (header + subject) to group
            # the message into the original conversation.
            stash = self._last_inbound_email.get(str(chat_id), {})
            in_reply_to = (
                meta.get("in_reply_to_message_id")
                or reply_to
                or stash.get("rfc_message_id")
                or None
            )
            inbound_subject = stash.get("subject", "")
            if meta.get("subject"):
                subject = str(meta["subject"])
            elif inbound_subject:
                # Don't double-prefix if the agent's reply target already had Re:.
                if inbound_subject.lower().startswith("re:"):
                    subject = inbound_subject
                else:
                    subject = f"Re: {inbound_subject}"
            else:
                subject = "(no subject)"

            try:
                msg = await asyncio.to_thread(
                    identity.send_email,
                    to=[to_addr],
                    subject=subject,
                    body_text=content,
                    in_reply_to_message_id=in_reply_to or None,
                )
                return SendResult(success=True, message_id=str(getattr(msg, "id", "")))
            except Exception as exc:
                return SendResult(success=False, error=f"send_email failed: {exc}")

        return SendResult(success=False, error=f"Unknown Inkbox send mode: {mode!r}")

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Stream incremental deltas to an open call. No-op for mail/SMS."""
        ws = self._active_call_ws.get(chat_id)
        if ws is None:
            return SendResult(success=False, error="Not supported")
        # Same admin-notice guard as send() — runtime banners are even more
        # offensive when read aloud over a live call than when delivered as
        # text to email/SMS.
        if _is_hermes_admin_notice(content):
            return SendResult(success=True, message_id="suppressed-admin-notice")
        try:
            # Match the bridge's two-frame protocol — Inkbox's TTS pipeline
            # mixes ``delta`` and ``done`` into separate frames rather than
            # one combined message.
            if content:
                await ws.send_str(json.dumps(
                    {"event": "text", "delta": content, "turn_id": message_id}
                ))
            if finalize:
                await ws.send_str(json.dumps(
                    {"event": "text", "done": True, "turn_id": message_id}
                ))
            return SendResult(success=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return ``{name, type, chat_id}`` for a contact-keyed chat."""
        info = {"name": chat_id, "type": "dm", "chat_id": chat_id}
        if self._inkbox is None:
            return info
        try:
            contact = await asyncio.to_thread(self._inkbox.contacts.get, chat_id)
        except Exception:
            return info
        info["name"] = (
            getattr(contact, "preferred_name", None)
            or getattr(contact, "given_name", None)
            or chat_id
        )
        return info

    # ------------------------------------------------------------------
    # Inbound: webhook handler
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({
            "status": "ok",
            "platform": "inkbox",
            "identity": self._identity_handle,
            "public_url": self._public_url,
        })

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        # Verify HMAC over the raw body before doing anything with it — both
        # public-URL and tunnel-delivered traffic hits this handler, so this
        # is the one chokepoint where we can refuse spoofed requests.
        if self._require_signature:
            ok = verify_webhook(
                payload=body,
                headers=dict(request.headers),
                secret=self._signing_key,
            )
            if not ok:
                return web.Response(status=401, text="invalid signature")

        request_id = request.headers.get("X-Inkbox-Request-Id", "")
        if request_id and self._is_duplicate(request_id):
            return web.Response(status=200, text="duplicate")

        try:
            envelope = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")

        # Mail / SMS webhooks come wrapped in {event_type, data:{...}}; the
        # incoming-call webhook is delivered as a flat object with a
        # ``phone_number_id`` field at the top level (no envelope).
        event_type = envelope.get("event_type")
        if event_type == "message.received":
            return await self._on_mail_received(envelope)
        if event_type and event_type.startswith("message."):
            # Outbound mail lifecycle (sent/delivered/bounced/failed) — log only.
            return web.Response(status=200, text="ok")
        if event_type == "text.received":
            return await self._on_text_received(envelope)
        if event_type and event_type.startswith("text."):
            return await self._on_text_lifecycle(envelope)
        if "phone_number_id" in envelope and "remote_phone_number" in envelope:
            return await self._on_incoming_call(envelope)
        return web.Response(status=200, text="ignored")

    def _is_duplicate(self, request_id: str) -> bool:
        now = time.time()
        # Prune expired entries opportunistically.
        if len(self._seen_request_ids) > 2000:
            self._seen_request_ids = {
                rid: ts
                for rid, ts in self._seen_request_ids.items()
                if now - ts < WEBHOOK_DEDUP_TTL_SECONDS
            }
        prior = self._seen_request_ids.get(request_id)
        if prior is not None and now - prior < WEBHOOK_DEDUP_TTL_SECONDS:
            return True
        self._seen_request_ids[request_id] = now
        return False

    async def _on_mail_received(self, envelope: Dict[str, Any]) -> "web.Response":
        message = (envelope.get("data") or {}).get("message") or {}
        from_address = (message.get("from_address") or "").strip().lower()
        if not from_address:
            return web.Response(status=200, text="ok")

        contact = await self._resolve_contact_full(kind="email", value=from_address)
        chat_id = (contact["id"] if contact else from_address)
        contact_name = contact["name"] if contact and contact.get("name") else None
        thread_id = message.get("thread_id")
        rfc_message_id = message.get("message_id")  # RFC 5322 Message-ID for threading
        subject = message.get("subject") or ""

        # Stash the subject + RFC 5322 Message-ID so send() can populate
        # Re: <subject> and the In-Reply-To header on replies.  Keyed by
        # chat_id so unsolicited cron sends to the same chat fall back to
        # the most-recent inbound for threading context.
        self._last_inbound_email[str(chat_id)] = {
            "subject": subject,
            "rfc_message_id": rfc_message_id or "",
            "from_address": from_address,
        }
        self._last_inbound_modality[str(chat_id)] = "email"

        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=contact_name or from_address,
            chat_type="dm",
            user_id=str(chat_id),
            user_name=contact_name or from_address,
            thread_id=f"email:{thread_id}" if thread_id else None,
            chat_topic=subject or None,
            # MessageEvent.message_id is what the gateway passes back as
            # ``reply_to`` on send().  Use the RFC 5322 Message-ID (not the
            # Inkbox UUID) so SDK send_email(in_reply_to_message_id=...)
            # actually threads the reply.
            message_id=rfc_message_id or message.get("id"),
        )
        body_text = message.get("snippet") or subject or ""
        # Modality marker — every inbound is prefixed with one line that
        # tells the agent which modality + which Inkbox Contact (if any)
        # this message belongs to.  PLATFORM_HINTS["inkbox"] explains how
        # the agent should use this and tells it never to echo the line.
        contact_block = self._contact_marker(contact)
        tagged = (
            f"[inkbox:email from={from_address}"
            f"{f' subject={subject!r}' if subject else ''}"
            f" | {contact_block}]\n{body_text}"
        )
        event = MessageEvent(
            text=tagged,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=envelope,
            message_id=rfc_message_id or str(message.get("id") or ""),
            # Auto-load the Inkbox SDK skill on the first turn of every new
            # session so the agent has the SDK reference (texts.list,
            # iter_emails, contacts.create, etc.) in conversation history.
            auto_skill="inkbox",
        )
        await self._enqueue(event)
        return web.Response(status=200, text="ok")

    def _sms_text_batch_key(self, event: MessageEvent) -> str:
        from gateway.session import build_session_key

        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    @staticmethod
    def _sms_text_batch_chars(batch: Dict[str, Any]) -> int:
        return sum(len(str(fragment.get("text") or "")) for fragment in batch["fragments"])

    def _build_sms_text_event(
        self,
        *,
        envelope: Dict[str, Any],
        text_id: str,
        remote: str,
        contact: Optional[Dict[str, Any]],
        chat_id: Any,
        contact_name: Optional[str],
        body: str,
        timestamp: datetime,
        text: Optional[str] = None,
        message_type: MessageType = MessageType.TEXT,
    ) -> MessageEvent:
        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=contact_name or remote,
            chat_type="dm",
            user_id=str(chat_id),
            user_name=contact_name or remote,
            message_id=text_id,
        )
        if text is None:
            contact_block = self._contact_marker(contact)
            text = f"[inkbox:sms from={remote} | {contact_block}]\n{body}"
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=envelope,
            message_id=text_id,
            auto_skill="inkbox" if message_type == MessageType.TEXT else None,
            timestamp=timestamp,
        )

    async def _enqueue_sms_text_event(self, event: MessageEvent) -> None:
        key = self._sms_text_batch_key(event)
        text = event.text or ""
        marker, body = text.split("\n", 1) if "\n" in text else ("[inkbox:sms]", text)
        batch = self._pending_sms_text_batches.get(key)
        if batch is not None:
            next_count = len(batch["fragments"]) + 1
            next_chars = self._sms_text_batch_chars(batch) + len(body)
            if (
                next_count > self._sms_text_batch_max_messages
                or next_chars > self._sms_text_batch_max_chars
            ):
                await self._flush_sms_text_batch_now(key)
                batch = self._pending_sms_text_batches.get(key)

        if batch is None:
            batch = {
                "marker": marker,
                "fragments": [],
                "raw_messages": [],
            }
            self._pending_sms_text_batches[key] = batch

        batch["fragments"].append({
            "text": body,
            "timestamp": event.timestamp,
            "message_id": event.message_id,
            "source": event.source,
        })
        batch["raw_messages"].append(event.raw_message)
        batch["last_event"] = event

        if self._sms_text_batch_delay_seconds <= 0:
            await self._flush_sms_text_batch_now(key)
            return

        prior_task = self._pending_sms_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_sms_text_batch_tasks[key] = asyncio.create_task(
            self._flush_sms_text_batch_after_delay(key)
        )

    async def _flush_sms_text_batch_after_delay(self, key: str) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._sms_text_batch_delay_seconds)
            await self._flush_sms_text_batch_now(key)
        finally:
            if self._pending_sms_text_batch_tasks.get(key) is current_task:
                self._pending_sms_text_batch_tasks.pop(key, None)

    async def _flush_sms_text_batch_now(self, key: str) -> None:
        current_task = asyncio.current_task()
        task = self._pending_sms_text_batch_tasks.get(key)
        if task is not None and task is not current_task and not task.done():
            task.cancel()
        if task is not None and task is not current_task:
            self._pending_sms_text_batch_tasks.pop(key, None)

        batch = self._pending_sms_text_batches.pop(key, None)
        if not batch:
            return

        fragments = batch["fragments"]
        event = batch["last_event"]
        if len(fragments) == 1:
            body = fragments[0]["text"]
            event.text = f"{batch['marker']}\n{body}"
        else:
            first_at = fragments[0]["timestamp"]
            last_at = fragments[-1]["timestamp"]
            burst_marker = batch["marker"].replace(
                "[inkbox:sms ",
                (
                    f"[inkbox:sms_burst messages={len(fragments)} "
                    f"first_at={_format_inkbox_timestamp(first_at)} "
                    f"last_at={_format_inkbox_timestamp(last_at)} "
                ),
                1,
            )
            lines = [burst_marker]
            for fragment in fragments:
                delta = _format_sms_delta(first_at, fragment["timestamp"])
                lines.append(f"[{delta}] {fragment['text']}")
            event.text = "\n".join(lines)
            event.raw_message = {
                "event_type": "text.received.batch",
                "items": batch["raw_messages"],
            }
        event.message_id = fragments[-1].get("message_id") or event.message_id
        event.source = fragments[-1].get("source") or event.source
        event.timestamp = fragments[-1].get("timestamp") or event.timestamp
        await self._enqueue(event)

    async def _on_text_received(self, envelope: Dict[str, Any]) -> "web.Response":
        text_msg = (envelope.get("data") or {}).get("text_message") or {}
        text_id = str(text_msg.get("id") or "").strip()
        if text_id and self._is_duplicate(f"text:{text_id}"):
            return web.Response(status=200, text="duplicate")
        direction = str(text_msg.get("direction") or "").strip().lower()
        if direction and direction != "inbound":
            return web.Response(status=200, text="ok")
        remote = (text_msg.get("remote_phone_number") or "").strip()
        if not remote:
            return web.Response(status=200, text="ok")

        contact = await self._resolve_contact_full(kind="phone", value=remote)
        chat_id = (contact["id"] if contact else remote)
        contact_name = contact["name"] if contact and contact.get("name") else None
        body = text_msg.get("text") or ""
        timestamp = _parse_inkbox_timestamp(text_msg.get("created_at"))

        control_word = _normalized_sms_control_word(body)
        if control_word:
            logger.info(
                "[Inkbox] SMS control '%s' from %s handled as protocol text",
                control_word.upper(),
                redact_phone(remote),
            )
            return web.Response(status=200, text="ok")

        self._last_inbound_modality[str(chat_id)] = "sms"

        if body.lstrip().startswith("/"):
            event = self._build_sms_text_event(
                envelope=envelope,
                text_id=text_id,
                remote=remote,
                contact=contact,
                chat_id=chat_id,
                contact_name=contact_name,
                body=body,
                timestamp=timestamp,
                text=body.strip(),
                message_type=MessageType.COMMAND,
            )
            await self._enqueue(event)
            return web.Response(status=200, text="ok")

        event = self._build_sms_text_event(
            envelope=envelope,
            text_id=text_id,
            remote=remote,
            contact=contact,
            chat_id=chat_id,
            contact_name=contact_name,
            body=body,
            timestamp=timestamp,
        )
        await self._enqueue_sms_text_event(event)
        return web.Response(status=200, text="ok")

    async def _on_text_lifecycle(self, envelope: Dict[str, Any]) -> "web.Response":
        """Log SMS delivery/status callbacks without enqueueing an agent turn."""
        event_type = str(envelope.get("event_type") or "")
        text_msg = (envelope.get("data") or {}).get("text_message") or {}
        text_id = str(text_msg.get("id") or "").strip()
        direction = str(text_msg.get("direction") or "").strip()
        remote = str(text_msg.get("remote_phone_number") or "").strip()
        status = (
            _plain_value(text_msg.get("delivery_status"))
            or _plain_value(text_msg.get("status"))
            or ""
        )
        error_code = _plain_value(text_msg.get("error") or text_msg.get("error_code"))
        logger.info(
            "[Inkbox] Text lifecycle event=%s id=%s direction=%s status=%s remote=%s error=%s",
            event_type,
            text_id,
            direction,
            status,
            redact_phone(remote),
            error_code or "",
        )
        return web.Response(status=200, text="ok")

    async def _on_incoming_call(self, envelope: Dict[str, Any]) -> "web.Response":
        """Answer the call and return the WS URL Inkbox should connect to.

        The remote-party → contact lookup is done eagerly so the WS handler
        already has the contact_id mapped when Inkbox opens the WebSocket
        moments later (avoiding a race where the first transcript fires
        before the contact is resolved).
        """
        remote = (envelope.get("remote_phone_number") or "").strip()
        contact = await self._resolve_contact_full(kind="phone", value=remote)
        contact_id = contact["id"] if contact else None
        contact_name = contact["name"] if contact and contact.get("name") else None
        # Stash the resolved identity under the call_id so the WS handler
        # can pick it up via the ``client_websocket_url`` query string.
        call_id = envelope.get("id")
        if call_id:
            self._call_ws_meta[hash(str(call_id))] = {
                "call_id": str(call_id),
                "contact_id": str(contact_id or remote),
                "contact_name": contact_name or remote,
                "contact": contact,
                "remote_phone_number": remote,
            }

        ws_url = f"wss://{self._public_host}{self._ws_path}?call_id={call_id}"
        return web.json_response({
            "action": "answer",
            "client_websocket_url": ws_url,
        })

    # ------------------------------------------------------------------
    # Inbound: WebSocket (live calls)
    # ------------------------------------------------------------------

    async def _handle_call_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        # Verify the HMAC on the upgrade BEFORE prepare(). The public tunnel
        # URL is reachable by anyone on the internet — the tunnel's TLS
        # only auths the SDK<->edge channel, not the requests flowing
        # through it. Inkbox-server signs the WS upgrade with the same
        # scheme as webhooks (sign_webhook_payload over the
        # X-Call-Context body), so the same verify_webhook works here.
        if self._require_signature:
            call_context = request.headers.get("X-Call-Context", "") or ""
            ok = verify_webhook(
                payload=call_context.encode(),
                headers=dict(request.headers),
                secret=self._signing_key,
            )
            if not ok:
                return web.Response(status=401, text="invalid signature")

        # ``WebSocketResponse`` doesn't take ``headers=`` as a constructor kwarg;
        # we mutate ``ws.headers`` before ``prepare()`` instead, which is what
        # aiohttp's ``StreamResponse`` accepts.  These two headers tell Inkbox
        # to handle STT + TTS itself and bridge text events on the WS — without
        # them Inkbox would expect raw audio frames in both directions.
        ws = web.WebSocketResponse()
        ws.headers["x-use-inkbox-text-to-speech"] = "true"
        ws.headers["x-use-inkbox-speech-to-text"] = "true"
        await ws.prepare(request)

        # Resolve call context.  Three sources, tried in order:
        #   1. webhook-mode pre-stash from ``_on_incoming_call`` (legacy)
        #   2. ``x-call-context`` header (some Inkbox versions ship it)
        #   3. ``ink.phone.calls.get(...)`` round-trip — the only reliable
        #      source when Inkbox accepts the call itself and connects the
        #      WS without forwarding caller metadata.  Without this, every
        #      call lands as ``contact=unknown`` and the agent can't tell
        #      who's on the line until it manually queries the SDK.
        call_id = request.query.get("call_id", "")
        meta = self._call_ws_meta.pop(hash(call_id), None) or {}

        if not meta:
            ctx_raw = request.headers.get("x-call-context", "") or ""
            try:
                ctx = json.loads(ctx_raw) if ctx_raw else {}
            except json.JSONDecodeError:
                ctx = {}
            call_id = call_id or str(ctx.get("call_id") or ctx.get("id") or "")
            remote = (ctx.get("remote_phone_number") or "").strip()
            # NOTE: ``ctx`` may carry a ``direction`` field but it's reported
            # from Inkbox-server perspective (always "inbound to them"), so
            # we cannot trust it here.  The SDK call record is the only
            # authoritative source — fetched below.
            direction = ""

            # Always round-trip through the SDK to learn ``direction`` (and
            # backfill ``remote_phone_number`` if the header didn't carry it).
            # Direction drives session keying below — outbound calls join the
            # contact's main session for context continuity, inbound calls
            # stay isolated under their own thread.
            if call_id and self._inkbox is not None:
                try:
                    identity = await asyncio.to_thread(
                        self._inkbox.get_identity, self._identity_handle,
                    )
                    pn_id = getattr(identity.phone_number, "id", None)
                    if pn_id:
                        # ``_calls`` is the SDK's private call resource — same
                        # accessor the legacy phone-bridge used.  Public
                        # attribute is not yet exposed on Inkbox().
                        call = await asyncio.to_thread(
                            self._inkbox._calls.get, pn_id, call_id,
                        )
                        direction = (getattr(call, "direction", "") or "").strip().lower()
                        if not remote:
                            remote = (getattr(call, "remote_phone_number", "") or "").strip()
                except Exception as exc:
                    logger.warning(
                        "[Inkbox] Call lookup failed for call_id=%s: %s", call_id, exc,
                    )

            # Header value is only a fallback if the SDK round-trip failed.
            if not direction:
                direction = (ctx.get("direction") or "").strip().lower()

            contact = (
                await self._resolve_contact_full(kind="phone", value=remote)
                if remote else None
            )
            meta = {
                "call_id": call_id,
                "contact_id": (contact["id"] if contact else (remote or call_id or "unknown")),
                "contact_name": (
                    contact["name"] if contact and contact.get("name") else (remote or "unknown")
                ),
                "contact": contact,
                "remote_phone_number": remote,
                "direction": direction or "inbound",
            }

        contact_id = meta.get("contact_id") or call_id or "unknown"
        contact_name = meta.get("contact_name") or contact_id
        direction = (meta.get("direction") or "inbound").strip().lower()

        # Direction-aware session keying:
        #   - Outbound calls (the agent placed them) collapse into the
        #     contact's main session — same session SMS/email use — so the
        #     agent inherits the conversation that decided to call.  This
        #     is what lets it answer "why are you calling me?" without any
        #     external context-token plumbing.
        #   - Inbound calls (someone dialled us) stay isolated under their
        #     own ``call:<call_id>`` thread so the caller's fresh intent
        #     isn't drowned in old SMS/email history.
        call_thread_id = None if direction == "outbound" else f"call:{call_id}"

        # Bind this WS as the active sink for the contact, and tag the
        # contact's most-recent inbound modality as ``voice`` so the gateway's
        # outbound ``send()`` path routes the agent's reply onto this WS
        # rather than falling through to the SMS/email default heuristic.
        self._active_call_ws[contact_id] = ws
        self._last_inbound_modality[str(contact_id)] = "voice"

        # Outbound-call purpose: the agent that placed the call writes a
        # context file under ``$HERMES_HOME/inkbox_call_contexts/<token>.json``
        # and includes ``?context_token=<token>`` on the WS URL.  We load it
        # here so the in-call agent — which runs in a brand-new session and
        # has zero memory of why it's calling — can be told the reason on
        # its first transcript turn.
        call_context: Dict[str, Any] = {}
        ctx_token = (request.query.get("context_token") or "").strip()
        if ctx_token:
            try:
                from hermes_cli.config import get_hermes_home
                ctx_path = get_hermes_home() / "inkbox_call_contexts" / f"{ctx_token}.json"
                if ctx_path.exists():
                    call_context = json.loads(ctx_path.read_text())
                    # Single-use: drop the file so abandoned tokens don't pile up.
                    with suppress(Exception):
                        ctx_path.unlink()
                else:
                    logger.warning(
                        "[Inkbox] Outbound-call context_token %s not found at %s",
                        ctx_token, ctx_path,
                    )
            except Exception as exc:
                logger.warning(
                    "[Inkbox] Failed to load context_token %s: %s", ctx_token, exc,
                )

        logger.info(
            "[Inkbox] Call WS open: call_id=%s contact_id=%s remote=%s "
            "direction=%s thread=%s context=%s",
            call_id, contact_id, meta.get("remote_phone_number"),
            direction, call_thread_id,
            (call_context.get("reason") or "")[:80] if call_context else "(none)",
        )

        async def _send_text_delta(text: str, *, turn_id: str) -> None:
            await ws.send_str(json.dumps(
                {"event": "text", "delta": text, "turn_id": turn_id}
            ))

        async def _send_text_done(*, turn_id: str) -> None:
            await ws.send_str(json.dumps(
                {"event": "text", "done": True, "turn_id": turn_id}
            ))

        greeting_sent = False

        async def _send_static_greeting() -> None:
            """Static opener for INBOUND calls — caller is unknown intent.

            Sent direct from the adapter without going through the agent so
            the caller hears something within ~1s of pickup.  Inbound calls
            don't have prior context worth opening on, so a generic greeting
            is fine.
            """
            contact = meta.get("contact") or {}
            first_name = ""
            if contact.get("name"):
                first_name = str(contact["name"]).split()[0]
            who = f"{first_name}" if first_name else "there"
            text = f"Hi {who}, how can I help?"
            try:
                await _send_text_delta(text, turn_id="greeting")
                await _send_text_done(turn_id="greeting")
                logger.info("[Inkbox] Sent static greeting to call_id=%s", call_id)
            except Exception as exc:
                logger.warning("[Inkbox] Failed to send greeting: %s", exc)

        async def _trigger_outbound_opening() -> None:
            """Opener for OUTBOUND calls — let the agent speak first.

            We placed this call.  The session this call lands on is the same
            one that decided to call (SMS thread / email thread for the
            contact), so the agent already has full context for *why* it's
            calling.  Enqueue a synthetic event that asks the agent to greet
            with that context in mind — its reply rides the call WS as the
            first audio the callee hears.

            Trade-off: a 1-2s pause at pickup while the agent generates the
            opener.  Worth it: the caller gets "Hey Dima, calling about the
            cats thing as you asked" instead of a generic "How can I help?"
            from a system that just dialed them.
            """
            contact_block = self._contact_marker(meta.get("contact"))
            tagged = (
                f"[inkbox:voice_call call_id={call_id} | {contact_block}]\n"
                "[outbound_call_connected] You just placed this call. The "
                "callee picked up. Greet them by name and open with the "
                "reason for the call, drawing from the conversation that "
                "decided to place it (above in this thread). Keep it to one "
                "short sentence; the rest of the conversation will follow."
            )
            source = self.build_source(
                chat_id=str(contact_id),
                chat_name=contact_name,
                chat_type="dm",
                user_id=str(contact_id),
                user_name=contact_name,
                thread_id=call_thread_id,
                chat_topic="voice_call",
                message_id=f"call:{call_id}:opening",
            )
            event = MessageEvent(
                text=tagged,
                message_type=MessageType.TEXT,
                source=source,
                raw_message={"synthetic": "outbound_call_opening"},
                message_id=f"call:{call_id}:opening",
                auto_skill="inkbox",
            )
            try:
                await self._enqueue(event)
                logger.info(
                    "[Inkbox] Triggered outbound opener for call_id=%s", call_id,
                )
            except Exception as exc:
                logger.warning(
                    "[Inkbox] Failed to enqueue outbound opener: %s", exc,
                )

        first_transcript_seen = False

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                ev = payload.get("event")
                if ev == "start" and not greeting_sent:
                    greeting_sent = True
                    if direction == "outbound":
                        asyncio.create_task(_trigger_outbound_opening())
                    else:
                        asyncio.create_task(_send_static_greeting())
                    continue
                if ev == "transcript" and payload.get("is_final"):
                    text = (payload.get("text") or "").strip()
                    if not text:
                        continue
                    source = self.build_source(
                        chat_id=str(contact_id),
                        chat_name=contact_name,
                        chat_type="dm",
                        user_id=str(contact_id),
                        user_name=contact_name,
                        thread_id=call_thread_id,
                        chat_topic="voice_call",
                        message_id=payload.get("turn_id"),
                    )
                    contact_block = self._contact_marker(meta.get("contact"))

                    # On the FIRST transcript only, prepend the call-purpose
                    # block (if any) so the in-call agent — which has no
                    # memory of why it's calling — has authoritative context.
                    purpose_block = ""
                    if call_context and not first_transcript_seen:
                        reason = (call_context.get("reason") or "").strip()
                        scheduled_by = (call_context.get("scheduled_by") or "").strip()
                        prior = (call_context.get("conversation_summary") or "").strip()
                        lines = ["[outbound_call_context]"]
                        if reason:
                            lines.append(f"reason: {reason}")
                        if scheduled_by:
                            lines.append(f"scheduled_by: {scheduled_by}")
                        if prior:
                            lines.append(f"prior_conversation: {prior}")
                        lines.append("[/outbound_call_context]")
                        purpose_block = "\n".join(lines) + "\n"
                    first_transcript_seen = True

                    tagged = (
                        f"[inkbox:voice_call call_id={call_id} | {contact_block}]\n"
                        f"{purpose_block}{text}"
                    )
                    event = MessageEvent(
                        text=tagged,
                        message_type=MessageType.TEXT,
                        source=source,
                        raw_message=payload,
                        message_id=f"call:{call_id}:{payload.get('turn_id') or ''}",
                        auto_skill="inkbox",
                    )
                    await self._enqueue(event)
                elif ev == "stop":
                    break
                # 'barge_in' is informational here — proper interruption
                # would require cancelling the in-flight gateway turn, which
                # crosses module boundaries we don't yet expose.
        finally:
            self._active_call_ws.pop(contact_id, None)
            # Clear the voice tag so a follow-up SMS/email from this contact
            # doesn't get mis-routed to a closed call socket.
            if self._last_inbound_modality.get(str(contact_id)) == "voice":
                self._last_inbound_modality.pop(str(contact_id), None)
            # Stamp the close time so send() can drop in-flight voice replies
            # that finish generating after the WS is gone, instead of letting
            # them leak to email/SMS via the default mode heuristic.
            self._voice_recently_closed[str(contact_id)] = time.time()
            with suppress(Exception):
                await ws.close()
            logger.info("[Inkbox] Call WS closed: call_id=%s", call_id)

            # Post-call reflection: enqueue a synthetic [call_ended] turn so
            # the agent has a chance to do follow-up work (send promised
            # emails, schedule callbacks, save notes, update memory).  The
            # agent's text reply will be suppressed by the voice-grace guard
            # in send() — only its TOOL CALLS produce side effects.  If the
            # agent has nothing to do it can answer "[SILENT]" and the
            # cron-style suppression delivers nothing.
            try:
                contact_block = self._contact_marker(meta.get("contact"))
                tagged = (
                    f"[inkbox:voice_call call_id={call_id} | {contact_block}]\n"
                    "[call_ended] The call has ended. Reflect on what just "
                    "happened and decide if any follow-up actions are "
                    "needed:\n"
                    "  - if you committed to anything during the call (send "
                    "an email, schedule a callback, text a contact, save a "
                    "note, update a contact record), perform that now via "
                    "tool calls — execute_code/terminal for SDK actions, "
                    "cronjob create deliver=local for delayed work, memory/"
                    "send_message for the obvious cases.\n"
                    "  - if there's nothing to do, reply with exactly "
                    "[SILENT] and no other text.\n"
                    "Note: any plain-text reply you produce here will be "
                    "suppressed (the caller hung up — they don't want a "
                    "trailing TTS or email containing your thoughts). "
                    "Side effects must come from tool calls."
                )
                source = self.build_source(
                    chat_id=str(contact_id),
                    chat_name=contact_name,
                    chat_type="dm",
                    user_id=str(contact_id),
                    user_name=contact_name,
                    thread_id=call_thread_id,
                    chat_topic="voice_call",
                    message_id=f"call:{call_id}:ended",
                )
                event = MessageEvent(
                    text=tagged,
                    message_type=MessageType.TEXT,
                    source=source,
                    raw_message={"synthetic": "call_ended"},
                    message_id=f"call:{call_id}:ended",
                    auto_skill="inkbox",
                )
                await self._enqueue(event)
                logger.info(
                    "[Inkbox] Enqueued [call_ended] reflection for call_id=%s",
                    call_id,
                )
            except Exception as exc:
                logger.warning(
                    "[Inkbox] Failed to enqueue call_ended event: %s", exc,
                )
        return ws

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _resolve_contact(
        self, *, kind: str, value: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Thin wrapper that returns just ``(contact_id, display_name)``.

        Kept for the call-sites that only need the chat-routing pair.  New
        code that wants emails / phones / company / notes should use
        :meth:`_resolve_contact_full` instead.
        """
        details = await self._resolve_contact_full(kind=kind, value=value)
        if details is None:
            return (None, None)
        return (details.get("id"), details.get("name"))

    async def _resolve_contact_full(
        self, *, kind: str, value: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a serialisable summary of the Inkbox Contact matched by *value*.

        Shape::

            {
                "id":       "<uuid>",
                "name":     "Dima Vremenko",
                "emails":   ["dima@vectorly.app", ...],   # primary first
                "phones":   ["+15167251294", ...],         # primary first
                "company":  "Inkbox",
                "job_title": "Cofounder",
                "notes":    "...",
            }

        ``None`` when the lookup returns 0 or >1 matches.  Cached for
        ``CONTACT_CACHE_TTL_SECONDS`` (positive *and* negative results).
        """
        if not value:
            return None
        cache_key = (kind, value.lower())
        now = time.time()
        cached = self._contact_cache.get(cache_key)
        if cached and cached[1] > now:
            return cached[0]

        if self._inkbox is None:
            return None

        kwargs = {kind: value}
        try:
            contacts = await asyncio.to_thread(self._inkbox.contacts.lookup, **kwargs)
        except Exception as exc:
            logger.debug("[Inkbox] contacts.lookup(%s=%s) failed: %s", kind, value, exc)
            self._contact_cache[cache_key] = (None, now + CONTACT_CACHE_TTL_SECONDS)
            return None

        if len(contacts) != 1:
            self._contact_cache[cache_key] = (None, now + CONTACT_CACHE_TTL_SECONDS)
            return None

        contact = contacts[0]
        emails_raw = list(getattr(contact, "emails", None) or [])
        phones_raw = list(getattr(contact, "phones", None) or [])
        emails_raw.sort(key=lambda e: not getattr(e, "is_primary", False))
        phones_raw.sort(key=lambda p: not getattr(p, "is_primary", False))
        details: Dict[str, Any] = {
            "id": str(getattr(contact, "id", "")),
            "name": (
                getattr(contact, "preferred_name", None)
                or getattr(contact, "given_name", None)
                or None
            ),
            "emails": [getattr(e, "value", "") for e in emails_raw if getattr(e, "value", "")],
            "phones": [getattr(p, "value", "") for p in phones_raw if getattr(p, "value", "")],
            "company": getattr(contact, "company_name", None) or None,
            "job_title": getattr(contact, "job_title", None) or None,
            "notes": ((getattr(contact, "notes", None) or "")[:200].strip() or None),
        }
        self._contact_cache[cache_key] = (details, now + CONTACT_CACHE_TTL_SECONDS)
        return details

    @staticmethod
    def _contact_marker(details: Optional[Dict[str, Any]]) -> str:
        """Render a one-line contact summary for embedding in MessageEvent text."""
        if not details:
            return "contact=unknown_in_inkbox"
        parts = [f"contact_id={details['id']}"]
        if details.get("name"):
            parts.append(f"contact_name={details['name']!r}")
        if details.get("company"):
            parts.append(f"contact_company={details['company']!r}")
        if details.get("emails"):
            parts.append(f"contact_emails={details['emails']}")
        if details.get("phones"):
            parts.append(f"contact_phones={details['phones']}")
        return " ".join(parts)

    def _lookup_contact_email(self, contact_id: str) -> Optional[str]:
        """Fetch the primary email address for a contact (sync helper)."""
        if self._inkbox is None:
            return None
        try:
            contact = self._inkbox.contacts.get(contact_id)
        except Exception:
            return None
        emails = getattr(contact, "emails", None) or []
        primary = next((e for e in emails if getattr(e, "is_primary", False)), None)
        chosen = primary or (emails[0] if emails else None)
        return getattr(chosen, "value", None) if chosen else None

    def _lookup_contact_phone(self, contact_id: str) -> Optional[str]:
        """Fetch the primary phone number (E.164) for a contact (sync helper)."""
        if self._inkbox is None:
            return None
        try:
            contact = self._inkbox.contacts.get(contact_id)
        except Exception:
            return None
        phones = getattr(contact, "phones", None) or []
        primary = next((p for p in phones if getattr(p, "is_primary", False)), None)
        chosen = primary or (phones[0] if phones else None)
        return getattr(chosen, "value", None) if chosen else None

    async def _enqueue(self, event: MessageEvent) -> None:
        """Dispatch an inbound event to the gateway runner as a background task."""
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


# ---------------------------------------------------------------------------
# Standalone send helper (for cron + send_message tool outside the gateway)
# ---------------------------------------------------------------------------


async def send_inkbox_direct(
    extra: Dict[str, Any],
    chat_id: str,
    message: str,
    *,
    mode: Optional[str] = None,
    subject: Optional[str] = None,
    thread_id: Optional[str] = None,  # noqa: ARG001 — reserved for future email-thread replies
) -> Dict[str, Any]:
    """One-shot send via the Inkbox SDK — no aiohttp server, no WS.

    Mirrors the ``_send_*_direct`` helpers used by the other platforms for
    cron delivery and ``send_message`` calls outside an active gateway.
    """
    if not INKBOX_AVAILABLE:
        return {
            "error": "Inkbox SDK not installed. Run: pip install inkbox",
        }

    api_key = (extra.get("api_key") or os.getenv("INKBOX_API_KEY") or "").strip()
    if not api_key:
        return {"error": "INKBOX_API_KEY not set"}
    handle = (extra.get("identity") or os.getenv("INKBOX_IDENTITY") or "").strip()
    if not handle:
        return {"error": "INKBOX_IDENTITY not set"}
    base_url = (
        extra.get("base_url") or os.getenv("INKBOX_BASE_URL") or INKBOX_BASE_URL_DEFAULT
    )

    def _do_send() -> Dict[str, Any]:
        with Inkbox(api_key=api_key, base_url=base_url) as client:
            identity = client.get_identity(handle)
            chosen_mode = (mode or "").lower().strip()
            if not chosen_mode:
                chosen_mode = "sms" if str(chat_id).startswith("+") else "email"

            if chosen_mode == "sms":
                target = chat_id
                if not str(target).startswith("+"):
                    contact = client.contacts.get(chat_id)
                    phones = getattr(contact, "phones", None) or []
                    primary = next((p for p in phones if getattr(p, "is_primary", False)), None)
                    chosen = primary or (phones[0] if phones else None)
                    target = getattr(chosen, "value", None) if chosen else None
                if not target:
                    return {"error": f"No phone for contact {chat_id}"}
                try:
                    msg = identity.send_text(to=target, text=message[:SMS_MAX_LENGTH])
                except Exception as exc:
                    logger.error(
                        "[Inkbox] Direct SMS send failed to %s",
                        redact_phone(str(target)),
                    )
                    return _sms_send_failure_dict(exc)
                return {
                    "success": True,
                    "platform": "inkbox",
                    "chat_id": chat_id,
                    "message_id": str(getattr(msg, "id", "")),
                    "mode": "sms",
                    "delivery_status": _plain_value(
                        getattr(msg, "delivery_status", None),
                    ),
                }

            if chosen_mode == "email":
                target = str(chat_id).strip()
                if "@" not in target:
                    # chat_id is a contact UUID — look up the primary email.
                    contact = client.contacts.get(chat_id)
                    emails = getattr(contact, "emails", None) or []
                    primary = next((e for e in emails if getattr(e, "is_primary", False)), None)
                    chosen = primary or (emails[0] if emails else None)
                    target = getattr(chosen, "value", None) if chosen else None
                if not target:
                    return {"error": f"No email for contact {chat_id}"}
                msg = identity.send_email(
                    to=[target],
                    subject=subject or "(no subject)",
                    body_text=message,
                )
                return {
                    "success": True,
                    "platform": "inkbox",
                    "chat_id": chat_id,
                    "message_id": str(getattr(msg, "id", "")),
                    "mode": "email",
                }

            return {"error": f"Unknown Inkbox send mode: {chosen_mode!r}"}

    try:
        return await asyncio.to_thread(_do_send)
    except Exception as exc:
        return {"error": f"Inkbox send failed: {exc}"}
