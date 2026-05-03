"""Inkbox platform adapter.

Inkbox (https://inkbox.ai) is API-first communication infrastructure that
gives an AI agent a stable email address, phone number, and persistent
contact list scoped to one *agent identity*. This adapter routes the
three Inkbox modalities — inbound email, inbound SMS, and live voice
calls — into a single contact-keyed Hermes session per remote party.

Architecture
------------
On ``connect()`` the adapter:
  1. Opens an ngrok HTTPS tunnel to its local listen port (for local
     development; production deployments can disable this and supply
     ``INKBOX_PUBLIC_URL`` directly).
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
    live call turn → chat_id=contact_id, thread_id=f"call:{call_id}"

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
import socket as _socket
import time
from contextlib import suppress
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
    from pyngrok import conf as _ngrok_conf
    from pyngrok import ngrok as _ngrok

    PYNGROK_AVAILABLE = True
except ImportError:
    _ngrok = None  # type: ignore[assignment]
    _ngrok_conf = None  # type: ignore[assignment]
    PYNGROK_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_BASE_URL = "https://inkbox.ai"
DEFAULT_WEBHOOK_PATH = "/webhook"
DEFAULT_WS_PATH = "/phone/media/ws"
CONTACT_CACHE_TTL_SECONDS = 300
WEBHOOK_DEDUP_TTL_SECONDS = 300
SMS_MAX_LENGTH = 1600  # Inkbox SMS hard cap

# Hermes emits a few classes of admin/system notice via adapter.send() —
# session-reset banners ("◐ ..."), runtime info blocks ("◆ Model: ..."),
# the home-channel prompt ("📬 No home channel..."), update/restart notes
# ("🔄 ..."), check/x-mark status pings ("✓ ..." / "✗ ..."), and the
# warning prefix ("⚠️ ...").  These are CLI/terminal-style chatter that
# leaks into the user's actual mailbox or SMS thread on Inkbox.  Drop
# them at adapter.send() so they never get delivered as real messages.
_ADMIN_NOTICE_PREFIXES: Tuple[str, ...] = (
    "◐", "◆", "📬", "🔄", "✓", "✗", "⚠️", "⚠", "⚡", "💡",
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


def check_inkbox_requirements() -> bool:
    """Return True iff the Python ``inkbox`` SDK and aiohttp are importable.

    pyngrok is optional — if unavailable, the adapter requires
    ``INKBOX_PUBLIC_URL`` to be set so it knows where webhooks will arrive
    (e.g. behind a reverse proxy or a hosted tunnel).
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
            extra.get("base_url") or os.getenv("INKBOX_BASE_URL") or DEFAULT_BASE_URL
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
        self._ngrok_token = (
            extra.get("ngrok_authtoken") or os.getenv("NGROK_AUTHTOKEN") or ""
        ).strip()
        self._require_signature = str(
            extra.get("require_signature")
            or os.getenv("INKBOX_REQUIRE_SIGNATURE", "true")
        ).lower() not in ("false", "0", "no")

        # Live state.
        self._inkbox: Optional[Any] = None
        self._public_url: Optional[str] = None
        self._public_host: Optional[str] = None
        self._app: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._site: Optional[Any] = None
        self._ngrok_tunnel: Any = None
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
                "Generate a signing key in console.inkbox.ai or set "
                "INKBOX_REQUIRE_SIGNATURE=false for local-only testing.",
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

        # Resolve the public URL: explicit override wins, else open an ngrok tunnel.
        if self._public_url_override:
            self._public_url = self._public_url_override.rstrip("/")
        elif PYNGROK_AVAILABLE and self._ngrok_token:
            try:
                await asyncio.to_thread(self._open_ngrok_tunnel)
            except Exception as exc:
                logger.error("[Inkbox] Failed to open ngrok tunnel: %s", exc)
                self._release_platform_lock()
                return False
        else:
            logger.error(
                "[Inkbox] No public URL configured. Set INKBOX_PUBLIC_URL or "
                "install pyngrok and set NGROK_AUTHTOKEN.",
            )
            self._release_platform_lock()
            return False

        self._public_host = urlparse(self._public_url).netloc

        # Start the aiohttp server first so the upstream PATCH calls can verify
        # the URLs are reachable.
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
        if self._ngrok_tunnel is not None:
            with suppress(Exception):
                _ngrok.disconnect(self._ngrok_tunnel.public_url)
            with suppress(Exception):
                _ngrok.kill()
            self._ngrok_tunnel = None
        if self._inkbox is not None:
            with suppress(Exception):
                self._inkbox.close()
            self._inkbox = None

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------

    def _open_ngrok_tunnel(self) -> None:
        """Open an HTTPS ngrok tunnel and populate ``self._public_url``."""
        runtime_dir = os.environ.get(
            "INKBOX_NGROK_DIR", "/tmp/hermes-inkbox-ngrok",
        )
        os.makedirs(runtime_dir, exist_ok=True)
        cfg = _ngrok_conf.PyngrokConfig(
            auth_token=self._ngrok_token,
            config_path=os.path.join(runtime_dir, "ngrok.yml"),
            ngrok_path=os.path.join(runtime_dir, "ngrok"),
        )
        tunnel = _ngrok.connect(
            addr=self._port, proto="http", bind_tls=True, pyngrok_config=cfg,
        )
        url = tunnel.public_url
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        self._ngrok_tunnel = tunnel
        self._public_url = url.rstrip("/")

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
            state_path.write_text(json.dumps(state, indent=2) + "\n")
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

        meta = metadata or {}
        mode = (meta.get("mode") or "").lower().strip()

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
                return SendResult(success=True, message_id=str(getattr(msg, "id", "")))
            except Exception as exc:
                return SendResult(success=False, error=f"send_text failed: {exc}")

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
            auto_skill="inkbox-python",
        )
        await self._enqueue(event)
        return web.Response(status=200, text="ok")

    async def _on_text_received(self, envelope: Dict[str, Any]) -> "web.Response":
        text_msg = (envelope.get("data") or {}).get("text_message") or {}
        remote = (text_msg.get("remote_phone_number") or "").strip()
        if not remote:
            return web.Response(status=200, text="ok")

        contact = await self._resolve_contact_full(kind="phone", value=remote)
        chat_id = (contact["id"] if contact else remote)
        contact_name = contact["name"] if contact and contact.get("name") else None

        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=contact_name or remote,
            chat_type="dm",
            user_id=str(chat_id),
            user_name=contact_name or remote,
            message_id=text_msg.get("id"),
        )
        body = text_msg.get("text") or ""
        contact_block = self._contact_marker(contact)
        tagged = f"[inkbox:sms from={remote} | {contact_block}]\n{body}"
        self._last_inbound_modality[str(chat_id)] = "sms"
        event = MessageEvent(
            text=tagged,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=envelope,
            message_id=str(text_msg.get("id") or ""),
            auto_skill="inkbox-python",
        )
        await self._enqueue(event)
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

            # SDK fallback when the header didn't carry the remote number.
            if not remote and call_id and self._inkbox is not None:
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
                        remote = (getattr(call, "remote_phone_number", "") or "").strip()
                except Exception as exc:
                    logger.warning(
                        "[Inkbox] Call lookup failed for call_id=%s: %s", call_id, exc,
                    )

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
            }

        contact_id = meta.get("contact_id") or call_id or "unknown"
        contact_name = meta.get("contact_name") or contact_id
        # Bind this WS as the active sink for the contact, and tag the
        # contact's most-recent inbound modality as ``voice`` so the gateway's
        # outbound ``send()`` path routes the agent's reply onto this WS
        # rather than falling through to the SMS/email default heuristic.
        self._active_call_ws[contact_id] = ws
        self._last_inbound_modality[str(contact_id)] = "voice"

        logger.info(
            "[Inkbox] Call WS open: call_id=%s contact_id=%s remote=%s",
            call_id, contact_id, meta.get("remote_phone_number"),
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

        async def _send_greeting() -> None:
            """Speak an opening line so the caller hears something immediately.

            Sent direct from the adapter without going through the agent — the
            agent's first turn will be the caller's first transcript.  Mirrors
            what the legacy phone-bridge did so call setup latency stays low.
            """
            contact = meta.get("contact") or {}
            first_name = ""
            if contact.get("name"):
                first_name = str(contact["name"]).split()[0]
            who = f"{first_name}" if first_name else "there"
            text = f"Hi {who}, this is your Inkbox on-call agent. How can I help?"
            try:
                await _send_text_delta(text, turn_id="greeting")
                await _send_text_done(turn_id="greeting")
                logger.info("[Inkbox] Sent greeting to call_id=%s", call_id)
            except Exception as exc:
                logger.warning("[Inkbox] Failed to send greeting: %s", exc)

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
                    asyncio.create_task(_send_greeting())
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
                        thread_id=f"call:{call_id}",
                        chat_topic="voice_call",
                        message_id=payload.get("turn_id"),
                    )
                    contact_block = self._contact_marker(meta.get("contact"))
                    tagged = (
                        f"[inkbox:voice_call call_id={call_id} | {contact_block}]\n{text}"
                    )
                    event = MessageEvent(
                        text=tagged,
                        message_type=MessageType.TEXT,
                        source=source,
                        raw_message=payload,
                        message_id=f"call:{call_id}:{payload.get('turn_id') or ''}",
                        auto_skill="inkbox-python",
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
            with suppress(Exception):
                await ws.close()
            logger.info("[Inkbox] Call WS closed: call_id=%s", call_id)
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
        extra.get("base_url") or os.getenv("INKBOX_BASE_URL") or DEFAULT_BASE_URL
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
                msg = identity.send_text(to=target, text=message[:SMS_MAX_LENGTH])
                return {
                    "success": True,
                    "platform": "inkbox",
                    "chat_id": chat_id,
                    "message_id": str(getattr(msg, "id", "")),
                    "mode": "sms",
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
