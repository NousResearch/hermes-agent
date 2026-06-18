"""Gateway-side inbound delivery receiver. EXPERIMENTAL.

The connector delivers normalized inbound events to a tenant's gateway over a
**signed HTTP POST** (connector ``src/relay/httpGatewayDelivery.ts``), NOT over
the gateway's outbound ``/relay`` WebSocket: the connector instance that owns a
platform socket is generally not the instance a given gateway dialed out to, so
inbound is delivered to a tenant ENDPOINT (which may load-balance across gateway
instances). Each delivery is HMAC-signed with the per-tenant **delivery key**
(``gateway/relay/auth.py``); this receiver verifies the signature over the EXACT
raw request bytes before accepting the event.

Two routes (mirroring the connector's two POST targets):
  POST {base}            {"type":"message",  "event": <MessageEvent>, ...}
  POST {base}/interrupt  {"type":"interrupt","session_key": ..., "reason"?}

The receiver:
  1. reads the RAW body bytes (never a reparsed/re-serialized form — the HMAC is
     over the literal bytes the connector signed),
  2. verifies ``x-relay-signature`` / ``x-relay-timestamp`` against the delivery
     key verify list (primary + secondary during rotation), within the replay
     window — rejects 401 on any failure,
  3. parses the JSON and dispatches: a ``message`` to the inbound handler (the
     RelayAdapter's ``handle_message`` via the transport's normal path), an
     ``interrupt`` to the interrupt handler.

EXPERIMENTAL: the transport protocol may change without a deprecation cycle
until ≥2 Class-1 platforms validate it. See docs/relay-connector-contract.md.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence

from gateway.platforms.base import MessageEvent
from gateway.relay.auth import (
    DELIVERY_SIG_HEADER,
    DELIVERY_TS_HEADER,
    verify_delivery_signature,
)

logger = logging.getLogger(__name__)

# Callbacks the receiver dispatches verified deliveries to.
InboundMessageHandler = Callable[[MessageEvent], Awaitable[None]]
InboundInterruptHandler = Callable[[str, str], Awaitable[None]]

try:  # lazy/optional dep — mirrors the other HTTP-receiving adapters
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised only when the extra is absent
    web = None  # type: ignore[assignment]

AIOHTTP_AVAILABLE = web is not None


def _event_from_wire(raw: dict) -> MessageEvent:
    """Rebuild a MessageEvent from the connector's normalized inbound payload.

    Identical mapping to the WS transport's ``_event_from_wire`` (the wire shape
    is the same; only the transport differs). Kept here so the HTTP receiver has
    no import dependency on the WS transport module.
    """
    from gateway.config import Platform
    from gateway.platforms.base import MessageType
    from gateway.session import SessionSource

    src = raw.get("source", {}) or {}
    platform = src.get("platform", "relay")
    try:
        platform_enum = Platform(platform)
    except ValueError:
        platform_enum = Platform.RELAY

    source = SessionSource(
        platform=platform_enum,
        chat_id=src.get("chat_id", ""),
        chat_type=src.get("chat_type", "dm"),
        chat_name=src.get("chat_name"),
        user_id=src.get("user_id"),
        user_name=src.get("user_name"),
        thread_id=src.get("thread_id"),
        chat_topic=src.get("chat_topic"),
        user_id_alt=src.get("user_id_alt"),
        chat_id_alt=src.get("chat_id_alt"),
        guild_id=src.get("guild_id"),
        parent_chat_id=src.get("parent_chat_id"),
        message_id=src.get("message_id"),
    )
    try:
        msg_type = MessageType(raw.get("message_type", "text"))
    except ValueError:
        msg_type = MessageType.TEXT

    return MessageEvent(
        text=raw.get("text", ""),
        message_type=msg_type,
        source=source,
        message_id=raw.get("message_id"),
        reply_to_message_id=raw.get("reply_to_message_id"),
        media_urls=raw.get("media_urls") or [],
    )


class InboundDeliveryReceiver:
    """Verifies + dispatches signed connector→gateway inbound deliveries.

    Transport-agnostic core: ``handle_raw`` takes the raw body bytes + headers +
    which route was hit and returns ``(status, body)``. The aiohttp wiring
    (``build_app`` / ``serve``) is a thin shell so the verify+dispatch logic is
    unit-testable without a live socket.
    """

    def __init__(
        self,
        *,
        delivery_key_verify_list: Callable[[], Sequence[str]],
        on_message: InboundMessageHandler,
        on_interrupt: Optional[InboundInterruptHandler] = None,
        max_skew_seconds: int = 300,
    ) -> None:
        # A callable (not a static list) so a rotated delivery key is picked up
        # without rebuilding the receiver — mirrors the connector's verify list.
        self._verify_list = delivery_key_verify_list
        self._on_message = on_message
        self._on_interrupt = on_interrupt
        self._max_skew_seconds = max_skew_seconds

    async def handle_raw(
        self, *, raw_body: bytes, timestamp: Optional[str], signature: Optional[str], is_interrupt: bool
    ) -> tuple[int, dict]:
        """Verify the signature over ``raw_body`` and dispatch. Returns (status, json).

        401 on a missing/invalid/expired signature (never dispatches unverified).
        400 on malformed JSON. 200 on a verified, dispatched delivery.
        """
        verify_keys = list(self._verify_list() or [])
        if not verify_keys:
            # No delivery key provisioned -> we cannot verify -> reject. A gateway
            # that hasn't enrolled must not accept inbound (fail closed).
            logger.warning("relay inbound: no delivery key configured; rejecting")
            return 401, {"error": "no delivery key configured"}

        # Verify over the EXACT raw bytes the connector signed. Decode to text
        # with the same UTF-8 the connector's JSON.stringify produced; a single
        # differing byte breaks the HMAC (raw-body-preservation discipline).
        body_text = raw_body.decode("utf-8", errors="strict")
        if not verify_delivery_signature(
            body_text, timestamp, signature, verify_keys, self._max_skew_seconds
        ):
            return 401, {"error": "invalid delivery signature"}

        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            return 400, {"error": "invalid JSON body"}

        if is_interrupt or payload.get("type") == "interrupt":
            session_key = str(payload.get("session_key", ""))
            chat_id = str(payload.get("chat_id", "") or payload.get("reason", "") or "")
            if self._on_interrupt is not None and session_key:
                await self._on_interrupt(session_key, chat_id)
            return 200, {"ok": True}

        # Default: a normalized inbound message event.
        event_raw = payload.get("event")
        if not isinstance(event_raw, dict):
            return 400, {"error": "missing event"}
        event = _event_from_wire(event_raw)
        await self._on_message(event)
        return 200, {"ok": True}

    # ── aiohttp wiring (thin shell over handle_raw) ──────────────────────
    def build_app(self) -> Any:
        """Build an aiohttp Application exposing the delivery + interrupt routes."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "InboundDeliveryReceiver requires the 'aiohttp' package "
                "(install the messaging extra)."
            )

        async def _deliver(request: Any) -> Any:
            return await self._respond(request, is_interrupt=False)

        async def _interrupt(request: Any) -> Any:
            return await self._respond(request, is_interrupt=True)

        app = web.Application()
        app.router.add_get("/healthz", lambda _: web.Response(text="ok"))
        app.router.add_post("/", _deliver)
        app.router.add_post("/interrupt", _interrupt)
        return app

    async def _respond(self, request: Any, *, is_interrupt: bool) -> Any:
        # Read the RAW bytes — do NOT use request.json() (it reparses and we'd
        # verify over a re-serialized form, breaking the HMAC).
        raw_body = await request.read()
        status, body = await self.handle_raw(
            raw_body=raw_body,
            timestamp=request.headers.get(DELIVERY_TS_HEADER),
            signature=request.headers.get(DELIVERY_SIG_HEADER),
            is_interrupt=is_interrupt,
        )
        return web.json_response(body, status=status)
