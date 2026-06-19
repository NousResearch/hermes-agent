"""Production WebSocket RelayTransport — the gateway's live link to the connector.

The gateway dials OUT to the connector's relay endpoint over a WebSocket and
speaks the newline-delimited JSON frame protocol defined in the connector repo
(``gateway-gateway`` ``src/relay/protocol.ts``) and mirrored in
``docs/relay-connector-contract.md``:

  gateway -> connector : hello, outbound, interrupt
  connector -> gateway : descriptor, inbound, outbound_result, interrupt_inbound

Frames:
  hello            {type, platform, botId}
  descriptor       {type, descriptor}                       (handshake reply)
  inbound          {type, event, bufferId?}                 (a normalized MessageEvent)
  outbound         {type, requestId, action}                (send/edit/typing/follow_up)
  outbound_result  {type, requestId, result}
  interrupt        {type, session_key, reason?}             (gateway egresses /stop)
  interrupt_inbound{type, session_key, chat_id}             (connector -> owning gateway)

This is the concrete transport behind the ``RelayTransport`` Protocol; the
``RelayAdapter`` delegates all wire I/O to it. Outbound calls block on a
per-request future keyed by ``requestId`` until the matching ``outbound_result``
arrives. A background reader task pumps inbound frames to the registered handler
and resolves pending outbound futures.

EXPERIMENTAL: the frame schema may change without a deprecation cycle until at
least two Class-1 platforms validate it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import InboundHandler

logger = logging.getLogger(__name__)

try:  # lazy/optional dep — mirrors gateway/platforms/feishu.py
    import websockets
except ImportError:  # pragma: no cover - exercised only when the extra is absent
    websockets = None  # type: ignore[assignment]

WEBSOCKETS_AVAILABLE = websockets is not None

# How long to wait for the handshake descriptor and for each outbound result.
_HANDSHAKE_TIMEOUT_S = 30.0
_OUTBOUND_TIMEOUT_S = 30.0


def _ws_dial_url(url: str) -> str:
    """Normalize a connector URL to the ``ws(s)://…/relay`` dial target.

    The relay URL is configured once (``GATEWAY_RELAY_URL`` / ``gateway.relay_url``)
    as the connector's BASE URL (e.g. ``https://connector.example``) and shared by
    both the provision POST (which needs ``http(s)://…/relay/provision`` — see
    ``_provision_url``) and the WS dial (which needs ``ws(s)://…/relay``, the path
    the connector mounts its ``WebSocketServer`` on). Two normalizations, both
    load-bearing:

      - scheme: ``https -> wss``, ``http -> ws`` (``websockets.connect`` raises
        "scheme isn't ws or wss" on an http(s) URL).
      - path: ensure it ends in ``/relay`` (the connector returns HTTP 400 on an
        upgrade to any other path, since the WS server is mounted at ``/relay``).

    Idempotent: an already-``ws(s)://…/relay`` URL is returned unchanged, so a URL
    configured WITH the scheme and/or ``/relay`` still works.
    """
    raw = (url or "").strip()
    if raw.startswith("https://"):
        raw = "wss://" + raw[len("https://"):]
    elif raw.startswith("http://"):
        raw = "ws://" + raw[len("http://"):]
    raw = raw.rstrip("/")
    if not raw.endswith("/relay"):
        raw = f"{raw}/relay"
    return raw


def _event_from_wire(raw: Dict[str, Any]) -> MessageEvent:
    """Rebuild a MessageEvent from the connector's normalized inbound payload.

    The connector emits SessionSource as the snake_case wire form (§3); map it
    back onto the gateway dataclasses. Unknown message types fall back to TEXT.
    """
    src = raw.get("source", {}) or {}
    from gateway.config import Platform

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


class WebSocketRelayTransport:
    """RelayTransport over a WebSocket connection the gateway dials to the connector."""

    def __init__(
        self,
        url: str,
        platform: str,
        bot_id: str,
        *,
        connect_timeout_s: float = _HANDSHAKE_TIMEOUT_S,
        outbound_timeout_s: float = _OUTBOUND_TIMEOUT_S,
        gateway_id: Optional[str] = None,
        upgrade_secret: Optional[str] = None,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "WebSocketRelayTransport requires the 'websockets' package "
                "(install the messaging extra)."
            )
        self._url = _ws_dial_url(url)
        self._platform = platform
        self._bot_id = bot_id
        self._connect_timeout_s = connect_timeout_s
        self._outbound_timeout_s = outbound_timeout_s
        # Connection auth (Phase 2): when a per-gateway secret is configured the
        # gateway presents an HMAC bearer on the WS upgrade so the connector can
        # authenticate it (reject 4401 otherwise). gateway_id identifies the
        # enrolled instance — the connector peeks it to index its secret verify
        # list, then verifies the signature. Absent -> unauthenticated upgrade
        # (dev/test, or a connector that doesn't enforce auth).
        self._gateway_id = gateway_id
        self._upgrade_secret = upgrade_secret

        self._ws: Any = None
        self._reader: Optional[asyncio.Task[None]] = None
        self._inbound: Optional[InboundHandler] = None
        self._descriptor: Optional[CapabilityDescriptor] = None
        self._descriptor_ready: asyncio.Future[CapabilityDescriptor] | None = None
        # requestId -> future awaiting the matching outbound_result.
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._closing = False

    # ── lifecycle ────────────────────────────────────────────────────────
    async def connect(self) -> bool:
        loop = asyncio.get_running_loop()
        self._descriptor_ready = loop.create_future()
        headers = self._upgrade_headers()
        if headers:
            self._ws = await websockets.connect(self._url, additional_headers=headers)  # type: ignore[union-attr]
        else:
            self._ws = await websockets.connect(self._url)  # type: ignore[union-attr]
        self._reader = asyncio.create_task(self._read_loop(), name="relay-ws-reader")
        # Send hello; the descriptor arrives via the reader and resolves handshake().
        await self._send({"type": "hello", "platform": self._platform, "botId": self._bot_id})
        return True

    def _upgrade_headers(self) -> Dict[str, str]:
        """Auth headers for the WS upgrade, or {} when no secret is configured.

        Presents ``Authorization: Bearer *** where the token is a signed
        bearer built with the per-gateway secret (``gateway/relay/auth.py``
        ``make_upgrade_token``), keyed by ``gateway_id`` so the connector can
        index its verify list. The connector rejects the upgrade (close 4401)
        when this is missing/invalid/revoked; an unauthenticated connector
        ignores it.
        """
        if not (self._upgrade_secret and self._gateway_id):
            return {}
        from gateway.relay.auth import make_upgrade_token

        token = make_upgrade_token(self._gateway_id, self._upgrade_secret)
        return {"Authorization": f"Bearer {token}"}

    async def disconnect(self) -> None:
        self._closing = True
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        # Fail any in-flight outbound waiters so callers don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("relay transport closed"))
        self._pending.clear()

    async def handshake(self) -> CapabilityDescriptor:
        if self._descriptor is not None:
            return self._descriptor
        if self._descriptor_ready is None:
            raise RuntimeError("handshake() called before connect()")
        return await asyncio.wait_for(self._descriptor_ready, timeout=self._connect_timeout_s)

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._inbound = handler

    # ── outbound ─────────────────────────────────────────────────────────
    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request_response(action)

    async def send_follow_up(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # follow_up rides the same outbound frame; the connector dispatches by
        # action.op. Kept as a distinct method to satisfy the transport Protocol
        # and to make the A2 call site explicit.
        return await self._request_response(action)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        result = await self._request_response(
            {"op": "get_chat_info", "chat_id": chat_id}, frame_type="outbound"
        )
        # The connector answers chat-info inside the outbound_result envelope.
        info = result.get("chat_info") or result
        return {"name": info.get("name", chat_id), "type": info.get("type", "dm")}

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        await self._send({"type": "interrupt", "session_key": session_key, "reason": reason})

    async def _request_response(
        self, action: Dict[str, Any], frame_type: str = "outbound"
    ) -> Dict[str, Any]:
        if self._ws is None:
            return {"success": False, "error": "relay transport not connected"}
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        try:
            await self._send({"type": frame_type, "requestId": request_id, "action": action})
            return await asyncio.wait_for(fut, timeout=self._outbound_timeout_s)
        except asyncio.TimeoutError:
            return {"success": False, "error": "relay outbound timed out"}
        finally:
            self._pending.pop(request_id, None)

    # ── wire I/O ─────────────────────────────────────────────────────────
    async def _send(self, frame: Dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("relay transport not connected")
        await self._ws.send(json.dumps(frame) + "\n")

    async def _read_loop(self) -> None:
        if self._ws is None:
            raise RuntimeError("_read_loop called before WebSocket was connected")
        buf = ""
        try:
            async for chunk in self._ws:
                buf += chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                # Newline-delimited frames; keep any trailing partial line.
                *lines, buf = buf.split("\n")
                for line in lines:
                    if line.strip():
                        await self._handle_frame(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - log + let the task end; reconnection is caller policy
            if not self._closing:
                logger.warning("relay ws read loop ended: %s", exc)

    async def _handle_frame(self, line: str) -> None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("relay: skipping malformed frame")
            return
        ftype = frame.get("type")
        if ftype == "descriptor":
            descriptor = CapabilityDescriptor.from_json(json.dumps(frame.get("descriptor", {})))
            self._descriptor = descriptor
            if self._descriptor_ready is not None and not self._descriptor_ready.done():
                self._descriptor_ready.set_result(descriptor)
        elif ftype == "inbound":
            if self._inbound is not None:
                event = _event_from_wire(frame.get("event", {}))
                await self._inbound(event)
        elif ftype == "outbound_result":
            fut = self._pending.get(frame.get("requestId", ""))
            if fut is not None and not fut.done():
                fut.set_result(frame.get("result", {}))
        elif ftype == "interrupt_inbound":
            # Bridged into the adapter's interrupt path by the runner wiring.
            handler = getattr(self, "_interrupt_inbound_handler", None)
            if handler is not None:
                await handler(frame.get("session_key", ""), frame.get("chat_id", ""))
        else:
            # hello/outbound/interrupt are gateway->connector; ignore if echoed.
            pass

    def set_interrupt_inbound_handler(self, handler: Any) -> None:
        """Register the callback for connector->gateway interrupt_inbound frames."""
        self._interrupt_inbound_handler = handler
