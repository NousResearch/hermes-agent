"""RelayAdapter — one generic gateway adapter fronted by the connector. EXPERIMENTAL.

A single ``BasePlatformAdapter`` subclass that, at handshake, receives a
``CapabilityDescriptor`` from the connector telling it which platform it is
fronting and which capabilities to advertise to the ``GatewayStreamConsumer``.
It implements the four abstract methods (``connect`` / ``disconnect`` / ``send``
/ ``get_chat_info``) plus the capability surface (``MAX_MESSAGE_LENGTH``,
``message_len_fn``, ``supports_draft_streaming``) by delegating wire I/O to an
injected transport and reading capabilities off the descriptor.

There is NO per-platform gateway code: the connector is the only side that knows
"this chat_id maps to a Discord channel, send it via the Discord websocket."
The gateway sees an ordinary ``MessageEvent`` in and calls ``adapter.send`` out.

EXPERIMENTAL: the transport protocol and descriptor schema may change without a
deprecation cycle until >=2 Class-1 platforms validate them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import RelayTransport

logger = logging.getLogger(__name__)


def _utf16_len(text: str) -> int:
    """Count UTF-16 code units (Telegram's length unit)."""
    return len(text.encode("utf-16-le")) // 2


# Table-driven length-unit selection from the descriptor's ``len_unit``.
_LEN_FNS: Dict[str, Callable[[str], int]] = {
    "chars": len,
    "utf16": _utf16_len,
}


class RelayAdapter(BasePlatformAdapter):
    """Generic relay adapter advertising a connector-negotiated capability profile."""

    def __init__(
        self,
        config: PlatformConfig,
        descriptor: CapabilityDescriptor,
        transport: Optional[RelayTransport] = None,
    ) -> None:
        # The relay adapter fronts many platforms but presents as a single
        # logical platform to the runner; Platform.RELAY identifies it.
        super().__init__(config, Platform.RELAY)
        self.descriptor = descriptor
        self._transport = transport
        # Capability surface read by stream_consumer (getattr(..., 4096)).
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        # Inbound delivery receiver (signed connector→gateway HTTP POSTs). Built
        # lazily in connect() when a delivery key + bind port are configured; a
        # purely-outbound dev gateway runs without it. See inbound_receiver.py.
        self._inbound_runner: Any = None

    # ── capability surface (from descriptor) ─────────────────────────────
    @property
    def message_len_fn(self) -> Callable[[str], int]:
        return _LEN_FNS.get(self.descriptor.len_unit, len)

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.descriptor.supports_draft_streaming

    # ── abstract methods (delegated to the transport) ────────────────────
    async def connect(self) -> bool:
        if self._transport is None:
            raise RuntimeError("RelayAdapter has no transport configured")
        self._transport.set_inbound_handler(self._on_inbound)
        ok = await self._transport.connect()
        if not ok:
            return False
        # Negotiate the real capability descriptor from the connector and adopt
        # it — the placeholder passed at construction is replaced by what the
        # connector advertises for the platform this gateway actually fronts.
        try:
            descriptor = await self._transport.handshake()
        except Exception as exc:  # noqa: BLE001 - a failed handshake = a failed connect
            logger.warning("relay handshake failed: %s", exc)
            return False
        self._apply_descriptor(descriptor)
        # Start the signed inbound-delivery receiver if configured (the connector
        # POSTs normalized events to it over HTTP, verified with the tenant
        # delivery key). Non-fatal: a receiver bind failure must not fail the
        # outbound connection — the gateway can still send.
        await self._maybe_start_inbound_receiver()
        return True

    async def _maybe_start_inbound_receiver(self) -> None:
        """Start the inbound HTTP receiver when a delivery key + port are set."""
        from gateway.relay import relay_inbound_config

        delivery_key, host, port = relay_inbound_config()
        if not (delivery_key and port):
            return  # no inbound URL configured -> outbound-only gateway
        try:
            from aiohttp import web

            from gateway.relay.inbound_receiver import InboundDeliveryReceiver

            receiver = InboundDeliveryReceiver(
                delivery_key_verify_list=lambda: [delivery_key],
                on_message=self._on_inbound,
                on_interrupt=self.on_interrupt,
            )
            runner = web.AppRunner(receiver.build_app(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            self._inbound_runner = runner
            logger.info("relay inbound receiver listening on http://%s:%s", host, port)
        except Exception as exc:  # noqa: BLE001 - inbound bind failure must not kill outbound
            logger.warning("relay inbound receiver failed to start: %s", exc)
            self._inbound_runner = None

    def _apply_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Adopt a (re)negotiated descriptor into the live capability surface."""
        self.descriptor = descriptor
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")

    async def _on_inbound(self, event) -> None:
        """Bridge a connector-delivered MessageEvent into the normal adapter path."""
        await self.handle_message(event)

    async def on_interrupt(self, session_key: str, chat_id: str) -> None:
        """Bridge a connector-delivered /stop into the adapter's interrupt path.

        The connector forwards a mid-turn interrupt down the socket owned by
        the gateway instance running ``session_key``; this routes it to the
        existing per-session interrupt mechanism (sets the
        ``_active_sessions[session_key]`` Event and clears typing), cancelling
        the right turn without touching sibling sessions.
        """
        await self.interrupt_session_activity(session_key, chat_id)

    async def disconnect(self) -> None:
        if self._inbound_runner is not None:
            try:
                await self._inbound_runner.cleanup()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._inbound_runner = None
        if self._transport is not None:
            await self._transport.disconnect()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata or {},
            }
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Proxied to the connector (it owns the platform connection / cache).
        if self._transport is None:
            return {"name": chat_id, "type": "dm"}
        return await self._transport.get_chat_info(chat_id)

    async def send_follow_up(
        self,
        session_key: str,
        kind: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send via a shared-identity capability bound to a session (A2 outbound).

        The gateway never holds the credential: it names the session it is
        already in plus the capability ``kind``, and the connector resolves the
        real value from its vault and egresses (enforcing the tenant match). Used
        e.g. to post a Discord interaction follow-up as the shared bot without
        the token ever reaching the gateway. See RelayTransport.send_follow_up.
        """
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._transport.send_follow_up(
            {
                "op": "follow_up",
                "session_key": session_key,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            }
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )
