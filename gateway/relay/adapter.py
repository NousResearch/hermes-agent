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
        # chat_id -> guild_id (Discord) / workspace scope, learned from inbound
        # events. The connector's egress guard resolves the owning tenant from
        # the OUTBOUND action's metadata.guild_id; the gateway's generic delivery
        # path (run.py _thread_metadata_for_source) only carries thread_id, so we
        # re-attach the scope here from what we saw inbound. Keyed by chat_id
        # (channel) since that's what send() receives. See routedEgressGuard.ts.
        self._scope_by_chat: Dict[str, str] = {}
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")

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
        # Inbound interrupts (connector -> owning gateway) arrive as
        # interrupt_inbound frames over the SAME outbound WS; bridge them to the
        # adapter's interrupt path. WS-only: there is no inbound HTTP receiver.
        set_interrupt = getattr(self._transport, "set_interrupt_inbound_handler", None)
        if callable(set_interrupt):
            set_interrupt(self.on_interrupt)
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
        # Inbound (messages + interrupts) is delivered over the outbound WS via
        # the connector's relay bus — there is NO inbound HTTP endpoint (hosted
        # gateways have no public IP). The transport's reader already dispatches
        # `inbound` / `interrupt_inbound` frames to the handlers wired above.
        return True

    def _apply_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Adopt a (re)negotiated descriptor into the live capability surface."""
        self.descriptor = descriptor
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")

    async def _on_inbound(self, event) -> None:
        """Bridge a connector-delivered MessageEvent into the normal adapter path."""
        self._capture_scope(event)
        await self.handle_message(event)

    def _capture_scope(self, event) -> None:
        """Remember chat_id -> guild scope from an inbound event so our outbound
        (the agent's reply) can re-assert it for the connector's egress tenant
        resolution. Never raises — scope tracking must not break inbound."""
        try:
            src = getattr(event, "source", None)
            scope = getattr(src, "guild_id", None) if src else None
            chat = getattr(src, "chat_id", None) if src else None
            if scope and chat:
                self._scope_by_chat[str(chat)] = str(scope)
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure the outbound metadata carries guild_id for the connector's
        egress tenant resolution. The connector resolves the owning tenant from
        metadata.guild_id (Discord); without it egress is declined as
        'target not routed to an onboarded tenant'. No-op when we have no scope
        for this chat (e.g. DMs) or it's already present."""
        meta: Dict[str, Any] = dict(metadata or {})
        if not meta.get("guild_id"):
            scope = self._scope_by_chat.get(str(chat_id))
            if scope:
                meta["guild_id"] = scope
        return meta

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
                "metadata": self._with_scope(chat_id, metadata),
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
