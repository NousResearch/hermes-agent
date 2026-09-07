"""Relay transport protocol — the gateway<->connector wire contract. EXPERIMENTAL.

The ``RelayAdapter`` delegates all wire I/O to a ``RelayTransport``. The gateway
dials OUT to the connector, so production is a WebSocket client (``ws_transport.py``)
and tests use an in-memory stub (``tests/gateway/relay/stub_connector.py``). This
module defines the protocol surface only. May change without a deprecation cycle
until >=2 Class-1 platforms validate it. See docs/relay-connector-contract.md.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

from gateway.platforms.base import MessageEvent
from gateway.relay.descriptor import CapabilityDescriptor

# Callback the transport invokes for each inbound normalized event.
InboundHandler = Callable[[MessageEvent], Awaitable[None]]

# Callback for each forwarded passthrough request (§5.1): ``(forward, buffer_id)``.
# ``forward`` is a ws_transport.PassthroughForward, typed Any here because
# ws_transport imports FROM this module; ``buffer_id`` (§5.3 buffered flip) is
# acked by the handler after durable handoff.
PassthroughHandler = Callable[[Any, Optional[str]], Awaitable[None]]


@runtime_checkable
class RelayTransport(Protocol):
    """Full gateway<->connector transport contract."""

    async def connect(self) -> bool:
        """Open the connection to the connector; return True on success."""
        ...

    async def disconnect(self) -> None:
        ...

    async def handshake(self) -> CapabilityDescriptor:
        """Return the capability descriptor the connector advertises."""
        ...

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        ...

    def set_passthrough_handler(self, handler: "PassthroughHandler") -> None:
        """Register the callback for each forwarded passthrough request (§5.1).

        The connector answers the provider's edge ACK itself, then forwards the real
        request over this same outbound socket (a hosted gateway has no public
        inbound port). Optional on a transport (a stub may not implement it).
        """
        ...

    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """Carry an outbound action (send/edit/typing) to the connector.

        Returns a result dict; for ``op == "send"`` it carries ``success`` and
        optionally ``message_id`` / ``error``. ``platform`` tags WHICH fronted
        platform this reply targets (the transport resolves the matching botId);
        omitted ⇒ the connector uses the session's default platform.
        """
        ...

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Proxy a chat-info lookup to the connector."""
        ...

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        """Route a mid-turn /stop to the connector for ``session_key`` (OUTBOUND
        direction; the actual cancellation happens when the connector echoes an
        interrupt inbound down the socket owning that session)."""
        ...

    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        """Ask the connector to flip this instance to buffered-only (§5.3).

        Sends ``going_idle`` and awaits the connector-authoritative ``going_idle_ack``
        (live delivery stopped; inbound now buffers for replay on reconnect). True on
        ack, False on timeout / not-connected — the caller closes regardless.
        Optional on a transport; part of the gateway's EXISTING drain transition.
        """
        ...

    async def send_follow_up(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """Act on a shared-identity capability bound to a session (A2 outbound).

        A credential acting on the SHARED bot identity (e.g. a Discord interaction
        follow-up token) NEVER reaches the gateway: the connector vaults it keyed by
        session and the gateway issues a SEMANTIC action (``op == "follow_up"``,
        ``session_key``, ``kind`` e.g. ``"discord.interaction_token"``, ``content``,
        optional ``metadata``). Returns ``{success, message_id?, error?}``; ``success``
        is False when the capability is absent/expired or the tenant mismatches —
        nothing to retry with (a leaked gateway holds zero capability material).
        """
        ...
