"""Relay transport protocol — the gateway<->connector wire contract. EXPERIMENTAL.

The ``RelayAdapter`` (gateway side) delegates all wire I/O to a ``RelayTransport``.
The gateway dials OUT to the connector, so a production transport is a WebSocket
client; in tests it is an in-memory stub (``tests/gateway/relay/stub_connector.py``).

This module defines the protocol surface only — no concrete transport. The
contract has four concerns:

  1. Lifecycle: ``connect`` / ``disconnect``.
  2. Handshake: ``handshake`` returns the ``CapabilityDescriptor`` the connector
     advertises for the platform this adapter fronts.
  3. Inbound: ``set_inbound_handler`` registers a callback the transport invokes
     with each normalized ``MessageEvent`` the connector delivers.
  4. Outbound: ``send_outbound`` carries send/edit/typing actions back to the
     connector; ``get_chat_info`` proxies a chat-info lookup; ``send_interrupt``
     routes a mid-turn /stop down the socket that owns the session_key.

EXPERIMENTAL: may change without a deprecation cycle until >=2 Class-1 platforms
validate it. See docs/relay-connector-contract.md.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

from gateway.platforms.base import MessageEvent
from gateway.relay.descriptor import CapabilityDescriptor

# Callback the transport invokes for each inbound normalized event.
InboundHandler = Callable[[MessageEvent], Awaitable[None]]

# Callback the transport invokes for each forwarded passthrough request (§5.1).
# The first arg is a PassthroughForward (gateway/relay/ws_transport.py) — typed
# as Any here to keep this protocol module free of a concrete-transport import
# (ws_transport imports FROM this module). The second is an optional bufferId
# (Phase 5 §5.3 buffered flip) the handler acks after durable handoff.
PassthroughHandler = Callable[[Any, Optional[str]], Awaitable[None]]


@runtime_checkable
class RelayTransport(Protocol):
    """Full gateway<->connector transport contract."""

    async def connect(self) -> bool:
        """Open the connection to the connector; return True on success."""
        ...

    async def disconnect(self) -> None:
        """Close the connection."""
        ...

    async def handshake(self) -> CapabilityDescriptor:
        """Return the capability descriptor the connector advertises."""
        ...

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        """Register the callback invoked with each inbound MessageEvent."""
        ...

    def set_passthrough_handler(self, handler: "PassthroughHandler") -> None:
        """Register the callback invoked with each forwarded passthrough request.

        Phase 5 §5.1: the passthrough plane (Discord interactions, Twilio, …)
        answers the provider's edge ACK at the connector, then forwards the real
        request to the gateway over this same outbound socket (a hosted gateway
        has no public inbound port). The transport invokes ``handler(forward,
        buffer_id)`` for each ``passthrough_forward`` frame. Optional on a
        transport (an in-memory stub may not implement it).
        """
        ...

    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Carry an outbound action (send/edit/typing) to the connector.

        Returns a result dict; for ``op == "send"`` it carries
        ``success`` and optionally ``message_id`` / ``error``.
        """
        ...

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Proxy a chat-info lookup to the connector."""
        ...

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        """Route a mid-turn /stop to the connector for ``session_key``.

        The connector forwards it down the socket owned by the gateway
        instance running that session (the /stop routing invariant). On the
        gateway side this is the OUTBOUND direction; the actual task
        cancellation happens when the connector echoes an interrupt inbound
        (handled in Task 1.4).
        """
        ...

    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        """Ask the connector to flip this instance to buffered-only (Phase 5 §5.3).

        Sends ``going_idle`` and awaits the connector's ``going_idle_ack`` — the
        connector-authoritative confirmation that live delivery stopped and inbound
        now buffers durably for replay on reconnect (Q-5.3c). Returns True on ack,
        False on timeout / not-connected (the caller proceeds to close regardless;
        without §5.3 wiring there is simply no buffering). Optional on a transport
        (an in-memory stub may not implement it). Emitted as part of the gateway's
        EXISTING drain transition — not a new idle path.
        """
        ...

    async def send_follow_up(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Act on a shared-identity capability bound to a session (A2 outbound).

        Some platforms hand the connector a credential that acts on the SHARED
        bot identity (e.g. a Discord interaction follow-up token, valid ~15min).
        Under A2 that credential NEVER reaches the gateway — the connector
        stripped it at the edge and bound it in its capability vault keyed by
        the session. To use it, the gateway issues a SEMANTIC action against the
        session it is already in; it never names or holds a token.

        The action dict carries:
          ``op``          == ``"follow_up"``
          ``session_key`` the session whose bound capability to wield
          ``kind``        the capability kind (e.g. ``"discord.interaction_token"``)
          ``content``     the message content to send via that capability
          ``metadata?``   optional extras

        The connector resolves the real capability (``resolveOutboundCapability``
        on its side), enforces the tenant match (tenant B can never wield tenant
        A's capability), and egresses. Returns ``{success, message_id?, error?}``;
        ``success`` is False when the capability is absent/expired or the tenant
        doesn't match — the gateway then has nothing to retry with (by design: a
        leaked gateway holds zero capability material).
        """
        ...
