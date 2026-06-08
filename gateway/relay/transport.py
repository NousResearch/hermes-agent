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
