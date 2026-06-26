"""Test-only in-memory stub connector implementing RelayTransport.

MUST stay under tests/ — never under plugins/ or gateway/ (a CI guard in
test_no_stub_leak.py asserts this). It lets Phase 1 prove the gateway side of
the relay end-to-end with zero dependency on the real (Node) connector.

The stub:
  - hands back a fixed CapabilityDescriptor at handshake,
  - lets a test push synthetic inbound MessageEvents (push_inbound),
  - records every outbound action (sent/interrupts) for assertions,
  - answers get_chat_info from a small fixture map.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gateway.platforms.base import MessageEvent
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import InboundHandler


class StubConnector:
    """In-memory RelayTransport for tests."""

    def __init__(self, descriptor: CapabilityDescriptor) -> None:
        self._descriptor = descriptor
        self._inbound: Optional[InboundHandler] = None
        self._interrupt_inbound: Optional[Any] = None
        self._passthrough: Optional[Any] = None
        self.connected = False
        self.sent: List[Dict[str, Any]] = []
        self.interrupts: List[Dict[str, Any]] = []
        self.follow_ups: List[Dict[str, Any]] = []
        self.chat_info: Dict[str, Dict[str, Any]] = {}
        # Canned result for the next send_outbound (override per-test).
        self.next_send_result: Dict[str, Any] = {"success": True, "message_id": "m1"}
        # Canned result for the next send_follow_up (override per-test). Default
        # mimics a resolved capability egress; set success=False to simulate an
        # absent/expired capability or a tenant mismatch on the connector side.
        self.next_follow_up_result: Dict[str, Any] = {"success": True, "message_id": "f1"}

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def handshake(self) -> CapabilityDescriptor:
        return self._descriptor

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._inbound = handler

    def set_interrupt_inbound_handler(self, handler: Any) -> None:
        """Mirror the real WS transport: the adapter registers its interrupt
        bridge here so connector→gateway interrupt_inbound frames route to it."""
        self._interrupt_inbound = handler

    def set_passthrough_handler(self, handler: Any) -> None:
        """Mirror the real WS transport: the adapter registers its passthrough
        bridge here so connector→gateway passthrough_forward frames route to it
        (Phase 5 §5.1)."""
        self._passthrough = handler

    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self.sent.append(action)
        if action.get("op") == "send":
            return dict(self.next_send_result)
        return {"success": True}

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return self.chat_info.get(chat_id, {"name": chat_id, "type": "dm"})

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        self.interrupts.append({"session_key": session_key, "reason": reason})

    async def send_follow_up(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self.follow_ups.append(action)
        return dict(self.next_follow_up_result)

    # ── test driver ──────────────────────────────────────────────────────
    async def push_inbound(self, event: MessageEvent) -> None:
        """Simulate the connector delivering a normalized inbound event."""
        if self._inbound is None:
            raise RuntimeError("no inbound handler registered (call adapter.connect first)")
        await self._inbound(event)

    async def push_interrupt(self, session_key: str, chat_id: str) -> None:
        """Simulate the connector delivering an interrupt_inbound over the WS."""
        if self._interrupt_inbound is None:
            raise RuntimeError("no interrupt_inbound handler registered (call adapter.connect first)")
        await self._interrupt_inbound(session_key, chat_id)

    async def push_passthrough(self, forward: Any, buffer_id: Optional[str] = None) -> None:
        """Simulate the connector forwarding a passthrough request over the WS (§5.1)."""
        if self._passthrough is None:
            raise RuntimeError("no passthrough handler registered (call adapter.connect first)")
        await self._passthrough(forward, buffer_id)
