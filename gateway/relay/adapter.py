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
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.relay.descriptor import CapabilityDescriptor

logger = logging.getLogger(__name__)


def _utf16_len(text: str) -> int:
    """Count UTF-16 code units (Telegram's length unit)."""
    return len(text.encode("utf-16-le")) // 2


# Table-driven length-unit selection from the descriptor's ``len_unit``.
_LEN_FNS: Dict[str, Callable[[str], int]] = {
    "chars": len,
    "utf16": _utf16_len,
}


@runtime_checkable
class RelayTransport(Protocol):
    """Minimal transport contract the RelayAdapter delegates wire I/O to.

    The full protocol (inbound MessageEvent stream, interrupt channel) is
    fleshed out in gateway/relay/transport.py (Task 1.2); the adapter only
    needs these outbound + lifecycle calls to satisfy the abstract methods.
    """

    async def connect(self) -> bool: ...

    async def disconnect(self) -> None: ...

    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]: ...

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]: ...


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
        return await self._transport.connect()

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
