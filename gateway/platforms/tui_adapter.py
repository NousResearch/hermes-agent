"""Callback-based platform adapter for TUI/CLI/WebUI notification delivery.

This adapter registers with the gateway's adapter system so that
_kanban_notifier_watcher (and any other gateway-side event source) can
deliver events to connected TUI/CLI sessions via ``adapter.send()``.

Unlike Telegram/Discord adapters that push to external APIs, this adapter
invokes an in-process callback that bridges into the TUI's JSON-RPC emit
system or the CLI's drain_notifications path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

# Type for the delivery callback: (chat_id, message_text, metadata) -> None
DeliveryCallback = Callable[[str, str, Dict[str, Any]], None]


class TUIAdapter(BasePlatformAdapter):
    """Platform adapter that delivers messages via an in-process callback.

    The gateway's kanban notifier calls ``adapter.send(chat_id, msg)``
    which invokes the registered callback.  The callback bridges into
    whatever delivery mechanism the TUI/CLI uses (JSON-RPC emit, SSE,
    stdout print, etc.).
    """

    def __init__(self, config: PlatformConfig, platform: Platform):
        super().__init__(config, platform)
        self._delivery_callback: Optional[DeliveryCallback] = None

    def set_delivery_callback(self, cb: DeliveryCallback) -> None:
        """Register the callback that receives delivery requests."""
        self._delivery_callback = cb

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver a message to the TUI/CLI via the registered callback."""
        if self._delivery_callback is None:
            logger.debug("tui_adapter: no delivery callback registered, dropping message")
            return SendResult(success=False, error="no delivery callback")
        try:
            self._delivery_callback(chat_id, content, metadata or {})
            return SendResult(success=True)
        except Exception as exc:
            logger.warning("tui_adapter: delivery callback failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def start(self) -> None:
        """No external connection needed — callback is set externally."""
        self._running = True

    async def stop(self) -> None:
        self._running = False
