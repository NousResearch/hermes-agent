"""Delivery/read status of messages Hermes sends on WhatsApp is logged to gateway.log, one line per change.

Coatue request: Baileys already reports `messages.update` status (and per-recipient group receipts),
but the bridge only forwarded poll updates, so operators could not see whether a reply was delivered
or read. The bridge now queues a `delivery_status` event; the adapter logs it and never treats it as
an inbound message.
"""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


class _Resp:
    def __init__(self, payload):
        self.status, self._payload = 200, payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_delivery_status_is_logged_to_gateway_log_and_never_dispatched(caplog):
    from hermes_logging import COMPONENT_PREFIXES
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    batch = [
        {"event": "delivery_status", "messageId": "3EB0A1", "chatId": "15551234567@s.whatsapp.net",
         "status": "read", "timestamp": 1760000000},
        {"event": "delivery_status", "messageId": "3EB0A2", "chatId": "120363001234567890@g.us",
         "status": "delivered", "recipientId": "15550001111@s.whatsapp.net", "timestamp": 1760000050},
        {"event": "delivery_status", "messageId": "3EB0A3", "chatId": "15551234567@s.whatsapp.net",
         "status": "error", "error": "463", "timestamp": 1760000060},
    ]

    def _bridge_req(method, path, timeout, **kwargs):
        assert (method, path) == ("get", "messages")
        adapter._running = False  # one poll, then the loop exits after its 1s interval
        return _Resp(batch)

    adapter._http_session = object()
    adapter._bridge_req = _bridge_req
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._build_message_event = AsyncMock(return_value=None)
    adapter.handle_message = AsyncMock()

    with caplog.at_level(logging.INFO, logger="plugins.platforms.whatsapp.adapter"):
        await asyncio.wait_for(adapter._poll_messages(), timeout=10)

    tag = f"[{adapter.name}] WhatsApp delivery status:"
    lines = [r.getMessage() for r in caplog.records if r.name == "plugins.platforms.whatsapp.adapter"
             and "delivery status" in r.getMessage()]
    assert lines == [
        f"{tag} message=3EB0A1 chat=15551234567@s.whatsapp.net status=read at=1760000000",
        f"{tag} message=3EB0A2 chat=120363001234567890@g.us status=delivered recipient=15550001111@s.whatsapp.net at=1760000050",
        f"{tag} message=3EB0A3 chat=15551234567@s.whatsapp.net status=error at=1760000060 error=463",
    ]
    # gateway.log takes every plugins.platforms.* logger (hermes_logging component prefixes).
    assert any("plugins.platforms.whatsapp.adapter".startswith(p) for p in COMPONENT_PREFIXES["gateway"])
    adapter._build_message_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
