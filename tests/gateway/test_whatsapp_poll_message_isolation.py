"""The bridge drops a batch from its queue as soon as GET /messages returns it (bridge.js splices
the queue), so a message that raises while the adapter builds or dispatches it must not take the
rest of the batch down with it, and the log line must name the exception."""

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


def _adapter(batch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))

    def _bridge_req(method, path, timeout, **kwargs):
        adapter._running = False  # a single poll
        return _Resp(batch)

    adapter._http_session = object()
    adapter._bridge_req = _bridge_req
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._report_bridge_exit = AsyncMock(return_value=False)
    adapter._send_read_receipt = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_one_failing_message_does_not_drop_the_rest_of_the_batch(caplog):
    batch = [{"messageId": "A", "body": "boom"}, {"messageId": "B", "body": "/stop"}]
    adapter = _adapter(batch)
    seen = []

    async def _build(msg):
        if msg["messageId"] == "A":
            raise TimeoutError()
        seen.append(msg["messageId"])
        # nothing to dispatch; what matters is that B was reached

    adapter._build_message_event = _build
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.whatsapp.adapter"):
        await asyncio.wait_for(adapter._poll_messages(), timeout=15)

    assert seen == ["B"]
    failed = [r for r in caplog.records if "Inbound message A dropped" in r.getMessage()]
    assert failed and "TimeoutError" in failed[0].getMessage()
    assert failed[0].exc_info is not None


@pytest.mark.asyncio
async def test_cancellation_still_ends_the_poll_loop():
    batch = [{"messageId": "A", "body": "one"}, {"messageId": "B", "body": "two"}]
    adapter = _adapter(batch)
    seen = []

    async def _build(msg):
        seen.append(msg["messageId"])
        raise asyncio.CancelledError()

    adapter._build_message_event = _build
    await asyncio.wait_for(adapter._poll_messages(), timeout=15)

    assert seen == ["A"]  # the loop stopped at the cancellation instead of logging it and moving on
