"""Regression tests for Feishu inbound loop recovery.

Feishu callbacks arrive on an SDK worker thread, while inbound processing is
owned by the gateway event loop. A stale closed adapter-loop reference must not
leave queued events retrying forever, and a submit race must retain the event.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.asyncio
async def test_closed_loop_replays_pending_event_on_live_gateway_loop():
    """A closed adapter loop must rebind pending inbound work to the runner loop."""
    adapter = FeishuAdapter(PlatformConfig())
    adapter._running = True
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    gateway_loop = asyncio.get_running_loop()
    adapter._loop = stale_loop
    adapter.gateway_runner = SimpleNamespace(_gateway_loop=gateway_loop)
    received = []
    handled = asyncio.Event()

    async def _handle(data):
        received.append(data)
        handled.set()

    adapter._handle_message_event_data = _handle
    event = SimpleNamespace(tag="queued-before-rebind")

    def _failed_submit(_loop, coro):
        coro.close()
        return False

    try:
        # Force the first handoff to take the existing pending-queue path.
        with (
            patch.object(adapter, "_submit_on_loop", side_effect=_failed_submit),
            patch("plugins.platforms.feishu.adapter.threading.Thread"),
        ):
            adapter._on_message_event(event)
        assert adapter._pending_inbound_events == [event]

        # The runner loop is live; the drainer must rebind and replay FIFO.
        with patch("plugins.platforms.feishu.adapter.time.sleep", return_value=None):
            adapter._drain_pending_inbound_events()
        await asyncio.wait_for(handled.wait(), timeout=1.0)

        assert received == [event]
        assert adapter._loop is gateway_loop
        assert adapter._pending_inbound_events == []
    finally:
        adapter._running = False


@pytest.mark.asyncio
async def test_stopped_adapter_loop_rebinds_to_live_gateway_loop():
    """A stopped, not-yet-closed loop must not accept callbacks as live."""
    adapter = FeishuAdapter(PlatformConfig())
    gateway_loop = asyncio.get_running_loop()
    adapter._loop = SimpleNamespace(is_closed=lambda: False, is_running=lambda: False)
    adapter.gateway_runner = SimpleNamespace(_gateway_loop=gateway_loop)

    assert adapter._callback_loop() is gateway_loop


def test_message_event_requeues_when_submit_races_with_loop_shutdown():
    """A failed thread-safe submit must not silently lose an inbound event."""
    adapter = FeishuAdapter(PlatformConfig())
    adapter._running = True
    adapter._loop = SimpleNamespace(is_closed=lambda: False)
    event = SimpleNamespace(tag="submit-race")

    def _failed_submit(_loop, coro):
        coro.close()
        return False

    try:
        with (
            patch.object(adapter, "_submit_on_loop", side_effect=_failed_submit) as submit,
            patch("plugins.platforms.feishu.adapter.threading.Thread") as thread_cls,
        ):
            adapter._on_message_event(event)

        submit.assert_called_once()
        assert adapter._pending_inbound_events == [event]
        assert adapter._pending_drain_scheduled is True
        thread_cls.assert_called_once()
    finally:
        adapter._running = False
        adapter._pending_inbound_events.clear()
        adapter._pending_drain_scheduled = False
