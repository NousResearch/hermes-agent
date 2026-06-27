"""Tests for the grace window in ``cancel_background_tasks``.

The /restart handler queues its ack reply and ``request_restart`` begins
teardown ~50ms later.  Teardown's ``cancel_background_tasks`` used to
cancel the in-flight ``_process_message_background`` task mid-send, so
the "Restarting gateway" ack was reliably lost whenever no active agents
forced a longer drain (the send needs a full HTTPS round trip).  The
planned-restart path now passes ``grace_seconds`` so in-flight tasks can
finish naturally before cancellation.
"""
import asyncio

import pytest

from tests.gateway.restart_test_helpers import RestartTestAdapter


def _track(adapter, coro):
    task = asyncio.get_event_loop().create_task(coro)
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    return task


@pytest.mark.asyncio
async def test_grace_lets_inflight_send_finish():
    """A task that completes inside the grace window is not cancelled."""
    adapter = RestartTestAdapter()
    delivered = asyncio.Event()

    async def fake_send():
        await asyncio.sleep(0.05)  # simulated network round trip
        delivered.set()

    task = _track(adapter, fake_send())
    await adapter.cancel_background_tasks(grace_seconds=2.0)

    assert delivered.is_set()
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_no_grace_cancels_inflight_send():
    """Without a grace window the in-flight task is cancelled (old behavior)."""
    adapter = RestartTestAdapter()
    delivered = asyncio.Event()

    async def fake_send():
        await asyncio.sleep(0.5)
        delivered.set()

    task = _track(adapter, fake_send())
    await adapter.cancel_background_tasks()

    assert not delivered.is_set()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_grace_expiry_still_cancels():
    """A task that outlives the grace window is cancelled; the call stays bounded."""
    adapter = RestartTestAdapter()

    async def slow_task():
        await asyncio.sleep(30)

    task = _track(adapter, slow_task())
    await asyncio.wait_for(
        adapter.cancel_background_tasks(grace_seconds=0.1), timeout=6.0
    )

    assert task.cancelled()
