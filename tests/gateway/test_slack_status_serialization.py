"""Ordering tests for Slack assistant status set/clear serialization.

A delayed "is thinking" write must never land after a completion clear:
the per-thread lock orders the writes, and a queued set skips itself when
the tracked entry was cleared while it waited.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


class RecordingClient:
    def __init__(self):
        self.calls = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.block_next_set = False
        self.block_next_clear = False

    async def assistant_threads_setStatus(self, channel_id, thread_ts, status):
        if status and self.block_next_set:
            self.block_next_set = False
            self.entered.set()
            await self.release.wait()
        if not status and self.block_next_clear:
            self.block_next_clear = False
            self.entered.set()
            await self.release.wait()
        self.calls.append((thread_ts, status))


def make_adapter(client):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._app = MagicMock()
    adapter._get_client = lambda chat_id, team_id=None: client
    adapter._channel_team = {"C123": "T1"}
    return adapter


META = {"thread_id": "100.000", "team_id": "T1"}


@pytest.mark.asyncio
async def test_clear_lands_after_in_flight_set():
    client = RecordingClient()
    adapter = make_adapter(client)

    client.block_next_set = True
    set_task = asyncio.create_task(adapter.send_typing("C123", dict(META)))
    await client.entered.wait()  # the set is mid-flight at the Slack API

    stop_task = asyncio.create_task(adapter.stop_typing("C123", dict(META)))
    await asyncio.sleep(0.01)
    assert client.calls == []  # the clear is queued behind the lock

    client.release.set()
    await set_task
    await stop_task

    assert [s for _, s in client.calls] == ["is thinking...", ""]


@pytest.mark.asyncio
async def test_set_queued_behind_in_flight_clear_is_dropped():
    """A send_typing that starts while stop_typing is mid-clear re-registers
    its entry, but the clear pops it inside the lock, so the queued set must
    drop itself instead of landing after the clear."""
    client = RecordingClient()
    adapter = make_adapter(client)

    # A tracked status exists, then the clear starts and blocks mid-API-call.
    await adapter.send_typing("C123", dict(META))
    client.block_next_clear = True
    stop_task = asyncio.create_task(adapter.stop_typing("C123", dict(META)))
    await client.entered.wait()

    # A refresh tick arrives while the clear is in flight.
    set_task = asyncio.create_task(adapter.send_typing("C123", dict(META)))
    await asyncio.sleep(0.01)

    client.release.set()
    await stop_task
    await set_task

    statuses = [s for _, s in client.calls]
    assert statuses == ["is thinking...", ""]  # no set after the final clear


@pytest.mark.asyncio
async def test_queued_set_skips_after_clear():
    client = RecordingClient()
    adapter = make_adapter(client)

    key = ("T1", "C123", "100.000")
    lock = adapter._status_thread_lock(key)
    await lock.acquire()  # simulate a clear already holding the lock
    try:
        set_task = asyncio.create_task(adapter.send_typing("C123", dict(META)))
        await asyncio.sleep(0.01)
        assert key in adapter._active_status_threads  # registered, waiting

        # The completion clears the tracked entry while the set waits.
        adapter._active_status_threads.pop(key)
    finally:
        lock.release()
    await set_task

    assert client.calls == []  # the stale set never reached Slack
