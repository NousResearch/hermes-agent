"""Regression test: _schedule_ephemeral_delete must retain its task.

CPython's event loop only holds a weak reference to running tasks,
so a bare ``asyncio.create_task(coro)`` whose return value is dropped
can be garbage-collected mid-flight.  That produced two failure modes
in BasePlatformAdapter._schedule_ephemeral_delete:

  1. "Task was destroyed but it is pending!" warnings.
  2. Ephemeral message deletions silently never running.

Fix: park the task in ``self._background_tasks`` (already initialized
in BasePlatformAdapter.__init__ and used by handle_message) and
discard via a done-callback.  Same pattern as gateway/platforms/sms.py.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter.delete_message = AsyncMock(return_value=None)
    return adapter


@pytest.mark.asyncio
async def test_schedule_ephemeral_delete_retains_then_discards_task():
    """Task must be added to _background_tasks while pending, then
    discarded after completion via the done-callback."""
    adapter = _make_adapter()

    # Patch asyncio.sleep so the task is observable while pending but
    # finishes deterministically when we release it.
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_sleep(_):
        started.set()
        await release.wait()

    orig_sleep = asyncio.sleep
    asyncio.sleep = fake_sleep
    try:
        assert adapter._background_tasks == set()

        adapter._schedule_ephemeral_delete("chat-1", "msg-1", 5)
        # Hand control to the loop so the task starts and blocks in
        # fake_sleep.
        await started.wait()

        assert len(adapter._background_tasks) == 1
        (task,) = adapter._background_tasks
        assert not task.done()

        # Release the sleep; task completes and the done-callback fires.
        release.set()
        await task

        assert adapter._background_tasks == set(), (
            "done-callback must evict completed task"
        )
        adapter.delete_message.assert_awaited_once_with(
            chat_id="chat-1", message_id="msg-1"
        )
    finally:
        asyncio.sleep = orig_sleep


def test_schedule_ephemeral_delete_without_running_loop_is_safe():
    """When called outside a running loop, the coroutine is closed and
    no task is registered."""
    adapter = _make_adapter()
    # No running loop here — create_task raises RuntimeError, the
    # except branch closes the coroutine and returns.
    adapter._schedule_ephemeral_delete("chat-2", "msg-2", 5)
    assert adapter._background_tasks == set()
    adapter.delete_message.assert_not_called()
