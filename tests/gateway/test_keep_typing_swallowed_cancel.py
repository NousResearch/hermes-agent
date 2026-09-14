"""Regression: a cancel() that Python 3.11's ``asyncio.wait_for`` swallows must
not leave ``_keep_typing`` running forever.

On Python < 3.12, ``wait_for`` returns the inner result and swallows the
CancelledError when the inner future is already done at the moment the outer
task is cancelled.  ``_keep_typing`` bounds every ``send_typing`` with
``wait_for``, so a cancel that lands in that window is lost, the refresh loop
keeps sending ``sendChatAction`` every 2s and the platform shows "typing..."
indefinitely (live incident 2026-09-05, Telegram DM stuck for 1.5h until the
gateway was restarted).
"""

import asyncio

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    Platform,
    PlatformConfig,
    SendResult,
)


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


async def _run_swallowed_cancel(adapter, use_stop_refresh: bool):
    """Cancel the typing task from inside send_typing, right before it returns.

    That reproduces the wait_for race deterministically: the outer task's
    wakeup (CancelledError) is scheduled first, the inner future completes in
    the same loop iteration, and wait_for then sees ``fut.done()`` and returns
    normally instead of re-raising the cancellation.
    """
    sends = []
    state = {"task": None, "armed": True}

    async def send_typing(chat_id, metadata=None):
        sends.append(chat_id)
        if state["armed"] and state["task"] is not None:
            state["armed"] = False
            state["task"].cancel()

    adapter.send_typing = send_typing
    task = asyncio.create_task(adapter._keep_typing("chat-1", interval=0.05))
    state["task"] = task
    if use_stop_refresh:
        # Let the first tick run so the cancel from inside send_typing fires,
        # then go through the real stop path.
        await asyncio.sleep(0.02)
        await adapter._stop_typing_refresh("chat-1", task, timeout=0.2)
    await asyncio.sleep(0.5)
    return task, sends


@pytest.mark.asyncio
async def test_keep_typing_exits_when_cancel_is_swallowed():
    adapter = _StubAdapter()
    task, sends = await _run_swallowed_cancel(adapter, use_stop_refresh=False)
    assert task.done(), "typing refresh loop kept running after a swallowed cancel"
    # At most the tick that carried the cancel; no further refreshes.
    assert len(sends) <= 2


@pytest.mark.asyncio
async def test_stop_typing_refresh_survives_swallowed_cancel():
    adapter = _StubAdapter()
    task, sends = await _run_swallowed_cancel(adapter, use_stop_refresh=True)
    assert task.done(), "_stop_typing_refresh left the typing task alive"
    assert len(sends) <= 3
    assert "chat-1" not in adapter._typing_paused
