"""Tests for BasePlatformAdapter._keep_typing surviving a lost cancel().

The bug: ``_keep_typing`` (gateway/platforms/base.py) trusts that the
``cancel()`` sent by ``_stop_typing_refresh`` reaches it.  When that cancel
is lost, the orphaned loop keeps refreshing the platform typing indicator
on its 2-second cadence with nothing left to stop it: ``_stop_typing_refresh``
gives up its bounded wait after 0.5s, un-pauses the chat, and the session's
``stop_event`` is only ever set by a user interrupt.  Observed in production
(Python 3.11, Matrix): a leaked refresh task hammered the homeserver's
``PUT /typing`` every 2 seconds for over three hours after its turn had
ended normally, pinning every client's "working" indicator.

Two real ways the cancel gets lost, both exercised below by one stub:

* Python 3.11's ``asyncio.wait_for`` returns the inner result and drops an
  external ``cancel()`` that lands in the same loop iteration as the inner
  awaitable completing (CPython gh-86296; rewritten in 3.12).  The per-tick
  ``wait_for(self.send_typing(...))`` in ``_keep_typing`` is exactly that
  shape, so any turn whose cleanup coincides with a ``send_typing``
  round-trip finishing can leak its loop.
* An adapter or client library that catches ``CancelledError`` inside
  ``send_typing`` swallows the cancel on any Python version.

The fix: ``_keep_typing`` registers itself in ``_typing_refresh_tasks`` and
re-checks its membership every tick; ``_stop_typing_refresh`` removes the
task it was handed *before* cancelling it.  A loop whose cancel was lost
exits on its next tick instead of refreshing forever.  Only the handed-in
task is touched: other live loops for the same chat_id belong to concurrent
sessions (per-user group sessions, Slack threads sharing a channel) and
must keep running.
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
        self.send_typing_calls = 0
        self.stop_typing_calls = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def send_typing(self, chat_id, metadata=None):
        self.send_typing_calls += 1

    async def stop_typing(self, chat_id):
        self.stop_typing_calls += 1


async def _drain(*tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _finished(task: asyncio.Task, timeout: float = 2.0) -> bool:
    """True if ``task`` completes within ``timeout``.

    Uses ``asyncio.wait`` rather than ``wait_for``: a ``wait_for`` timeout
    cancels the task, and ``_keep_typing`` treats cancellation as a normal
    exit — which would make a leaked loop look like it stopped on its own.
    """
    done, _ = await asyncio.wait({task}, timeout=timeout)
    return task in done


class TestTypingRefreshLostCancel:
    @pytest.mark.asyncio
    async def test_loop_exits_after_stop_even_when_cancel_is_lost(self):
        """_stop_typing_refresh must stop the loop it was handed even if the
        cancel() it sends never takes effect."""
        adapter = _StubAdapter()
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        gate = loop.create_future()

        async def send_typing(chat_id, metadata=None):
            adapter.send_typing_calls += 1
            entered.set()
            try:
                await gate
            except asyncio.CancelledError:
                # A library/adapter swallowing the cancel (any Python).
                pass

        adapter.send_typing = send_typing
        task = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.wait_for(entered.wait(), timeout=2.0)

        # Complete send_typing in the same loop iteration as the turn's
        # cleanup cancels the loop: on Python 3.11 wait_for returns the
        # result and the cancel is dropped (gh-86296).  _stop_typing_refresh
        # calls typing_task.cancel() before its first await, so this is one
        # iteration.
        gate.set_result(None)
        await adapter._stop_typing_refresh("123", task)

        if not await _finished(task):
            await _drain(task)
            pytest.fail("refresh loop kept running after its turn stopped it")

        calls_at_exit = adapter.send_typing_calls
        assert adapter.stop_typing_calls >= 1, "loop exit must clear platform typing"
        assert task not in adapter._typing_refresh_tasks
        # A finished task cannot refresh again.
        await asyncio.sleep(0.15)
        assert adapter.send_typing_calls == calls_at_exit

    @pytest.mark.asyncio
    async def test_stopping_one_turn_leaves_sibling_loop_in_same_chat_alone(self):
        """Two sessions can run turns in one chat_id at once (per-user group
        sessions, Slack threads).  Stopping one turn's loop, or the
        handle-less final stop, must not kill the other turn's loop."""
        adapter = _StubAdapter()
        first = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        second = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        assert not first.done() and not second.done()

        await adapter._stop_typing_refresh("123", first)
        assert await _finished(first)
        assert not second.done(), "sibling loop must survive the other turn's stop"

        # The handle-less safety-net stop at the end of a turn.
        await adapter._stop_typing_refresh("123", None, stop_attempts=1)
        await asyncio.sleep(0.15)
        assert not second.done(), "handle-less stop must not touch a live loop"
        assert second in adapter._typing_refresh_tasks

        await _drain(second)

    @pytest.mark.asyncio
    async def test_normal_stop_paths_leave_no_registration_behind(self):
        """Neither the cancel path nor the stop_event path may leak a task
        reference in the registry."""
        adapter = _StubAdapter()

        task = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        assert task in adapter._typing_refresh_tasks
        await adapter._stop_typing_refresh("123", task)
        assert await _finished(task)
        assert task not in adapter._typing_refresh_tasks

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing("123", interval=0.05, stop_event=stop_event)
        )
        await asyncio.sleep(0.12)
        stop_event.set()
        assert await _finished(task)
        assert adapter._typing_refresh_tasks == set()
