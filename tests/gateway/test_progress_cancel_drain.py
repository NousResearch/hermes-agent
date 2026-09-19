"""The idle progress loop must still drain its queue when the turn cancels it.

``TurnRunner.send_progress_messages`` spends almost all of its time waiting for
the next tool event.  When the turn ends, ``_run_agent_cleanup_turn_tasks``
cancels the task, and the cancellation handler is what flushes whatever is still
sitting in ``progress_queue`` into one last bubble edit.

If that idle wait lives inside a sibling ``except queue.Empty`` handler, the
``CancelledError`` raised there escapes the whole ``try`` statement — a sibling
``except`` clause never catches an exception raised inside another clause of the
same ``try`` — so the drain never runs and the trailing tool lines are lost.

These tests park the loop in its idle wait, then cancel it, and assert the drain
actually happened.
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
from contextlib import suppress

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class ProgressBubbleAdapter(BasePlatformAdapter):
    """Editable-message adapter that records every send/edit/typing call."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[str] = []
        self.edits: list[str] = []
        self.typing: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="bubble-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_runner(adapter, ctx) -> TurnRunner:
    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return adapter

    return TurnRunner(_StubGatewayRunner(), ctx)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the progress loop")
        await asyncio.sleep(0.01)


async def _park_loop_in_idle_wait(adapter, ctx, runner):
    """Render one tool line, then let the loop settle into its idle wait."""
    task = asyncio.create_task(runner.send_progress_messages())
    ctx.progress_queue.put("🔍 web_search — first")
    # send_typing is the last call of a rendered tick, so once it lands the loop
    # is heading straight back to an empty queue.
    await _wait_for(lambda: adapter.typing)
    await asyncio.sleep(0.05)
    return task


@pytest.fixture
def progress_turn():
    adapter = ProgressBubbleAdapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"),
        _run_still_current=lambda: True,
        progress_queue=queue_mod.Queue(),
        tool_progress_enabled=True,
    )
    return adapter, ctx, _make_runner(adapter, ctx)


@pytest.mark.asyncio
async def test_cancel_during_idle_wait_flushes_queued_tool_lines(progress_turn):
    """A line queued just before cancellation must reach the final bubble edit."""
    adapter, ctx, runner = progress_turn
    task = await _park_loop_in_idle_wait(adapter, ctx, runner)

    ctx.progress_queue.put("🔍 web_search — second")
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert adapter.edits, "cancellation produced no final progress edit"
    assert "second" in adapter.edits[-1], (
        "the tool line queued before cancellation was dropped — the cancel "
        f"handler never drained the queue (edits: {adapter.edits})"
    )


@pytest.mark.asyncio
async def test_cancel_during_idle_wait_reaches_the_drain_handler(progress_turn, monkeypatch):
    """The cancellation handler itself must run, not just happen to be reachable."""
    adapter, ctx, runner = progress_turn
    calls: list[int] = []
    original = TurnRunner._drain_progress_on_cancel

    async def counting_drain(self, st):
        calls.append(1)
        return await original(self, st)

    monkeypatch.setattr(TurnRunner, "_drain_progress_on_cancel", counting_drain)

    task = await _park_loop_in_idle_wait(adapter, ctx, runner)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert calls == [1], "_drain_progress_on_cancel did not run on cancellation"
