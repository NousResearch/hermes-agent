"""Exercise real serial card consumer, including slow I/O and stale turns."""
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from gateway.progress_cards import run_progress_card, finish_progress_card
from gateway.turn_context import TurnContext


def context():
    return TurnContext(
        source=SimpleNamespace(chat_id="chat"), session_id="same-session",
        progress_queue=queue.Queue(), _run_still_current=lambda: True,
        result_holder=[{"final_response": "done"}], _progress_cards=True,
    )


async def until(predicate, timeout=3):
    async def check():
        while not predicate():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(check(), timeout)


@pytest.mark.asyncio
async def test_slow_create_and_finish_are_serialized_without_duplicate_cards():
    ctx = context()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def card(chat, snapshot, **kwargs):
        calls.append((snapshot, kwargs))
        if kwargs["message_id"] is None:
            entered.set()
            await release.wait()
        return SendResult(success=True, message_id="one-card")
    adapter = SimpleNamespace(send_progress_card=card, send=AsyncMock())
    ctx.progress_queue.put({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal", "preview": "pwd"})
    task = asyncio.create_task(run_progress_card(ctx, adapter))
    await entered.wait()
    finish = asyncio.create_task(finish_progress_card(ctx, task))
    await asyncio.sleep(0.02)
    release.set()
    await finish
    assert len(calls) == 2
    assert calls[0][1]["message_id"] is None
    assert calls[1][1]["message_id"] == "one-card"
    assert calls[1][0]["status"] == "completed"
    assert task.done()


@pytest.mark.asyncio
async def test_throttled_update_flushes_while_next_tool_is_still_running():
    ctx = context()
    adapter = SimpleNamespace(send_progress_card=AsyncMock(return_value=SendResult(success=True, message_id="card")), send=AsyncMock())
    ctx.progress_queue.put({"type": "commentary", "text": "First"})
    task = asyncio.create_task(run_progress_card(ctx, adapter))
    try:
        await until(lambda: adapter.send_progress_card.await_count == 1)
        ctx.progress_queue.put({"type": "tool.started", "tool_call_id": "long", "tool_name": "terminal", "preview": "long build"})
        await until(lambda: adapter.send_progress_card.await_count == 2)
        snapshot = adapter.send_progress_card.call_args.args[1]
        assert any("long build" in row and "running" in row for row in snapshot["details"])
    finally:
        await finish_progress_card(ctx, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_empty_or_stale_turn_never_creates_card(stale):
    ctx = context()
    adapter = SimpleNamespace(send_progress_card=AsyncMock(), send=AsyncMock())
    if stale:
        ctx._run_still_current = lambda: False
        ctx.progress_queue.put({"type": "commentary", "text": "Obsolete"})
    task = asyncio.create_task(run_progress_card(ctx, adapter))
    await finish_progress_card(ctx, task)
    adapter.send_progress_card.assert_not_called()
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_new_cancellation_while_flushing_is_not_swallowed():
    ctx = context()
    entered = asyncio.Event()
    async def card(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    adapter = SimpleNamespace(send_progress_card=card, send=AsyncMock())
    ctx.progress_queue.put({"type": "commentary", "text": "Activity"})
    task = asyncio.create_task(run_progress_card(ctx, adapter))
    await entered.wait()
    finisher = asyncio.create_task(finish_progress_card(ctx, task))
    await asyncio.sleep(0)
    finisher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finisher
    await asyncio.sleep(0)
    if not task.done():
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_consecutive_turns_share_session_but_not_card_identity():
    calls = []
    async def card(chat, snapshot, **kwargs):
        calls.append(kwargs)
        return SendResult(success=True, message_id=f"card-{len(calls)}")
    adapter = SimpleNamespace(send_progress_card=card, send=AsyncMock())
    for _ in range(2):
        ctx = context()
        ctx.progress_queue.put({"type": "commentary", "text": "New turn"})
        task = asyncio.create_task(run_progress_card(ctx, adapter))
        await finish_progress_card(ctx, task)
    assert len(calls) == 2
    assert all(call["message_id"] is None for call in calls)
