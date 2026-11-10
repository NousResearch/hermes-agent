"""Turn/lane ownership and concurrent Telegram status updates (#92210).

Import the real gateway, adapter and Telegram SDK; only Bot API I/O is fake.
"""

import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def status_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", typing_indicator=False))
    adapter._telegram_chat_outbound_slot_secs = 0
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=[SimpleNamespace(message_id=i) for i in range(100, 120)]),
        edit_message_text=AsyncMock(),
    )
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("other_lane,other_generation,other_event,thread", [
    ("topic-a", 2, "lifecycle", "1001"),
    ("topic-b", 1, "lifecycle", "1002"),
    ("topic-a", 1, "model-switch", "1001"),
])
async def test_gateway_status_belongs_to_turn_lane_and_event(
    status_adapter, monkeypatch, other_lane, other_generation, other_event, thread,
):
    pending = []
    monkeypatch.setattr(TurnRunner, "_schedule", lambda self, coro, message: pending.append(coro))

    async def emit(lane, generation, event="lifecycle", thread_id="1001"):
        ctx = TurnContext(
            source=SimpleNamespace(platform=Platform.TELEGRAM),
            session_key=lane, run_generation=generation, _run_still_current=lambda: True,
            _status_adapter=status_adapter, _status_chat_id="-100123",
            _status_thread_metadata={"thread_id": thread_id}, user_config={},
        )
        TurnRunner(MagicMock(), ctx)._status_callback_sync(event, "Recalling memories")
        assert len(pending) == 1
        assert (await pending.pop()).success

    bot = status_adapter._bot
    await emit("topic-a", 1)
    await emit("topic-a", 1)
    assert bot.send_message.await_count == 1
    assert bot.edit_message_text.await_args.kwargs["message_id"] == 100
    await emit(other_lane, other_generation, other_event, thread)
    assert bot.send_message.await_count == 2
    assert bot.send_message.await_args.kwargs["message_thread_id"] == int(thread)
    await emit(other_lane, other_generation, other_event, thread)
    assert bot.edit_message_text.await_args.kwargs["message_id"] == 101
    # Incomplete identities retain the legacy event-type fallback, including generation zero.
    await emit(None, 1)
    await emit("topic-a", None)
    assert bot.send_message.await_count == 3
    assert bot.edit_message_text.await_args.kwargs["message_id"] == 102
    await emit("topic-a", 0)
    assert bot.send_message.await_count == 4


@pytest.mark.asyncio
async def test_same_key_updates_serialize_through_gc_and_recover_failed_edit(status_adapter):
    adapter, bot = status_adapter, status_adapter._bot
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(message_id=100)

    bot.send_message.side_effect = delayed_send
    tasks = [asyncio.create_task(adapter.send_or_update_status("-100123", "turn", "first"))]
    try:
        await asyncio.wait_for(started.wait(), 5)
        tasks.append(asyncio.create_task(adapter.send_or_update_status("-100123", "turn", "second")))
        await asyncio.sleep(0)
        gc.collect()
        tasks.append(asyncio.create_task(adapter.send_or_update_status("-100123", "turn", "third")))
        await asyncio.sleep(0)
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert all(result.success for result in results)
    assert bot.send_message.await_count == 1
    assert bot.edit_message_text.await_count == 2
    assert all(call.kwargs["message_id"] == 100 for call in bot.edit_message_text.await_args_list)
    gc.collect()
    assert not adapter._status_locks
    # Permanent edit failure (including the Markdown fallback) recovers with a fresh send.
    bot.edit_message_text.side_effect = RuntimeError("message to edit not found")
    bot.send_message.side_effect = None
    bot.send_message.return_value = SimpleNamespace(message_id=101)
    result = await adapter.send_or_update_status("-100123", "turn", "recover")
    assert result.success and result.message_id == "101"
    bot.edit_message_text.side_effect = None
    await adapter.send_or_update_status("-100123", "turn", "recovered")
    assert bot.edit_message_text.await_args.kwargs["message_id"] == 101
    assert bot.send_message.await_count == 2


@pytest.mark.asyncio
async def test_pending_status_edit_does_not_resurrect_cross_key_eviction(status_adapter):
    adapter, bot = status_adapter, status_adapter._bot
    adapter._STATUS_MESSAGE_IDS_MAX = 4
    for index in range(4):
        await adapter.send_or_update_status("-100123", f"turn-{index}", "starting")
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_edit(**kwargs):
        started.set()
        await release.wait()

    bot.edit_message_text.side_effect = delayed_edit
    task = asyncio.create_task(adapter.send_or_update_status("-100123", "turn-0", "editing"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        for index in range(4, 6):
            await adapter.send_or_update_status("-100123", f"turn-{index}", "starting")
        assert ("-100123", "turn-0") not in adapter._status_message_ids
    finally:
        release.set()
        result = await task
    assert result.success
    assert ("-100123", "turn-0") not in adapter._status_message_ids
    assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
