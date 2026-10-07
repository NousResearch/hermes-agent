"""Native Telegram stop controls must only affect the draft's owning turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="123:test"))


def stop_update(chat_id=123, draft_id=9, thread_id=None):
    stopped = {"chat": {"id": chat_id, "type": "private"}, "draft_id": draft_id}
    if thread_id is not None:
        stopped["message_thread_id"] = thread_id
    return SimpleNamespace(api_kwargs={"stopped_message_generation": stopped})


@pytest.mark.asyncio
async def test_native_stop_cancels_current_draft_once():
    tg = adapter()
    cancel = AsyncMock()
    tg.bind_generation_control("123", 9, None, cancel, lambda: True)
    await tg._handle_generation_stopped(stop_update(), None)
    await tg._handle_generation_stopped(stop_update(), None)
    cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_late_stop_cannot_cancel_successor_turn():
    tg = adapter()
    old_cancel, new_cancel = AsyncMock(), AsyncMock()
    tg.bind_generation_control("123", 9, None, old_cancel, lambda: True)
    tg.bind_generation_control("123", 10, None, new_cancel, lambda: True)
    await tg._handle_generation_stopped(stop_update(draft_id=9), None)
    old_cancel.assert_not_awaited()
    new_cancel.assert_not_awaited()
    await tg._handle_generation_stopped(stop_update(draft_id=10), None)
    new_cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_stop_is_chat_thread_and_generation_scoped():
    tg = adapter()
    cancel = AsyncMock()
    current = [True]
    tg.bind_generation_control("123", 9, {"thread_id": "7"}, cancel, lambda: current[0])
    await tg._handle_generation_stopped(stop_update(chat_id=124, thread_id=7), None)
    await tg._handle_generation_stopped(stop_update(thread_id=8), None)
    cancel.assert_not_awaited()
    current[0] = False
    await tg._handle_generation_stopped(stop_update(thread_id=7), None)
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_finishing_old_draft_does_not_unbind_successor():
    tg = adapter()
    cancel = AsyncMock()
    tg.bind_generation_control("123", 9, None, AsyncMock(), lambda: True)
    tg.bind_generation_control("123", 10, None, cancel, lambda: True)
    tg.finish_generation_control("123", 9, None)
    await tg._handle_generation_stopped(stop_update(draft_id=10), None)
    cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_general_dm_topic_stop_matches_root_draft():
    tg = adapter()
    cancel = AsyncMock()
    tg.bind_generation_control("123", 9, {"thread_id": "1"}, cancel, lambda: True)
    await tg._handle_generation_stopped(stop_update(), None)
    cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_native_stop_callback_does_not_interrupt_replacement():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    source = SimpleNamespace()
    key = "telegram:dm:123"
    old_generation = runner._begin_session_run_generation(key)
    stop = runner._generation_stop_callback(source, key, old_generation)
    runner._begin_session_run_generation(key)
    runner._interrupt_and_clear_session = AsyncMock()
    await stop()
    runner._interrupt_and_clear_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalizing_consumer_fences_stop_binding():
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    tg = adapter()
    consumer = GatewayStreamConsumer(
        tg, "123", StreamConsumerConfig(transport="draft", chat_type="private"),
        run_still_current=lambda: True, on_generation_stop=AsyncMock())
    consumer._use_draft_streaming = True
    consumer._bump_draft_id()
    control = tg._generation_controls[("123", None)]
    assert control.current() is True
    consumer._finalizing = True
    assert control.current() is False
