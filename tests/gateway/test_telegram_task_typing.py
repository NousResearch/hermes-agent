"""Typing follows active work, not the previous stream or a sibling topic."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_topic_pause_does_not_silence_other_work():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(send_chat_action=AsyncMock())
    other_tick = asyncio.Event()
    resumed_tick = asyncio.Event()
    calls = []

    async def action(**kwargs):
        topic = kwargs.get("message_thread_id")
        calls.append(topic)
        (resumed_tick if topic == 10 else other_tick).set()

    adapter._bot.send_chat_action.side_effect = action
    runner = object.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="10")
    scfg = SimpleNamespace(cursor="", edit_interval=0.01, buffer_threshold=1,
                           fresh_final_after_seconds=0, transport="edit")
    _, finalize = runner._build_stream_consumer_config(source, scfg, adapter, on_missing_cursor="raise")
    finalize()
    stop = asyncio.Event()
    tasks = [asyncio.create_task(adapter._keep_typing(
        "123", interval=0.01, metadata={"thread_id": str(topic)}, stop_event=stop
    )) for topic in (10, 20)]
    try:
        await asyncio.wait_for(other_tick.wait(), 5)
        assert 10 not in calls
        adapter.resume_typing_for_chat("123", metadata={"thread_id": "10"})
        await asyncio.wait_for(resumed_tick.wait(), 5)
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
    settled = len(calls)
    await asyncio.sleep(0.05)
    assert len(calls) == settled


@pytest.mark.asyncio
async def test_queued_work_rearms_typing_after_stream_finalization():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(send_chat_action=AsyncMock())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="10")
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._refresh_agent_cache_message_count = AsyncMock()
    ctx = TurnContext(source=source, session_key="s", session_id="session")
    ctx._status_thread_metadata = {"thread_id": "10"}
    ctx.history = []
    ctx.context_prompt = ""
    ctx._interrupt_depth = 0
    stop = asyncio.Event()
    adapter._active_sessions["s"] = stop
    repeated = asyncio.Event()
    calls = []

    async def action(**kwargs):
        calls.append(kwargs)
        # A single send at handoff is insufficient: the regular refresher must run.
        if len(calls) >= 3:
            repeated.set()

    adapter._bot.send_chat_action.side_effect = action
    scfg = SimpleNamespace(cursor="", edit_interval=0.01, buffer_threshold=1,
                           fresh_final_after_seconds=0, transport="edit")
    _, before_finalize = runner._build_stream_consumer_config(source, scfg, adapter, on_missing_cursor="raise")

    async def deliver(*args):
        before_finalize()

    runner._run_agent_deliver_first_response = deliver

    async def followup(**kwargs):
        await asyncio.wait_for(repeated.wait(), 2)
        return {"final_response": "done", "messages": []}

    runner._run_agent = followup
    task = asyncio.create_task(adapter._keep_typing("123", interval=0.01,
                              metadata=ctx._status_thread_metadata, stop_event=stop))
    try:
        await runner._run_agent_queued_followup(ctx, adapter, "continue", None, "first",
                                              {"messages": []}, None)
        assert all(call["message_thread_id"] == 10 for call in calls)
    finally:
        stop.set()
        await asyncio.wait_for(task, 5)
