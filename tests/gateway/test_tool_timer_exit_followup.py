"""Consumer-exit contracts for the native tool timer."""

from __future__ import annotations

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.tool_timer import _TIMER_TICK
from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["cancel", "error", "stale"])
async def test_consumer_exit_tears_down_timer_and_rejects_late_rearm(exit_mode):
    adapter = _make_native_streaming_adapter()
    type(adapter).SUPPORTS_TOOL_TIMER = True
    seeded = asyncio.Event()
    original_send_frame = adapter.send_stream_frame

    async def send_frame(*args, **kwargs):
        result = await original_send_frame(*args, **kwargs)
        seeded.set()
        return result

    adapter.send_stream_frame = send_frame
    run_state = "current"

    def run_still_current():
        if run_state == "error":
            raise RuntimeError("run ownership probe failed")
        return run_state == "current"

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(chat_type="dm", cursor=""),
        run_still_current=run_still_current,
    )
    armed = asyncio.Event()
    original_arm = consumer._arm_tool_timer

    def arm_timer():
        original_arm()
        armed.set()

    consumer._arm_tool_timer = arm_timer
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(seeded.wait(), timeout=2)
    consumer.on_tool_started("terminal", "call-1")
    await asyncio.wait_for(armed.wait(), timeout=2)
    assert consumer._tool_timer_handle is not None
    assert consumer._tool_start_times

    if exit_mode == "cancel":
        task.cancel()
    else:
        run_state = exit_mode
        consumer._queue.put(_TIMER_TICK)
    await asyncio.wait_for(task, timeout=2)

    assert consumer._tool_timer_handle is None
    assert consumer._tool_start_times == {}

    consumer.on_tool_started("late-tool", "late-call")
    consumer.on_tool_progress("late detail", "late-call")
    consumer.on_tool_completed("late-tool", 1.0, "late-call")
    consumer.on_llm_thinking()
    await asyncio.sleep(0)
    assert consumer._tool_timer_handle is None
    assert consumer._tool_start_times == {}
    assert consumer._queue.empty()
