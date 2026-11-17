"""Regression for #138193: a task-card change outside the visible window still reaches the adapter."""

import asyncio
import queue
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner
from tests.gateway.test_run_progress_topics import NativeTaskCardAdapter

CALLS = [f"call-{n}" for n in range(9)]


def parallel_batch_events():
    # call-0 finishes after eight newer calls started, so it is outside the window by then.
    return [
        {"type": event_type, "tool_call_id": call_id, "tool_name": "web_search"}
        for event_type in ("tool.started", "tool.completed")
        for call_id in CALLS
    ]


def card_lane(events, run_still_current):
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1"), _progress_reply_to=None, _progress_metadata={},
        progress_queue=events, _run_still_current=run_still_current, agent_holder=[None],
    )
    return TurnRunner(None, ctx)


def final_statuses(adapter):
    statuses = {}
    for update in adapter.native_updates:
        for task in update["tasks"]:
            statuses[task["id"]] = task["status"]
    return statuses


@pytest.mark.asyncio
async def test_change_published_event_by_event_reaches_adapter():
    events = queue.Queue()
    for event in parallel_batch_events():
        events.put(event)
    adapter = NativeTaskCardAdapter()

    await card_lane(events, lambda: not events.empty())._send_native_task_card_progress(adapter)

    assert final_statuses(adapter) == {call_id: "complete" for call_id in CALLS}


@pytest.mark.asyncio
async def test_change_drained_at_turn_end_reaches_adapter():
    events = queue.Queue()
    adapter = NativeTaskCardAdapter()
    lane = asyncio.create_task(card_lane(events, lambda: True)._send_native_task_card_progress(adapter))
    # The lane is now polling the empty queue; cancelling it is the turn end, which drains
    # everything queued since into a single publish.
    await asyncio.sleep(0)
    for event in parallel_batch_events():
        events.put(event)
    lane.cancel()
    await asyncio.gather(lane, return_exceptions=True)

    assert final_statuses(adapter) == {call_id: "complete" for call_id in CALLS}
