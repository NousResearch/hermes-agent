"""A wake admitted before a restart drain begins must still run.

Its producer was told the wake was delivered and will not retry it, so the drain gate and
the pending follow-up discard must not drop it (#138123).
"""

import asyncio

import pytest

from gateway.platforms.event import MessageEvent
from gateway.wake import deliver_wake
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

WAKE = "[kanban] Task t_1 completed."


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
        await asyncio.sleep(0)


def wake_route():
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    runner._delivery_adapter_for = lambda source: adapter
    turns = []

    async def agent_turn(event, source, quick_key, run_generation):
        turns.append(event.text)

    runner._handle_message_with_agent = agent_turn
    return runner, adapter, make_restart_source(), turns


@pytest.mark.asyncio
async def test_wake_claimed_before_drain_runs():
    runner, adapter, source, turns = wake_route()
    await deliver_wake(adapter, text=WAKE, source=source)
    runner._draining = True  # /restart begins before the claimed turn reaches the drain gate
    await drain(adapter)
    assert turns == [WAKE]
    assert adapter.sent == []  # no drain notice for work the gateway already accepted


@pytest.mark.asyncio
async def test_wake_queued_before_drain_survives_the_pending_discard():
    runner, adapter, source, _turns = wake_route()
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    adapter._active_sessions[key] = asyncio.Event()  # a turn is running in this chat
    await deliver_wake(adapter, text=WAKE, source=source)
    runner._draining = True  # /restart begins while the wake waits behind that turn
    event, text = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key
    )
    assert text == WAKE and event.internal  # runs next instead of being discarded

    runner._enqueue_fifo(key, MessageEvent(text="a human follow-up", source=source), adapter)
    event, text = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key
    )
    assert (event, text) == (None, None)  # other follow-ups are still discarded
