"""A burst of process notifications must not use up the busy queue slots human follow-ups need."""
import asyncio

import pytest
import pytest_asyncio

from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.wake import WakeNotAccepted, admit_internal_event


def process_wake(source, i):
    return MessageEvent(text=f"wake {i}", source=source, internal=True,
                        metadata={"notification_origin": "process_registry_synthetic"})


@pytest_asyncio.fixture
async def busy_route():
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: adapter
    source = SessionSource(platform=adapter.platform, chat_id="42", user_id="42", chat_type="dm")
    key = build_session_key(source)
    release, started = asyncio.Event(), asyncio.Event()

    async def handler(event):
        started.set()
        await release.wait()

    adapter.set_message_handler(handler)
    await adapter.connect()
    await adapter.handle_message(MessageEvent(text="long turn", source=source))
    await asyncio.wait_for(started.wait(), 2)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    yield runner, adapter, source, key
    release.set()
    adapter._pending_messages.clear()
    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    await adapter.disconnect()


async def admit_burst(adapter, source, count):
    wakes = [process_wake(source, i) for i in range(count)]
    refused = []
    for wake in wakes:
        try:
            await admit_internal_event(adapter, wake)
        except WakeNotAccepted:
            refused.append(wake)
    return wakes, refused


@pytest.mark.asyncio
async def test_wake_burst_leaves_room_for_human_follow_ups(busy_route):
    runner, adapter, source, key = busy_route
    runner._BUSY_QUEUE_MAX_PENDING = 32
    wakes, refused = await admit_burst(adapter, source, 40)
    assert runner._queue_depth(key, adapter=adapter) == 24
    assert refused == wakes[24:]
    assert all(wake._gateway_accepted is False for wake in refused)

    humans = [MessageEvent(text=f"human {i}", source=source) for i in range(9)]
    for human in humans:
        runner._queue_or_replace_pending_event(key, human)
    assert [human._gateway_accepted for human in humans] == [True] * 8 + [False]
    assert runner._queue_depth(key, adapter=adapter) == 32


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 3])
async def test_tiny_cap_still_admits_wakes(busy_route, cap):
    runner, adapter, source, key = busy_route
    runner._BUSY_QUEUE_MAX_PENDING = cap
    _wakes, refused = await admit_burst(adapter, source, cap)
    assert refused == []
    assert runner._queue_depth(key, adapter=adapter) == cap
