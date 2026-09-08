"""Heartbeat polls wake idle adapters, never accumulate deferred ticks."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli.heartbeat import HeartbeatManager


class _HeartbeatAdapter(BasePlatformAdapter):
    """Real adapter lifecycle with an in-memory wire transport, no bot/model."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="wire-1")


@pytest.fixture
def poller(monkeypatch):
    from hermes_cli import goals, heartbeat

    goals._DB_CACHE.clear()
    goals._get_session_db()
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(heartbeat, "time", SimpleNamespace(time=lambda: clock.now))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    key = build_session_key(source)
    adapter = _HeartbeatAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._adapter_for_source = lambda source: adapter
    runner._run_in_executor_with_context = asyncio.to_thread
    watch = {key: (source, "heartbeat-session")}
    HeartbeatManager("heartbeat-session").set("check status", 60)
    yield runner, adapter, watch, key, clock
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_idle_wake_coalesces_intervals_while_adapter_owns_turn(poller):
    runner, adapter, watch, key, clock = poller
    started, release = asyncio.Event(), asyncio.Event()
    received = []

    async def handler(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary
        received.append(event)
        started.set()
        await release.wait()
        return None

    adapter.set_message_handler(handler)
    # A remembered topic must not redirect the watched session or introduce an
    # executor yield between the idle check and the adapter's session claim.
    adapter._topic_recovery_fn = lambda source: "another-topic"
    try:
        clock.now += 60
        await runner._heartbeat_poll_once(watch)
        assert key in adapter._active_sessions, "idle poll must start a turn, not park it in FIFO"
        await asyncio.wait_for(started.wait(), 2)
        for _ in range(14):
            clock.now += 60
            await runner._heartbeat_poll_once(watch)
        assert len(received) == 1
        assert not received[0].internal  # authorization and emergency-stop still apply
        assert runner._queue_depth(key, adapter=adapter) == 0
        assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    finally:
        release.set()
        await asyncio.gather(*adapter._background_tasks)


@pytest.mark.asyncio
async def test_unavailable_or_busy_session_leaves_persisted_tick_due(poller):
    runner, adapter, watch, key, clock = poller
    clock.now += 900
    original = HeartbeatManager("heartbeat-session").state.to_json()
    user = MessageEvent(text="user follow-up", source=watch[key][0])
    # Missing adapter and missing handler must not consume a due tick.
    runner._adapter_for_source = lambda source: None
    await runner._heartbeat_poll_once(watch)
    runner._adapter_for_source = lambda source: adapter
    await runner._heartbeat_poll_once(watch)

    async def handler(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary
        return None

    adapter.set_message_handler(handler)
    for state in (runner._running_agents, adapter._active_sessions, adapter._pending_messages):
        state[key] = user
        await runner._heartbeat_poll_once(watch)
        assert state[key] is user
        state.pop(key)
    assert HeartbeatManager("heartbeat-session").state.to_json() == original
    await runner._heartbeat_poll_once(watch)
    assert key in adapter._active_sessions
    await asyncio.gather(*adapter._background_tasks)
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    assert runner._queue_depth(key, adapter=adapter) == 0
