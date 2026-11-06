"""Offline initial Socket Mode connection starvation regression.

Drive the real adapter watchdog/start/stop/restart with a synthetic SDK client.
The virtual clock is scoped to the adapter, not asyncio's scheduling clock.
"""

import asyncio
from types import SimpleNamespace

import pytest

import plugins.platforms.slack.adapter as slack
from gateway.config import PlatformConfig


class _SyntheticClient:
    def __init__(self):
        self.current_session = None
        self.last_ping_pong_time = None
        self.ping_interval = 5.0
        self.web_client = SimpleNamespace(proxy=None)
        self.proxy = None
        self.closed = False

    async def is_connected(self):
        return self.current_session is not None and not self.current_session.closed


@pytest.fixture
def socket_rig(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        slack, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now)
    )
    monkeypatch.setattr(slack, "SLACK_AVAILABLE", True)
    handlers = []

    class Handler:
        def __init__(self, app, app_token, proxy=None):
            self.client = _SyntheticClient()
            self.entered = asyncio.Event()
            self.allow_connect = asyncio.Event()
            self.connected = asyncio.Event()
            self.finish = asyncio.Event()
            self.cancelled = False
            self.task_at_close = None
            handlers.append(self)

        async def start_async(self):
            self.entered.set()
            try:
                await self.allow_connect.wait()
                self.client.current_session = SimpleNamespace(closed=False)
                self.client.last_ping_pong_time = clock.now
                self.connected.set()
                await self.finish.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def close_async(self):
            self.task_at_close = self.cancelled
            self.client.closed = True
            if self.client.current_session is not None:
                self.client.current_session.closed = True

    monkeypatch.setattr(slack, "AsyncSocketModeHandler", Handler)
    adapter = slack.SlackAdapter(PlatformConfig(enabled=True, token="xoxb-synthetic"))
    adapter._app = object()
    adapter._app_token = "xapp-synthetic"
    adapter._proxy_url = None
    adapter._running = True
    return SimpleNamespace(adapter=adapter, handlers=handlers, clock=clock)


async def _watchdog_ticks(rig, monkeypatch, elapsed_times, before_tick=None):
    """Run the actual loop, yielding to actual SDK tasks at deterministic ticks."""
    adapter = rig.adapter
    tick_times = iter(elapsed_times)
    real_sleep = asyncio.sleep

    async def tick(_interval):
        try:
            elapsed = next(tick_times)
        except StopIteration:
            adapter._running = False
            return
        rig.clock.now = 1000.0 + elapsed
        if before_tick is not None:
            await before_tick(elapsed)
        await real_sleep(0)

    # Do not monkeypatch the shared asyncio module or event-loop monotonic clock.
    with monkeypatch.context() as scope:
        scope.setattr(
            slack, "asyncio",
            SimpleNamespace(
                sleep=tick, CancelledError=asyncio.CancelledError,
                iscoroutine=asyncio.iscoroutine, create_task=asyncio.create_task,
                wait=asyncio.wait,
            ),
        )
        await adapter._socket_watchdog_loop()


@pytest.mark.asyncio
async def test_slow_first_connection_survives_early_watchdog_ticks(socket_rig, monkeypatch):
    rig = socket_rig
    adapter = rig.adapter
    adapter._start_socket_mode_handler()
    original = rig.handlers[0]
    task = adapter._socket_mode_task
    await original.entered.wait()

    async def finish_at_45_seconds(elapsed):
        if elapsed == 45.0:
            original.allow_connect.set()
            # Never wait on a handler the watchdog has already cancelled.
            if not task.done():
                await original.connected.wait()

    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0, 30.0, 45.0], finish_at_45_seconds)
        assert rig.handlers == [original], "watchdog cancelled pending initial connection before grace"
        assert original.connected.is_set(), "slow first connection never got to complete"
        assert not original.cancelled
        assert adapter._socket_mode_task is task and not task.done()
    finally:
        adapter._running = False
        await adapter._stop_socket_mode_handler()
    assert task.done()
    assert original.task_at_close is True


@pytest.mark.asyncio
async def test_never_connected_client_restarts_at_existing_grace_bound(socket_rig, monkeypatch):
    rig = socket_rig
    adapter = rig.adapter
    adapter._start_socket_mode_handler()
    original = rig.handlers[0]
    task = adapter._socket_mode_task
    await original.entered.wait()
    observations = []

    async def observe(elapsed):
        observations.append((elapsed, len(rig.handlers), task.done()))

    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0, 30.0, 45.0, 59.999, 60.0], observe)
        assert observations == [(t, 1, False) for t in [15.0, 30.0, 45.0, 59.999, 60.0]]
        assert len(rig.handlers) == 2, "never-connected SDK must not wait indefinitely"
        assert task.done() and original.cancelled
        assert original.task_at_close is True
        assert adapter._socket_handler_started_monotonic == 1060.0
    finally:
        adapter._running = False
        await adapter._stop_socket_mode_handler()


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_state", ["closed-session", "session-cleared", "no-first-ping"])
async def test_established_dropped_socket_restarts_without_initial_grace(
    socket_rig, monkeypatch, drop_state
):
    rig = socket_rig
    adapter = rig.adapter
    adapter._start_socket_mode_handler()
    original = rig.handlers[0]
    task = adapter._socket_mode_task
    original.allow_connect.set()
    await original.connected.wait()
    if drop_state == "session-cleared":
        original.client.current_session = None
    else:
        original.client.current_session.closed = True
    if drop_state == "no-first-ping":
        original.client.last_ping_pong_time = None
    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0])
        assert len(rig.handlers) == 2, "initial grace masked an established dropped socket"
        assert task.done() and original.cancelled and original.task_at_close
    finally:
        adapter._running = False
        await adapter._stop_socket_mode_handler()


@pytest.mark.asyncio
async def test_missing_socket_task_recovers_even_during_initial_grace(socket_rig, monkeypatch):
    rig = socket_rig
    assert rig.adapter._socket_mode_task is None
    rig.adapter._socket_handler_started_monotonic = 1000.0
    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0])
        assert len(rig.handlers) == 1
        assert rig.adapter._socket_mode_task is not None
        assert not rig.adapter._socket_mode_task.done()
    finally:
        rig.adapter._running = False
        await rig.adapter._stop_socket_mode_handler()


@pytest.mark.asyncio
async def test_dead_socket_task_recovers_even_during_initial_grace(socket_rig, monkeypatch):
    rig = socket_rig
    rig.adapter._start_socket_mode_handler()
    original = rig.handlers[0]
    task = rig.adapter._socket_mode_task
    await original.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0])
        assert len(rig.handlers) == 2
        assert rig.adapter._socket_mode_task is not task
    finally:
        rig.adapter._running = False
        await rig.adapter._stop_socket_mode_handler()


@pytest.mark.asyncio
async def test_unknown_start_timestamp_does_not_suppress_recovery(socket_rig, monkeypatch):
    rig = socket_rig
    rig.adapter._start_socket_mode_handler()
    rig.adapter._socket_handler_started_monotonic = None
    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0])
        assert len(rig.handlers) == 2
    finally:
        rig.adapter._running = False
        await rig.adapter._stop_socket_mode_handler()


@pytest.mark.asyncio
async def test_connected_stale_ping_still_recovers_during_initial_grace(socket_rig, monkeypatch):
    rig = socket_rig
    rig.adapter._start_socket_mode_handler()
    original = rig.handlers[0]
    original.allow_connect.set()
    await original.connected.wait()
    original.client.last_ping_pong_time = 900.0
    try:
        await _watchdog_ticks(rig, monkeypatch, [15.0])
        assert len(rig.handlers) == 2
    finally:
        rig.adapter._running = False
        await rig.adapter._stop_socket_mode_handler()
