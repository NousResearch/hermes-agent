"""Tests for the optional systemd event-loop watchdog protocol."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any, cast

import pytest

pytestmark = pytest.mark.platforms("linux")


# abstract (NUL-prefixed) AF_UNIX names are a Linux kernel feature
def test_notify_supports_systemd_abstract_socket(monkeypatch):
    name = "\0hermes-test-notify"
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(name)
    receiver.settimeout(1.0)
    monkeypatch.setenv("NOTIFY_SOCKET", "@hermes-test-notify")

    try:
        from gateway.systemd_notify import notify

        assert notify("WATCHDOG=1") is True
        assert receiver.recv(4096) == b"WATCHDOG=1"
    finally:
        receiver.close()


def test_notify_uses_nonblocking_datagram_send(monkeypatch):
    calls: list[object] = []

    class _Sender:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def setblocking(self, value):
            calls.append(("setblocking", value))

        def connect(self, address):
            calls.append(("connect", address))

        def send(self, payload):
            calls.append(("send", payload))

    import gateway.systemd_notify as notify_mod

    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setattr(notify_mod.socket, "socket", lambda *_args: _Sender())

    assert notify_mod.notify("READY=1") is True
    assert calls[0] == ("setblocking", False)


@pytest.mark.asyncio
async def test_watchdog_sends_ready_heartbeat_and_stopping(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "20000")

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=1.0)

    assert watchdog.start() is True
    assert watchdog.ready("Gateway running") is True
    await asyncio.sleep(0.04)
    await watchdog.stop()

    assert any(message.startswith("READY=1") for message in calls)
    assert "WATCHDOG=1" in calls
    assert calls[-1] == "STOPPING=1"
    assert watchdog.unhealthy is False


def _watchdog(monkeypatch, **kwargs):
    """Build a SystemdWatchdog with a stubbed notify() and a fixed 20s interval."""
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "20000000")  # 20s, in microseconds

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    return notify_mod.SystemdWatchdog(**kwargs), calls


def test_single_late_tick_does_not_condemn_the_gateway(monkeypatch):
    """A transient stall (MCP discovery, a long cron turn) must not stop the heartbeat.

    One late tick used to mark the watchdog unhealthy immediately, so systemd would
    SIGABRT a perfectly healthy gateway — trading a transient stall for a restart
    loop (kanban t_b3e3f2cb).
    """
    watchdog, calls = _watchdog(monkeypatch, lag_tolerance_seconds=1.0, max_lag_seconds=60.0)

    assert watchdog.record_tick(scheduled_at=100.0, now=101.5) is True
    assert watchdog.unhealthy is False
    assert "WATCHDOG=1" in calls


def test_sustained_lag_condemns_the_gateway(monkeypatch):
    """Cumulative lateness beyond the grace budget stops feeding systemd."""
    watchdog, calls = _watchdog(monkeypatch, lag_tolerance_seconds=1.0, max_lag_seconds=10.0)

    # 6s late, twice: 12s cumulative >= 10s budget.
    assert watchdog.record_tick(scheduled_at=100.0, now=106.0) is True
    assert watchdog.unhealthy is False
    assert watchdog.record_tick(scheduled_at=200.0, now=206.0) is False
    assert watchdog.unhealthy is True
    assert any(message.startswith("STATUS=") for message in calls)


def test_on_time_tick_resets_the_lag_budget(monkeypatch):
    """A healthy tick clears accumulated lateness: only CONSECUTIVE stalls count."""
    watchdog, _calls = _watchdog(monkeypatch, lag_tolerance_seconds=1.0, max_lag_seconds=10.0)

    assert watchdog.record_tick(scheduled_at=100.0, now=106.0) is True
    assert watchdog.record_tick(scheduled_at=200.0, now=200.1) is True  # on time
    assert watchdog.record_tick(scheduled_at=300.0, now=306.0) is True  # budget reset
    assert watchdog.unhealthy is False


def test_negative_lag_is_not_treated_as_a_wedge(monkeypatch):
    """A backwards clock must not accumulate lateness."""
    watchdog, _calls = _watchdog(monkeypatch, lag_tolerance_seconds=1.0, max_lag_seconds=10.0)

    for _ in range(5):
        assert watchdog.record_tick(scheduled_at=100.0, now=50.0) is True
    assert watchdog.unhealthy is False


def test_max_lag_defaults_to_the_watchdog_interval(monkeypatch):
    """Without an explicit budget, the grace window is one whole watchdog period."""
    watchdog, _calls = _watchdog(monkeypatch, lag_tolerance_seconds=1.0)

    assert watchdog._max_lag() == 20.0


# ---------------------------------------------------------------------------
# Off-loop feeder — regression for the 2026-11-03 00:12:07 UTC gateway stall.
#
# systemd's ``WatchdogSec`` fired while the gateway's main thread was in
# ``Garbage-collecting``. The heartbeat that silences systemd was produced by an
# ``asyncio`` task on that same main event loop, so a main-thread stall silenced the
# watchdog and systemd SIGABRTed a gateway that was about to recover on its own.
# ---------------------------------------------------------------------------


def _fast_watchdog(monkeypatch, *, interval_usec: int = 400000, **kwargs):
    """A watchdog whose systemd interval is short enough to observe beats in a test."""
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", str(interval_usec))

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    return notify_mod.SystemdWatchdog(**kwargs), calls


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.asyncio
async def test_feeder_thread_beats_without_the_event_loop_running(monkeypatch):
    """The systemd heartbeat must not depend on the event loop making progress.

    This is the regression in one assertion: the heartbeat is produced by a thread,
    so a gateway whose loop is busy (or whose main thread is in a long GC pass) still
    tells systemd it is alive.
    """
    watchdog, calls = _fast_watchdog(monkeypatch, interval_usec=200000)

    assert watchdog.start() is True
    try:
        assert _wait_until(lambda: calls.count("WATCHDOG=1") >= 2), calls
        assert watchdog.feeder_alive is True
        assert watchdog.unhealthy is False
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_main_thread_stall_does_not_silence_the_heartbeat(monkeypatch):
    """A main-thread stall must not stop systemd's heartbeat.

    The 2026-11-03 failure: the main thread was in ``Garbage-collecting`` and the
    loop-driven watchdog task could not run, so ``WATCHDOG=1`` stopped and systemd
    SIGABRTed the gateway. Here the main thread is held still while the feeder
    thread must keep beating.
    """
    watchdog, calls = _fast_watchdog(monkeypatch, interval_usec=200000)

    assert watchdog.start() is True
    try:
        assert _wait_until(lambda: calls.count("WATCHDOG=1") >= 1)
        beats_before = calls.count("WATCHDOG=1")

        # Hold the MAIN thread so the event loop cannot run a single callback —
        # the same shape as a long GC pass in the main thread.
        release = threading.Event()
        held = threading.Event()

        def _stall() -> None:
            held.set()
            release.wait(timeout=10.0)

        stall = threading.Thread(target=_stall, daemon=True)
        stall.start()
        assert held.wait(timeout=5.0)

        # The loop is now blocked; the feeder thread must still beat.
        assert _wait_until(lambda: calls.count("WATCHDOG=1") > beats_before, timeout=5.0), calls
        assert watchdog.unhealthy is False
        release.set()
        stall.join(timeout=5.0)
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_stop_halts_the_feeder_thread(monkeypatch):
    """``stop()`` must end the feeder thread so a restart never double-beats."""
    watchdog, calls = _fast_watchdog(monkeypatch, interval_usec=200000)

    assert watchdog.start() is True
    assert _wait_until(lambda: calls.count("WATCHDOG=1") >= 1)

    await watchdog.stop()

    assert watchdog.feeder_alive is False
    settled = calls.count("WATCHDOG=1")
    time.sleep(0.3)
    assert calls.count("WATCHDOG=1") == settled
    assert calls[-1] == "STOPPING=1"


@pytest.mark.asyncio
async def test_feeder_stops_beating_when_the_process_cannot_run_it(monkeypatch):
    """The systemd backstop must survive: a process that cannot run ANY Python thread
    stops beating, and systemd then reclaims the unit.

    This is the case the systemd watchdog exists for, and the off-loop feeder must not
    weaken it. A thread blocked on a C call that holds the GIL cannot run the feeder
    either, so the beat stops exactly as before the change.
    """
    watchdog, calls = _fast_watchdog(monkeypatch, interval_usec=200000)
    assert watchdog.start() is True
    assert _wait_until(lambda: calls.count("WATCHDOG=1") >= 1)

    # Simulate the process being unable to make progress: the feeder cannot run.
    watchdog._feeder_stop.set()
    settled = calls.count("WATCHDOG=1")
    time.sleep(0.3)

    assert calls.count("WATCHDOG=1") == settled
    await watchdog.stop()


def test_loop_liveness_watchdog_is_the_loop_wedge_detector(monkeypatch):
    """Loop-level wedge detection must remain owned by the off-loop loop-liveness
    watchdog, which the systemd heartbeat no longer duplicates.

    The systemd heartbeat is a PROCESS liveness contract; the loop-liveness watchdog
    is the LOOP contract. Moving the heartbeat off the loop must not leave loop
    wedges undetected, so the detector must still exist, must probe off the loop, and
    must keep probing across misses.
    """
    from gateway import shutdown_watchdog

    assert shutdown_watchdog.DEFAULT_LOOP_WATCHDOG_MAX_STRIKES >= 1
    assert shutdown_watchdog.DEFAULT_LOOP_WATCHDOG_TIMEOUT_S > 0

    scheduled: list[str] = []

    class _WedgedLoop:
        """A loop that accepts ``call_soon_threadsafe`` but never runs the callback —
        i.e. a wedged loop."""

        def call_soon_threadsafe(self, callback, *args):
            scheduled.append("scheduled")
            return None

    handle = shutdown_watchdog.start_loop_liveness_watchdog(
        cast(Any, _WedgedLoop()),
        probe_interval=0.05,
        probe_timeout=0.05,
        max_strikes=2,
        exit_code=75,
    )
    assert handle is not None
    try:
        # The detector probes off the loop and keeps probing across misses.
        assert _wait_until(lambda: len(scheduled) >= 2, timeout=5.0), scheduled
    finally:
        handle.stop()
        handle.join(timeout=5.0)


