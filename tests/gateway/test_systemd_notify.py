"""Tests for the optional systemd event-loop watchdog protocol."""

from __future__ import annotations

import asyncio
import socket

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


