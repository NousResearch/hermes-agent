"""Tests for the optional systemd event-loop watchdog protocol."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="Unix datagram sockets are unavailable"
)
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


def test_watchdog_recovers_after_loop_progress_is_late(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=0.1)

    assert watchdog.record_tick(scheduled_at=10.0, now=10.05) is True
    assert calls == ["WATCHDOG=1"]
    assert watchdog.record_tick(scheduled_at=10.0, now=10.2) is True
    assert watchdog.unhealthy is True
    assert "WATCHDOG=1" in calls[-1]
    assert "STATUS=watchdog degraded" in calls[-1]

    # A recovered event loop must keep feeding systemd.  Require two timely
    # samples before clearing the degraded status so one lucky wake-up does not
    # hide recurring starvation.
    assert watchdog.record_tick(scheduled_at=10.3, now=10.35) is True
    assert watchdog.unhealthy is True
    assert watchdog.record_tick(scheduled_at=10.4, now=10.45) is True
    assert watchdog.unhealthy is False
    assert "STATUS=watchdog healthy" in calls[-1]


@pytest.mark.asyncio
async def test_watchdog_task_survives_a_transient_event_loop_stall(monkeypatch):
    calls: list[str] = []
    degraded = asyncio.Event()
    healthy = asyncio.Event()
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "100000")

    import gateway.systemd_notify as notify_mod

    def _capture(message: str) -> bool:
        calls.append(message)
        if "STATUS=watchdog degraded" in message:
            degraded.set()
        if "STATUS=watchdog healthy" in message:
            healthy.set()
        return True

    monkeypatch.setattr(notify_mod, "notify", _capture)
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=0.01)

    assert watchdog.start() is True
    try:
        await asyncio.sleep(0)  # Let the watchdog establish its first deadline.
        time.sleep(0.08)  # Fault injection: delay the loop, but not past WatchdogSec.

        await asyncio.wait_for(degraded.wait(), timeout=2.0)
        await asyncio.wait_for(healthy.wait(), timeout=2.0)
        assert watchdog.task is not None
        assert not watchdog.task.done()
        assert calls.count("WATCHDOG=1") >= 1
    finally:
        await watchdog.stop()


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
