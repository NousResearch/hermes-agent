"""Tests for the optional systemd event-loop watchdog protocol."""

from __future__ import annotations

import asyncio
import os
import socket

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


def test_startup_deadline_rejects_mismatched_watchdog_pid(monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() + 1))

    from gateway.systemd_notify import SystemdStartupDeadline

    assert SystemdStartupDeadline().enabled is False


@pytest.mark.asyncio
async def test_startup_deadline_repeats_until_stopped(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    deadline = notify_mod.SystemdStartupDeadline(
        interval_seconds=0.01, extend_seconds=1
    )

    assert deadline.start() is True
    await asyncio.sleep(0.025)
    await deadline.stop()
    count_after_stop = len(calls)
    await asyncio.sleep(0.02)

    assert calls.count("EXTEND_TIMEOUT_USEC=1000000") >= 2
    assert len(calls) == count_after_stop
    assert deadline.task is None


@pytest.mark.asyncio
async def test_startup_deadline_is_inert_when_config_disabled(monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")

    from gateway.systemd_notify import SystemdStartupDeadline

    deadline = SystemdStartupDeadline(config_enabled=False)
    assert deadline.enabled is False
    assert deadline.start() is False
    await deadline.stop()


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


