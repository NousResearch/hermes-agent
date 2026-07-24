"""Tests for the optional systemd event-loop watchdog protocol."""

from __future__ import annotations

import asyncio
import socket

import pytest


def test_notify_without_notify_socket_is_a_noop(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    from gateway.systemd_notify import notify

    assert notify("READY=1") is False


def test_notify_sends_real_unix_datagram(tmp_path, monkeypatch):
    address = str(tmp_path / "notify.sock")
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(address)
    receiver.settimeout(1.0)
    monkeypatch.setenv("NOTIFY_SOCKET", address)

    from gateway.systemd_notify import notify

    assert notify("READY=1") is True
    assert receiver.recv(4096) == b"READY=1"
    receiver.close()


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


@pytest.mark.parametrize("raw", [None, "", "0", "-1", "not-a-number"])
def test_watchdog_interval_is_disabled_for_missing_invalid_or_nonpositive_values(
    monkeypatch, raw
):
    if raw is None:
        monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    else:
        monkeypatch.setenv("WATCHDOG_USEC", raw)
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify-does-not-exist")

    from gateway.systemd_notify import watchdog_interval_seconds

    assert watchdog_interval_seconds() is None


def test_watchdog_latches_when_loop_progress_is_late(monkeypatch):
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
    assert watchdog.record_tick(scheduled_at=10.0, now=10.2) is False
    assert watchdog.unhealthy is True
    assert calls[-1].startswith("STATUS=watchdog unhealthy")
    assert watchdog.record_tick(scheduled_at=10.0, now=10.3) is False


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


def test_watchdog_config_disabled_ignores_systemd_environment(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )
    watchdog = notify_mod.SystemdWatchdog(config_enabled=False)

    assert watchdog.enabled is False
    assert watchdog.start() is False
    assert watchdog.ready() is False
    assert calls == []
