"""Tests for gateway.systemd_notify — pure-Python sd_notify protocol.

Uses an AF_UNIX datagram socket in tmp_path as a stand-in for the
systemd notify socket. Verifies opt-in semantics (no-op when
``$NOTIFY_SOCKET`` is unset), abstract-namespace path handling, the
ready/watchdog/stopping helpers, and the heartbeat task's lifecycle.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys

import pytest


# ── Module-under-test loaded fresh per test to isolate env-var checks ─────


@pytest.fixture()
def sdn(monkeypatch: pytest.MonkeyPatch):
    """Reload gateway.systemd_notify with a clean env. Returns the module."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    import importlib

    import gateway.systemd_notify as systemd_notify

    importlib.reload(systemd_notify)
    return systemd_notify


@pytest.fixture()
def notify_socket(tmp_path):
    """Bind an AF_UNIX datagram listener and yield (path, sock) for assertions."""
    path = str(tmp_path / "notify.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(path)
    sock.settimeout(2.0)
    try:
        yield path, sock
    finally:
        sock.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def _is_posix() -> bool:
    return os.name == "posix"


# ── is_available / _notify_socket_path ─────────────────────────────────────


def test_is_available_false_when_env_unset(sdn):
    assert sdn.is_available() is False


def test_is_available_true_when_env_set(sdn, monkeypatch):
    # Force the AF_UNIX guard on so path-handling logic is testable on
    # platforms where the kernel doesn't support AF_UNIX (e.g. Windows CI).
    monkeypatch.setattr(sdn, "_HAS_AF_UNIX", True)
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    assert sdn.is_available() is True


def test_notify_socket_path_abstract_namespace(sdn, monkeypatch):
    """Linux abstract namespace prefix ``@`` is translated to NUL-prefix."""
    monkeypatch.setattr(sdn, "_HAS_AF_UNIX", True)
    monkeypatch.setenv("NOTIFY_SOCKET", "@/test/abstract")
    assert sdn._notify_socket_path() == "\0/test/abstract"


def test_notify_socket_path_filesystem(sdn, monkeypatch):
    monkeypatch.setattr(sdn, "_HAS_AF_UNIX", True)
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    assert sdn._notify_socket_path() == "/run/systemd/notify"


def test_notify_socket_path_unset(sdn):
    assert sdn._notify_socket_path() is None


# ── notify() send semantics ────────────────────────────────────────────────


def test_notify_noop_when_env_unset(sdn):
    """No socket configured → notify returns False, no side effects."""
    assert sdn.notify("READY=1") is False


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
def test_notify_sends_to_filesystem_socket(sdn, notify_socket, monkeypatch):
    path, sock = notify_socket
    monkeypatch.setenv("NOTIFY_SOCKET", path)

    assert sdn.notify("READY=1") is True

    data, _ = sock.recvfrom(4096)
    assert data == b"READY=1"


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
def test_notify_ready_sends_ready_payload(sdn, notify_socket, monkeypatch):
    path, sock = notify_socket
    monkeypatch.setenv("NOTIFY_SOCKET", path)

    assert sdn.notify_ready() is True
    data, _ = sock.recvfrom(4096)
    assert data == b"READY=1"


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
def test_notify_watchdog_sends_watchdog_payload(sdn, notify_socket, monkeypatch):
    path, sock = notify_socket
    monkeypatch.setenv("NOTIFY_SOCKET", path)

    assert sdn.notify_watchdog() is True
    data, _ = sock.recvfrom(4096)
    assert data == b"WATCHDOG=1"


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
def test_notify_stopping_sends_stopping_payload(sdn, notify_socket, monkeypatch):
    path, sock = notify_socket
    monkeypatch.setenv("NOTIFY_SOCKET", path)

    assert sdn.notify_stopping() is True
    data, _ = sock.recvfrom(4096)
    assert data == b"STOPPING=1"


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
def test_notify_returns_false_on_dead_socket(sdn, tmp_path, monkeypatch):
    """If the socket path doesn't exist, notify swallows the OSError + returns False."""
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "nonexistent.sock"))
    assert sdn.notify("READY=1") is False


def test_helpers_are_noop_on_non_posix(sdn, monkeypatch):
    """On platforms without AF_UNIX, every helper returns False / None safely."""
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    # Simulate a non-POSIX platform by clearing the module's AF_UNIX flag.
    monkeypatch.setattr(sdn, "_HAS_AF_UNIX", False)
    assert sdn._notify_socket_path() is None
    assert sdn.is_available() is False
    assert sdn.notify("READY=1") is False
    assert sdn.notify_ready() is False
    assert sdn.notify_watchdog() is False
    assert sdn.notify_stopping() is False


# ── watchdog_usec ──────────────────────────────────────────────────────────


def test_watchdog_usec_none_when_env_unset(sdn):
    assert sdn.watchdog_usec() is None


def test_watchdog_usec_parses_int(sdn, monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "60000000")
    assert sdn.watchdog_usec() == 60_000_000


def test_watchdog_usec_returns_none_on_malformed(sdn, monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "not-an-int")
    assert sdn.watchdog_usec() is None


# ── watchdog_heartbeat_task ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_heartbeat_returns_immediately_without_env(sdn):
    """No WATCHDOG_USEC set → task returns immediately (no infinite loop)."""
    await asyncio.wait_for(sdn.watchdog_heartbeat_task(), timeout=2.0)


@pytest.mark.skipif(not _is_posix(), reason="AF_UNIX sockets are POSIX-only")
@pytest.mark.asyncio
async def test_watchdog_heartbeat_pings_at_half_interval(sdn, notify_socket, monkeypatch):
    """Heartbeat task fires watchdog pings until cancelled."""
    path, sock = notify_socket
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    # Tiny watchdog interval (200 ms total → half-interval = 100 ms) for a
    # fast test. The helper clamps min interval to 1.0s — we monkeypatch it
    # down so the test stays fast.
    monkeypatch.setenv("WATCHDOG_USEC", "200000")

    # Override the min-interval clamp inside the helper for the test.
    original_sleep = asyncio.sleep

    async def fast_sleep(_secs):
        await original_sleep(0.05)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(sdn.watchdog_heartbeat_task())
    try:
        # Use blocking recv on the listener — let it accumulate pings.
        sock.settimeout(1.0)
        data1, _ = sock.recvfrom(4096)
        data2, _ = sock.recvfrom(4096)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert data1 == b"WATCHDOG=1"
    assert data2 == b"WATCHDOG=1"


@pytest.mark.asyncio
async def test_watchdog_heartbeat_propagates_cancellation(sdn, monkeypatch):
    """Cancelling the task re-raises CancelledError so callers can shut it down cleanly."""
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/nonexistent/notify")
    monkeypatch.setenv("WATCHDOG_USEC", "60000000")

    task = asyncio.create_task(sdn.watchdog_heartbeat_task())
    await asyncio.sleep(0)  # let the task enter the loop
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
