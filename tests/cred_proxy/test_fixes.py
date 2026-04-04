"""Tests for specific bug fixes in the credential proxy."""

import asyncio
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fix: is_running() returns False and cleans up stale PID file
# ---------------------------------------------------------------------------

def test_is_running_cleans_up_stale_pid_and_socket(tmp_path, monkeypatch):
    """is_running() returns False and removes stale PID + socket files."""
    import cred_proxy.daemon as daemon_module

    pid_file = tmp_path / "cred-proxy.pid"
    sock_file = tmp_path / "cred-proxy.sock"

    monkeypatch.setattr(daemon_module, "_PID_FILE", pid_file)
    monkeypatch.setattr(daemon_module, "_SOCKET_PATH", sock_file)

    # Find a PID that definitely does not exist on this system
    dead_pid = 999999
    try:
        os.kill(dead_pid, 0)
        pytest.skip("PID 999999 unexpectedly exists on this system")
    except ProcessLookupError:
        pass

    pid_file.write_text(str(dead_pid))
    sock_file.write_text("")  # fake socket file

    assert daemon_module.is_running() is False
    assert not pid_file.exists(), "Stale PID file was not removed"
    assert not sock_file.exists(), "Stale socket file was not removed"


# ---------------------------------------------------------------------------
# Fix: stop() preserves state files on PermissionError
# ---------------------------------------------------------------------------

def test_stop_preserves_files_on_permission_error(tmp_path, monkeypatch):
    """stop() leaves PID/socket files intact when os.kill raises PermissionError."""
    import cred_proxy.daemon as daemon_module

    pid_file = tmp_path / "cred-proxy.pid"
    sock_file = tmp_path / "cred-proxy.sock"

    monkeypatch.setattr(daemon_module, "_PID_FILE", pid_file)
    monkeypatch.setattr(daemon_module, "_SOCKET_PATH", sock_file)
    monkeypatch.setattr(daemon_module, "_STATE_DIR", tmp_path)

    pid_file.write_text("12345")
    sock_file.write_text("")

    with patch("os.kill", side_effect=PermissionError("Operation not permitted")):
        daemon_module.stop()

    assert pid_file.exists(), "PID file should be preserved on PermissionError"


def test_stop_cleans_files_on_success(tmp_path, monkeypatch):
    """stop() removes PID/socket files after a successful SIGTERM."""
    import cred_proxy.daemon as daemon_module

    pid_file = tmp_path / "cred-proxy.pid"
    sock_file = tmp_path / "cred-proxy.sock"

    monkeypatch.setattr(daemon_module, "_PID_FILE", pid_file)
    monkeypatch.setattr(daemon_module, "_SOCKET_PATH", sock_file)
    monkeypatch.setattr(daemon_module, "_STATE_DIR", tmp_path)

    pid_file.write_text("12345")
    sock_file.write_text("")

    with patch("os.kill"):
        daemon_module.stop()

    assert not pid_file.exists(), "PID file should be removed after successful stop"


# ---------------------------------------------------------------------------
# Fix: run_proxy binds socket and calls on_started
# ---------------------------------------------------------------------------

def test_run_proxy_unix_socket_calls_on_started(tmp_path):
    """run_proxy(unix_socket=...) calls on_started with the socket path."""
    from cred_proxy.server import run_proxy
    from cred_proxy.store import CredStore

    sock_path = str(tmp_path / "test.sock")
    reported_path = None

    def on_started(path: str) -> None:
        nonlocal reported_path
        reported_path = path

    async def _run():
        store = CredStore()
        server_task = asyncio.create_task(
            run_proxy(unix_socket=sock_path, on_started=on_started, store=store)
        )
        # Give the server a moment to bind
        await asyncio.sleep(0.1)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    assert reported_path == sock_path
    assert os.path.exists(sock_path)
