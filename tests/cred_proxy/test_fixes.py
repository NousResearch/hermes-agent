"""Tests for specific bug fixes in the credential proxy."""

import os

import pytest


# ---------------------------------------------------------------------------
# Fix 6: is_running() returns False and cleans up stale PID file
# ---------------------------------------------------------------------------

def test_is_running_cleans_up_stale_pid_and_port_files(tmp_path, monkeypatch):
    """is_running() returns False and removes stale PID + port files."""
    import cred_proxy.daemon as daemon_module

    pid_file = tmp_path / "cred-proxy.pid"
    port_file = tmp_path / "cred-proxy.port"

    monkeypatch.setattr(daemon_module, "_PID_FILE", pid_file)
    monkeypatch.setattr(daemon_module, "_PORT_FILE", port_file)

    # Find a PID that definitely does not exist on this system
    dead_pid = 999999
    try:
        os.kill(dead_pid, 0)
        pytest.skip("PID 999999 unexpectedly exists on this system")
    except ProcessLookupError:
        pass

    pid_file.write_text(str(dead_pid))
    port_file.write_text("12345")

    assert daemon_module.is_running() is False
    assert not pid_file.exists(), "Stale PID file was not removed"
    assert not port_file.exists(), "Stale port file was not removed"
