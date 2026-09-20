"""Regression coverage for the watchdog descendant-kill and Windows restart hand-off.

Complements test_shutdown_watchdog.py (drain/heartbeat contract for #66892) and
the restart tests in test_gateway_windows.py. The descendant-kill test spawns a
real sleeper child so the psutil path is exercised end-to-end, not mocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import gateway.shutdown_watchdog as shutdown_watchdog_module
from gateway.shutdown_watchdog import (
    _terminate_gateway_descendants_before_hard_exit,
    arm_shutdown_watchdog,
)


# ── descendant kill before hard exit ───────────────────────────────────────


def test_descendant_kill_terminates_a_real_child():
    """The kill helper terminates live descendants (real subprocess, not a mock)."""
    if sys.platform == "win32":
        cmd = [sys.executable, "-c", "import time; time.sleep(120)"]
    else:
        cmd = ["sleep", "120"]
    child = subprocess.Popen(cmd)
    try:
        # Give the child a moment to appear as a descendant of this process.
        deadline = time.monotonic() + 5.0
        terminated = 0
        while time.monotonic() < deadline:
            terminated = _terminate_gateway_descendants_before_hard_exit()
            if terminated:
                break
            time.sleep(0.1)
        assert terminated >= 1, "descendant kill did not terminate the child"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_descendant_kill_is_quiet_with_no_children():
    """With no descendants the helper returns 0 instead of raising."""
    assert _terminate_gateway_descendants_before_hard_exit() == 0


def test_shutdown_watchdog_kills_descendants_before_exit(tmp_path):
    """The shutdown watchdog terminates descendants, then hard-exits."""
    done = threading.Event()
    dump = tmp_path / "logs" / "watchdog.log"
    exit_codes = []

    if sys.platform == "win32":
        cmd = [sys.executable, "-c", "import time; time.sleep(120)"]
    else:
        cmd = ["sleep", "120"]
    child = subprocess.Popen(cmd)

    try:
        with patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append):
            arm_shutdown_watchdog(0.15, done_event=done, dump_path=dump, exit_code=9)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not exit_codes:
                time.sleep(0.05)
            assert exit_codes == [9], "watchdog did not fire"
            # The watchdog's own descendant kill must have terminated the child
            # before the patched os._exit — assert on the process, not a mock.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and child.poll() is None:
                time.sleep(0.05)
            assert child.poll() is not None, "watchdog left the descendant alive"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


# ── Windows planned-restart hand-off marker ────────────────────────────────


def test_restart_writes_online_marker_before_launching_replacement(tmp_path, monkeypatch):
    """A Windows restart must tell the next boot to announce it is online."""
    import hermes_cli.gateway_windows as gateway_windows

    calls = []
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(
        gateway_windows,
        "_wait_for_gateway_absent",
        lambda **_kwargs: calls.append("absent") or True,
    )
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda _seconds: None)

    # Neutralize the port-free wait (win32-only ctypes import chain).
    monkeypatch.setattr(
        "hermes_cli.gateway._wait_for_api_server_port_free", lambda: calls.append("portfree")
    )

    marker_seen_by_start = {}

    def fake_start():
        marker_seen_by_start["exists"] = (tmp_path / ".restart_pending.json").exists()
        calls.append("start")

    monkeypatch.setattr(gateway_windows, "start", fake_start)
    monkeypatch.setattr(
        gateway_windows,
        "_wait_for_gateway_ready",
        lambda **_kwargs: calls.append("ready") or True,
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    gateway_windows.restart()

    assert calls == ["stop", "absent", "portfree", "start", "ready"]
    assert marker_seen_by_start == {"exists": True}
    marker = tmp_path / ".restart_pending.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["detached"] is True


def test_restart_removes_online_marker_if_replacement_launch_fails(tmp_path, monkeypatch):
    """A failed launch must not make a later unrelated start claim a restart."""
    import hermes_cli.gateway_windows as gateway_windows

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "stop", lambda: None)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_absent", lambda **_kwargs: True)
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "hermes_cli.gateway._wait_for_api_server_port_free", lambda: None
    )
    monkeypatch.setattr(
        gateway_windows, "start", lambda: (_ for _ in ()).throw(RuntimeError("launch failed"))
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="launch failed"):
        gateway_windows.restart()

    assert not (tmp_path / ".restart_pending.json").exists()


def test_planned_restart_marker_shape_matches_gateway_replay_reader(tmp_path, monkeypatch):
    """The marker must be JSON the gateway's replay reader can consume."""
    import hermes_cli.gateway_windows as gateway_windows

    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    path = gateway_windows._write_planned_restart_notification_marker()

    assert path is not None and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "requested_at" in data
    # The gateway replay reader iterates delivered_targets as a list (possibly
    # absent); it must not choke on the writer's initial shape.
    assert data.get("delivered_targets", []) == []
