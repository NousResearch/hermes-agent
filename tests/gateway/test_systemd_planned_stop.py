"""Tests for the systemd ExecStop planned-stop marker helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import status
import hermes_systemd_planned_stop as systemd_planned_stop


def test_main_writes_consumable_marker_for_target_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert systemd_planned_stop.main([str(os.getpid())]) == 0
    # ExecStop publishes the marker before systemd sends SIGTERM. The generic
    # watcher must leave it for the real signal handler to consume, otherwise
    # the later SIGTERM would be misclassified as unexpected.
    assert status.planned_stop_marker_targets_self() is False
    assert (tmp_path / ".gateway-planned-stop.json").exists()
    assert status.consume_planned_stop_marker_for_self() is True


def test_main_rejects_missing_invalid_and_nonpositive_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert systemd_planned_stop.main([]) == 2
    assert systemd_planned_stop.main(["not-a-pid"]) == 2
    assert systemd_planned_stop.main(["0"]) == 2
    assert not (tmp_path / ".gateway-planned-stop.json").exists()


def test_main_reports_marker_write_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        systemd_planned_stop,
        "write_planned_stop_marker",
        lambda _pid, **_kwargs: False,
    )

    assert systemd_planned_stop.main(["1234"]) == 1
    assert "could not write the marker" in capsys.readouterr().err


def test_helper_runs_without_optional_dependencies(tmp_path):
    """ExecStop must work even when application dependencies are unavailable."""
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-S", "-m", "hermes_systemd_planned_stop", str(os.getpid())],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    marker = tmp_path / ".gateway-planned-stop.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["target_pid"] == os.getpid()
    assert record["trigger_watcher"] is False


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("comm", [b"hermes gateway", b"hermes-\xff"])
def test_marker_consumes_with_nontrivial_linux_comm(tmp_path, comm):
    """Spaces/non-UTF-8 bytes in comm must not corrupt PID/start-time identity."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", """
import ctypes
import json
import os
from pathlib import Path
import sys

from gateway import status
import hermes_systemd_planned_stop as helper

libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(15, ctypes.c_char_p(bytes.fromhex(sys.argv[1])), 0, 0, 0) == 0
assert helper.main([str(os.getpid())]) == 0
marker = Path(os.environ["HERMES_HOME"]) / ".gateway-planned-stop.json"
record = json.loads(marker.read_text(encoding="utf-8"))
assert marker.stat().st_mode & 0o777 == 0o600
assert status.consume_planned_stop_marker_for_self()
assert record["target_start_time"] == status._get_process_start_time(os.getpid())
assert record["target_start_time"] is not None
""", comm.hex()],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
