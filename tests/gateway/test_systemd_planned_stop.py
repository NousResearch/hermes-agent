"""Tests for the systemd ExecStop planned-stop marker helper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def test_helper_imports_without_optional_dependencies():
    """ExecStop must work even when application dependencies are unavailable."""
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [sys.executable, "-S", "-c", "import hermes_systemd_planned_stop"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
