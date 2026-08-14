"""Tests for the systemd ExecStop planned-stop marker helper."""

from __future__ import annotations

import os

from gateway import status, systemd_planned_stop


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
