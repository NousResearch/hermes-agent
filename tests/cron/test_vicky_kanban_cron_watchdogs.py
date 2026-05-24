"""Regression coverage for Vicky's profile-local Kanban cron watchdog scripts.

These scripts live outside the hermes-agent repo, but this repo-managed test
file pins the operational behavior that keeps no-agent cron jobs quiet while
still surfacing Kanban DB corruption to agents/operators.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import sqlite3
import sys
from pathlib import Path

SCRIPTS_DIR = Path("/home/john/.hermes/profiles/vicky/scripts")


def import_script(name: str):
    if not SCRIPTS_DIR.exists():
        raise AssertionError(f"missing vicky scripts dir: {SCRIPTS_DIR}")
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def make_corrupt_board(root: Path, board: str = "hermes-setup") -> Path:
    board_dir = root / board
    board_dir.mkdir(parents=True)
    db = board_dir / "kanban.db"
    db.write_bytes(b"not sqlite at all")
    return db


def test_quick_check_detects_corrupt_board_without_writing(tmp_path):
    alerts = import_script("kanban_cron_alerts")
    db = make_corrupt_board(tmp_path / "boards")
    before = db.read_bytes()

    errors = alerts.kanban_integrity_errors(tmp_path / "boards", boards={"hermes-setup"})

    assert errors
    assert "hermes-setup" in errors[0]
    assert "quick_check failed" in errors[0]
    assert db.read_bytes() == before


def test_integrity_routing_writes_operator_fallback_when_board_create_fails(tmp_path, monkeypatch):
    alerts = import_script("kanban_cron_alerts")
    log_path = tmp_path / "agent_alerts.jsonl"
    fallback_path = tmp_path / "fallback.md"
    monkeypatch.setattr(alerts, "LOG_PATH", log_path)
    monkeypatch.setattr(alerts, "FALLBACK_ALERT_PATH", fallback_path)

    class FailedCreate:
        returncode = 1
        stdout = ""
        stderr = "database disk image is malformed"

    monkeypatch.setattr(alerts.subprocess, "run", lambda *args, **kwargs: FailedCreate())

    stopped = alerts.route_integrity_errors_to_agent(
        "kanban_monitor",
        ["hermes-setup: wrong # of entries in index idx_events_task"],
        assignee="coder",
    )

    assert stopped is True
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["create_rc"] == 1
    assert record["fallback_breadcrumb"] == str(fallback_path)
    text = fallback_path.read_text(encoding="utf-8")
    assert "Kanban cron failure needs agent/operator review" in text
    assert "hermes-setup: wrong # of entries" in text
    assert "database disk image is malformed" in text


def test_no_agent_cron_mode_routes_integrity_failure_and_stays_silent(monkeypatch):
    queue_pusher = import_script("kanban_queue_pusher")
    calls = []
    monkeypatch.setattr(
        queue_pusher,
        "kanban_integrity_errors",
        lambda boards=None: ["hermes-setup: quick_check failed for test"],
    )
    monkeypatch.setattr(
        queue_pusher,
        "route_integrity_errors_to_agent",
        lambda source, errors, assignee="coder": calls.append((source, errors, assignee)) or True,
    )

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        rc = queue_pusher.main()

    assert rc == 0
    assert stdout.getvalue() == ""
    assert calls == [("kanban_queue_pusher", ["hermes-setup: quick_check failed for test"], "coder")]


def test_check_noop_fails_noisily_for_corrupt_board(tmp_path, monkeypatch):
    monitor = import_script("kanban_monitor")
    make_corrupt_board(tmp_path / "boards")
    monkeypatch.setattr(monitor, "KANBAN_ROOT", tmp_path / "boards")
    monkeypatch.setattr(sys, "argv", ["kanban_monitor.py", "--check-noop"])

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        rc = monitor.main()

    assert rc == 1
    assert "hermes-setup" in stdout.getvalue()
    assert "quick_check" in stdout.getvalue()
