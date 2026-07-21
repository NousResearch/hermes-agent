"""Behavior tests for the read-only Kanban reliability report."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_reliability import (
    build_reliability_report,
    categorize_failure,
    open_readonly,
)


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (
            {"outcome": "reclaimed", "metadata": {"failure_category": "heartbeat_stale_reclaimed"}},
            "heartbeat_reclaim",
        ),
        ({"outcome": "failed", "error": "completion gate rejected: missing PR evidence"}, "completion_gate"),
        ({"outcome": "spawn_failed", "error": "GitHub authentication token is missing"}, "auth_or_tooling"),
        ({"outcome": "crashed", "error": "worker pid is not alive"}, "pid_not_alive"),
        ({"outcome": "failed", "error": "protocol violation: kanban_complete was not called"}, "protocol_violation"),
        ({"outcome": "timed_out", "error": "Iteration budget exhausted (90/90)"}, "iteration_exhaustion"),
        ({"outcome": "spawn_failed", "error": "workspace parent artifact guard found missing files"}, "workspace_or_artifact"),
        ({"outcome": "blocked", "error": None}, "blocked"),
    ],
)
def test_categorize_failure_groups_operator_failure_classes(run, expected):
    assert categorize_failure(run) == expected


@pytest.fixture
def report_db(tmp_path: Path) -> Path:
    path = tmp_path / "kanban.db"
    kb.init_db(path)
    now = 2_000_000_000
    with kb.connect(path) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at, workspace_kind, "
            "block_kind, last_failure_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("t_blocked", "Needs credentials", "blocked", "coder", now - 500, "scratch", "needs_input", "missing token"),
        )
        conn.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at, workspace_kind) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("t_done", "Recovered", "done", "researcher", now - 700, "scratch"),
        )
        runs = [
            ("t_blocked", "coder", "reclaimed", "reclaimed", now - 400, now - 300,
             json.dumps({"failure_category": "heartbeat_stale_reclaimed", "stale_lock_host": "host-a"}), "stale_lock=host-a:1"),
            ("t_done", "researcher", "reclaimed", "reclaimed", now - 390, now - 290,
             json.dumps({"failure_category": "heartbeat_stale_reclaimed", "stale_lock_host": "host-a"}), "stale_lock=host-a:1"),
            ("t_done", "researcher", "done", "completed", now - 200, now - 100, None, None),
        ]
        conn.executemany(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            runs,
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            ("t_blocked", "blocked", json.dumps({"reason": "missing deploy credential"}), now - 250),
        )
        conn.commit()
    return path


def test_report_has_bounded_counts_bursts_reasons_and_active_blockers(report_db: Path):
    with open_readonly(report_db) as conn:
        report = build_reliability_report(conn, window_seconds=86_400, now=2_000_000_000, limit=1)

    assert report["runs"]["by_outcome"] == {"completed": 1, "reclaimed": 2}
    assert report["runs"]["by_profile"] == {"coder": 1, "researcher": 2}
    assert report["failures"]["by_category"] == {"heartbeat_reclaim": 2}
    assert report["stale_lock_bursts"] == [
        {"stale_lock": "host-a:1", "count": 2, "first_at": 1_999_999_700, "last_at": 1_999_999_710}
    ]
    assert report["top_blocked_reasons"] == [{"reason": "missing deploy credential", "count": 1}]
    assert report["active_actionable_blockers"] == [
        {
            "task_id": "t_blocked",
            "title": "Needs credentials",
            "assignee": "coder",
            "status": "blocked",
            "block_kind": "needs_input",
            "reason": "missing deploy credential",
        }
    ]
    assert report["truncated"]["active_actionable_blockers"] is False


def test_open_readonly_enables_sqlite_query_only(report_db: Path):
    with open_readonly(report_db) as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE tasks SET title='mutated'")


def test_reliability_cli_accepts_window_and_saves_output_without_initializing(
    report_db: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(report_db))
    monkeypatch.setattr(kb, "init_db", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must stay read-only")))
    output = tmp_path / "report.json"

    result = kc.run_slash(f"reliability --window 24h --json --output {output}")

    assert "Saved Kanban reliability report" in result
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["window"]["label"] == "24h"
    assert payload["runs"]["total"] == 3
