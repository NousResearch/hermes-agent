"""Regression tests for Kanban respawn-guard diagnostics."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_stale_error_from_older_run_does_not_guard_review_handoff(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale auth stamp", assignee="a")
        first = kb.claim_task(conn, tid)
        assert first is not None
        failed_run_id = first.current_run_id
        assert failed_run_id is not None
        conn.execute(
            "UPDATE task_runs SET outcome='failed', status='failed', ended_at=? WHERE id=?",
            (5_000_000, failed_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error="
            "'provider authentication failed', last_failure_run_id=? WHERE id=?",
            (failed_run_id, tid),
        )
        conn.commit()

        review_run = kb.claim_task(conn, tid)
        assert review_run is not None
        assert review_run.current_run_id != failed_run_id
        assert kb.request_review(
            conn, tid, summary="PR ready", expected_run_id=review_run.current_run_id,
        )

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.last_failure_error == "provider authentication failed"
        stored_run_id = conn.execute(
            "SELECT last_failure_run_id FROM tasks WHERE id=?", (tid,),
        ).fetchone()[0]
        assert stored_run_id == failed_run_id
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize(
    ("error_text", "expected"),
    [
        ("t_97912509-403-vocabulario; los dos tests del 403", None),
        ("403", None),
        ("429", None),
        ("HTTP 403 Forbidden", "blocker_auth"),
        ("HTTP/1.1 429 Too Many Requests", "blocker_auth"),
    ],
)
def test_status_codes_need_http_diagnostic_context(kanban_home, monkeypatch, error_text, expected):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="status-code prose", assignee="a")
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (error_text, tid))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == expected
