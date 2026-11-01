"""Tests for completing tasks in 'gave_up' status with evidence (#91833)."""

from __future__ import annotations

from pathlib import Path
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _gave_up_task(conn, assignee: str = "coder") -> str:
    tid = kb.create_task(conn, title="demo task", assignee=assignee)
    run_id = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert run_id is not None
    # Transition to gave_up
    conn.execute(
        "UPDATE tasks SET status = 'gave_up', claim_lock = NULL, worker_pid = NULL WHERE id = ?",
        (tid,),
    )
    conn.commit()
    return tid


def test_gave_up_without_comment_raises(conn):
    tid = _gave_up_task(conn)
    with pytest.raises(kb.GaveUpWithoutEvidenceError):
        kb.complete_task(conn, tid, result="attempting completion")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "gave_up"


def test_gave_up_with_non_result_comment_raises(conn):
    tid = _gave_up_task(conn, assignee="coder")
    kb.add_comment(conn, tid, author="coder", body="I am giving up on this task")
    with pytest.raises(kb.GaveUpWithoutEvidenceError):
        kb.complete_task(conn, tid, result="completing anyway")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "gave_up"


def test_gave_up_with_different_author_raises(conn):
    tid = _gave_up_task(conn, assignee="coder")
    kb.add_comment(conn, tid, author="reviewer", body="result: here is the fix")
    with pytest.raises(kb.GaveUpWithoutEvidenceError):
        kb.complete_task(conn, tid, result="done")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "gave_up"


def test_gave_up_with_assignee_result_completes(conn):
    tid = _gave_up_task(conn, assignee="Coder")
    # Case-insensitive author matching and prefix check
    kb.add_comment(conn, tid, author="coder", body="Result: PR opened at http://example.com/pr/1")
    ok = kb.complete_task(conn, tid, result="verified and done")
    assert ok is True
    row = conn.execute("SELECT status, result FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "done"
    assert row["result"] == "verified and done"


def test_gave_up_stale_expected_run_id_rejected(conn):
    tid = _gave_up_task(conn, assignee="coder")
    kb.add_comment(conn, tid, author="coder", body="evidence: all tests passing")
    current_run = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()["current_run_id"]
    stale_run_id = (current_run or 0) + 999
    ok = kb.complete_task(conn, tid, result="done", expected_run_id=stale_run_id)
    assert ok is False
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "gave_up"
