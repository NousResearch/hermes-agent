"""Retention bounds for ``kanban gc``: a negative window builds a future cutoff
that matches every row; zero disables the sweep rather than deleting all."""
import argparse
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_ops


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _done_task_with_old_event(conn):
    tid = kb.create_task(conn, title="finished")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (tid,))
    return tid


def _event_rows(conn, tid):
    return conn.execute(
        "SELECT count(*) FROM task_events WHERE task_id=?", (tid,)
    ).fetchone()[0]


def _old_log_file() -> Path:
    log_dir = kb.worker_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "worker-1.log"
    p.write_text("log line")
    os.utime(p, (0, 0))
    return p


def _args(event_days=30, log_days=30):
    return argparse.Namespace(event_retention_days=event_days,
                              log_retention_days=log_days)


def test_gc_events_rejects_negative_window(board):
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
        with pytest.raises(ValueError, match="older_than_seconds"):
            kb.gc_events(conn, older_than_seconds=-86400)
        assert _event_rows(conn, tid) > 0


def test_gc_worker_logs_rejects_negative_window(board):
    log = _old_log_file()
    with pytest.raises(ValueError, match="older_than_seconds"):
        kb.gc_worker_logs(older_than_seconds=-86400)
    assert log.exists()


def test_cmd_gc_negative_days_errors_and_deletes_nothing(board, capsys):
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_args(event_days=-1, log_days=-1)) != 0
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) > 0
    assert log.exists()


def test_cmd_gc_negative_days_leaves_workspaces_untouched(board):
    """Invalid retention must refuse before ANY sweep: the workspace collection
    runs first in the command body, so this fixture proves ordering, not just
    event/log preservation."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="archived with workspace")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', workspace_kind='scratch' WHERE id=?",
                (tid,),
            )
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "scratch.txt").write_text("keep me")
    assert kanban_ops._cmd_gc(_args(event_days=-1)) != 0
    assert (ws / "scratch.txt").exists()


def test_cmd_gc_zero_days_disables_sweeps(board):
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_args(event_days=0, log_days=0)) == 0
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) > 0
    assert log.exists()


def test_cmd_gc_positive_days_still_collects(board):
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_args(event_days=30, log_days=30)) == 0
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) == 0
    assert not log.exists()


def test_slash_kanban_gc_negative_days_blocked(board):
    """``/kanban gc`` from a chat session lands on the same _cmd_gc guard."""
    from hermes_cli import kanban
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    out = kanban.run_slash("gc --event-retention-days -1 --log-retention-days -1")
    assert "retention days must be >= 0" in out
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) > 0
    assert log.exists()


def test_slash_kanban_gc_zero_disables(board):
    from hermes_cli import kanban
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    out = kanban.run_slash("gc --event-retention-days 0 --log-retention-days 0")
    assert "GC complete" in out
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) > 0
    assert log.exists()
