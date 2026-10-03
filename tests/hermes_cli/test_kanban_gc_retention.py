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

def _archived_scratch_workspace(conn) -> Path:
    tid = kb.create_task(conn, title="archived with workspace")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='archived', workspace_kind='scratch' WHERE id=?",
            (tid,),
        )
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "scratch.txt").write_text("keep me")
    return ws

@pytest.mark.parametrize(
    ("days", "expect_rc", "expect_kept"),
    [
        pytest.param(-1, 2, True, id="negative-refuses"),
        pytest.param(0, 0, True, id="zero-disables"),
        pytest.param(30, 0, False, id="positive-collects"),
    ],
)
def test_cmd_gc_retention_bounds(board, days, expect_rc, expect_kept):
    """Invalid retention must refuse before ANY sweep: the (unconditional)
    workspace collection runs first in the command body, so the archived
    scratch workspace surviving the negative case proves ordering, not just
    event/log preservation. Valid values (0 or positive) let it run."""
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
        ws = _archived_scratch_workspace(conn)
    log = _old_log_file()
    assert kanban_ops._cmd_gc(_args(event_days=days, log_days=days)) == expect_rc
    with kbc.connect_closing() as conn:
        assert (_event_rows(conn, tid) > 0) is expect_kept
    assert log.exists() is expect_kept
    assert (ws / "scratch.txt").exists() is (expect_rc != 0)

@pytest.mark.parametrize(
    ("days", "expected"),
    [
        pytest.param("-1", "must be >= 0", id="negative-blocked"),
        pytest.param("0", "GC complete", id="zero-disables"),
    ],
)
def test_slash_kanban_gc_retention_bounds(board, days, expected):
    """``/kanban gc`` from a chat session uses the same argparse type as the
    shell command: ``-1`` is rejected by the parser type before ``_cmd_gc``
    runs (usage error); ``0`` parses, reaches ``_cmd_gc``, and disables the
    sweep."""
    from hermes_cli import kanban
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
    log = _old_log_file()
    out = kanban.run_slash(f"gc --event-retention-days {days} --log-retention-days {days}")
    assert expected in out
    with kbc.connect_closing() as conn:
        assert _event_rows(conn, tid) > 0
    assert log.exists()


def test_cmd_gc_never_removes_the_workspaces_root_itself(board):
    # A scratch task whose workspace_path is the managed root itself (reachable via
    # kanban_create) must never make gc wipe every other task's scratch dir.
    root = kb.workspaces_root()
    sibling = root / "t_other"
    sibling.mkdir(parents=True)
    (sibling / "work.txt").write_text("another task's scratch", encoding="utf-8")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="archived scratch")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', workspace_kind='scratch', "
                "workspace_path=? WHERE id=?",
                (str(root), tid),
            )
    assert kanban_ops._cmd_gc(_args()) == 0
    assert (sibling / "work.txt").exists()


def _shared_scratch_task(conn, title: str, path: Path, status: str) -> str:
    tid = kb.create_task(conn, title=title)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status=?, workspace_kind='scratch', workspace_path=? "
            "WHERE id=?",
            (status, str(path), tid),
        )
    return tid


def test_cmd_gc_keeps_workspace_used_by_live_task_same_board(board):
    shared = kb.workspaces_root() / "shared-live"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("keep", encoding="utf-8")
    with kbc.connect_closing() as conn:
        archived = _shared_scratch_task(conn, "archived", shared, "archived")
        live = _shared_scratch_task(conn, "live", shared, "ready")

    assert kanban_ops._cmd_gc(_args()) == 0
    assert (shared / "note.txt").exists()
    with kbc.connect_closing() as conn:
        event = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id=? "
            "AND kind='workspace_cleanup_deferred_shared'",
            (archived,),
        ).fetchone()
        assert event is not None
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (live,))

    assert kanban_ops._cmd_gc(_args()) == 0
    assert not shared.exists()


def test_cmd_gc_keeps_workspace_used_by_live_task_other_board(board, monkeypatch):
    kb.create_board("other")
    default_db = kb.kanban_db_path(board=kb.DEFAULT_BOARD)
    default_ws = kb.workspaces_root(board=kb.DEFAULT_BOARD)
    shared = default_ws / "shared-cross-board"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("keep", encoding="utf-8")
    with kbc.connect_closing(board=kb.DEFAULT_BOARD) as conn:
        _shared_scratch_task(conn, "archived default", shared, "archived")

    # Workers pin their own board DB/root in the environment. The cleanup guard
    # must still inspect physical sibling-board DBs rather than resolving every
    # board through these process-wide pins.
    with kbc.connect_closing(board="other") as other_conn:
        live = _shared_scratch_task(other_conn, "live other", shared, "ready")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(default_db))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(default_ws))

        with kb.scoped_current_board(kb.DEFAULT_BOARD):
            assert kanban_ops._cmd_gc(_args()) == 0
        assert (shared / "note.txt").exists()

        with kb.write_txn(other_conn):
            other_conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (live,))
        with kb.scoped_current_board(kb.DEFAULT_BOARD):
            assert kanban_ops._cmd_gc(_args()) == 0
    assert not shared.exists()


def test_cleanup_workspace_keeps_path_used_by_unrelated_live_task(board):
    from hermes_cli import kanban_db_workspace as kbw

    shared = kb.workspaces_root() / "shared-completion"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("keep", encoding="utf-8")
    with kbc.connect_closing() as conn:
        finished = _shared_scratch_task(conn, "finished", shared, "done")
        _shared_scratch_task(conn, "unrelated live", shared, "running")
        kbw._cleanup_workspace(conn, finished)
    assert (shared / "note.txt").exists()


def test_deferred_parent_cleanup_keeps_path_used_by_unrelated_live_task(board):
    from hermes_cli import kanban_db_workspace as kbw

    shared = kb.workspaces_root() / "shared-parent"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("keep", encoding="utf-8")
    with kbc.connect_closing() as conn:
        parent = _shared_scratch_task(conn, "parent", shared, "done")
        child = kb.create_task(conn, title="terminal child")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
            conn.execute(
                "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
                (parent, child),
            )
        _shared_scratch_task(conn, "unrelated live", shared, "review")
        kbw._try_cleanup_parent_workspaces(conn, child)
    assert (shared / "note.txt").exists()
