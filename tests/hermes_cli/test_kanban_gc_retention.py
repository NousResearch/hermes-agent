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


def _shared_scratch(conn) -> tuple[str, str, Path]:
    archived = kb.create_task(conn, title="archived sharer")
    live = kb.create_task(conn, title="live sharer")
    shared = kb.workspaces_root() / "shared-scratch"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("still needed", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='archived', workspace_kind='scratch', "
            "workspace_path=? WHERE id=?",
            (str(shared), archived),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', workspace_kind='scratch', "
            "workspace_path=? WHERE id=?",
            (str(shared), live),
        )
    return archived, live, shared


def test_cmd_gc_keeps_a_scratch_dir_a_live_task_still_uses(board):
    with kbc.connect_closing() as conn:
        archived, _live, shared = _shared_scratch(conn)
    assert kanban_ops._cmd_gc(_args()) == 0
    assert (shared / "note.txt").read_text(encoding="utf-8") == "still needed"
    with kbc.connect_closing() as conn:
        kinds = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? AND kind=?",
            (archived, "workspace_cleanup_deferred_shared"),
        ).fetchall()
    assert kinds


def test_cleanup_workspace_keeps_scratch_when_other_boards_cannot_be_listed(board, monkeypatch):
    """A named board's live task uses the dir; if boards cannot be listed, refuse the delete."""
    from hermes_cli import kanban_db_workspace as kbw

    kb.create_board("other")
    shared = kb.workspaces_root() / "cross-board"
    shared.mkdir(parents=True)
    (shared / "note.txt").write_text("still needed", encoding="utf-8")
    with kbc.connect_closing() as conn:
        archived = kb.create_task(conn, title="archived on default")
    with kbc.connect_closing(board="other") as other:
        live = kb.create_task(other, title="live on other")
    for board_slug, task_id, status in ((None, archived, "archived"), ("other", live, "ready")):
        with kbc.connect_closing(board=board_slug) as conn, kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status=?, workspace_kind='scratch', workspace_path=? WHERE id=?",
                (status, str(shared), task_id),
            )
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self.name == "boards":
            raise PermissionError("boards dir unreadable")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    with kbc.connect_closing() as conn:
        kbw._cleanup_workspace(conn, archived)
    assert (shared / "note.txt").read_text(encoding="utf-8") == "still needed"
