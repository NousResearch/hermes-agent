"""``hermes kanban gc`` recovers residual scratch workspaces of done tasks.

Completion cleanup is best-effort: when it is missed or deferred (for example
while a child was still active) and no later transition retries it, the
``done`` task's scratch directory survives. gc is the backstop for that
residue, under the same containment and active-child safety as completion.
"""

import argparse
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


def _gc() -> int:
    return kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0))


def _scratch_task(conn, *, status: str, title: str = "t") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title)
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "artifact.txt").write_text("residue", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status=?, workspace_kind='scratch', workspace_path=? WHERE id=?",
            (status, str(ws), tid),
        )
    return tid, ws


def test_gc_removes_residual_done_scratch_workspace(board, capsys):
    with kbc.connect_closing() as conn:
        _tid, ws = _scratch_task(conn, status="done")
    assert _gc() == 0
    assert not ws.exists()
    assert "GC complete: 1 workspace(s)" in capsys.readouterr().out


def test_gc_keeps_done_parent_scratch_while_a_child_is_active(board, capsys):
    with kbc.connect_closing() as conn:
        parent, ws = _scratch_task(conn, status="done", title="parent")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0] not in (
            "done", "archived", "failed", "cancelled",
        )
    assert _gc() == 0
    assert (ws / "artifact.txt").read_text(encoding="utf-8") == "residue"
    assert "GC complete: 0 workspace(s)" in capsys.readouterr().out

    # Once the child is terminal, the next gc recovers the parent's residue.
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
    assert _gc() == 0
    assert not ws.exists()


def test_gc_leaves_non_terminal_and_non_scratch_done_workspaces(board):
    with kbc.connect_closing() as conn:
        _running, running_ws = _scratch_task(conn, status="running")
        done_dir = board / "user-dir"
        done_dir.mkdir()
        (done_dir / "keep.txt").write_text("user data", encoding="utf-8")
        tid = kb.create_task(conn, title="done dir")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='dir', workspace_path=? WHERE id=?",
                (str(done_dir), tid),
            )
    assert _gc() == 0
    assert running_ws.is_dir()
    assert (done_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_gc_refuses_done_scratch_path_outside_managed_roots(board):
    outside = board / "src"
    outside.mkdir()
    (outside / "main.py").write_text("print()", encoding="utf-8")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="misconfigured")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='scratch', workspace_path=? "
                "WHERE id=?",
                (str(outside), tid),
            )
    assert _gc() == 0
    assert (outside / "main.py").exists()


def test_gc_archived_scratch_behaviour_unchanged(board, capsys):
    with kbc.connect_closing() as conn:
        _tid, ws = _scratch_task(conn, status="archived")
    assert _gc() == 0
    assert not ws.exists()
    assert "GC complete: 1 workspace(s)" in capsys.readouterr().out
