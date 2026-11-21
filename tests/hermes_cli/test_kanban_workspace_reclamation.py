"""Reclamation contract for a finished task's workspace.

``complete_task`` removes a done scratch workspace; every case it cannot prove
safe leaves the directory with a recorded reason, and ``hermes kanban gc``
collects those leftovers. Each test below pins one half of that contract: the
Yes cases (removed) and the No cases (kept), so a future widening of the
deletion predicate fails here instead of on a user's disk.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _gc_args(event_days=0, log_days=0):
    return argparse.Namespace(event_retention_days=event_days, log_retention_days=log_days)


def _scratch_task(conn, title, *, status=None):
    """A scratch task whose managed workspace exists on disk with one file."""
    tid = kb.create_task(conn, title=title, workspace_kind="scratch")
    ws = kbw.resolve_workspace(kb.get_task(conn, tid))
    kbw.set_workspace_path(conn, tid, str(ws))
    (Path(ws) / "work.txt").write_text("work", encoding="utf-8")
    if status:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
    return tid, Path(ws)


def _events(conn, tid, kind):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id", (tid, kind)
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def _archive_task(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))


def _live_process_in(path: Path):
    """A real process whose cwd is *path* (so it is not this run's process tree)."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------- Yes cases


def test_completion_removes_done_scratch_and_keeps_the_record(board):
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "finishes cleanly")
        attachment_dir = kb.task_attachments_dir(tid)
        attachment_dir.mkdir(parents=True, exist_ok=True)
        evidence = attachment_dir / "evidence.txt"
        evidence.write_text("kept", encoding="utf-8")

        assert kb.complete_task(conn, tid, result="done") is True

        assert not ws.exists(), "a done scratch workspace must not stay on disk"
        row = kb.get_task(conn, tid)
        assert row.status == "done" and row.workspace_path == str(ws)
        assert _events(conn, tid, "completed"), "the completion event must survive"
        assert evidence.read_text(encoding="utf-8") == "kept"


def test_completion_collects_a_deferred_parent_workspace(board):
    """Parent defers while its child is live; the child's completion collects it."""
    with kbc.connect_closing() as conn:
        parent, pws = _scratch_task(conn, "parent")
        (pws / "handoff.md").write_text("read me", encoding="utf-8")
        child = kb.create_task(conn, title="child", workspace_kind="scratch", parents=[parent])
        cws = kbw.resolve_workspace(kb.get_task(conn, child))
        kbw.set_workspace_path(conn, child, str(cws))

        assert kb.complete_task(conn, parent, result="parent done") is True
        assert pws.exists(), "the live child may still read the parent's handoff"
        assert _events(conn, parent, "workspace_cleanup_deferred_children")

        assert kb.complete_task(conn, child, result="child done") is True
        assert not cws.exists()
        assert not pws.exists(), "the deferred parent workspace must be collected"


def test_gc_collects_a_leftover_done_scratch(board, capsys):
    """A completion that could not remove its dir is recovered by the standard gc."""
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "leftover", status="done")
    assert ws.is_dir()

    assert kanban_ops._cmd_gc(_gc_args()) == 0

    assert not ws.exists()
    assert "1 workspace(s)" in capsys.readouterr().out
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"


def test_gc_still_collects_an_archived_scratch(board):
    with kbc.connect_closing() as conn:
        _tid, ws = _scratch_task(conn, "archived", status="archived")
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert not ws.exists()


# --------------------------------------------------------------------------- No cases


@pytest.mark.parametrize("status", ["todo", "ready", "running", "blocked", "scheduled", "triage", "review"])
def test_gc_keeps_a_workspace_of_any_non_terminal_status(board, status):
    with kbc.connect_closing() as conn:
        _tid, ws = _scratch_task(conn, f"{status} task", status=status)
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (ws / "work.txt").exists()


def test_gc_keeps_a_done_scratch_a_live_child_still_needs(board):
    with kbc.connect_closing() as conn:
        parent, pws = _scratch_task(conn, "done parent", status="done")
        kb.create_task(conn, title="live child", parents=[parent], initial_status="blocked")
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert pws.exists()


def test_gc_keeps_a_done_scratch_another_live_task_shares(board):
    with kbc.connect_closing() as conn:
        done_tid, ws = _scratch_task(conn, "done sharer", status="done")
        live = kb.create_task(conn, title="live sharer", workspace_kind="scratch")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (str(ws), live))
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (ws / "work.txt").exists()
    with kbc.connect_closing() as conn:
        assert [e["reason"] for e in _events(conn, done_tid, "workspace_cleanup_deferred_shared")] == ["shared"]


def test_gc_keeps_a_scratch_path_outside_the_managed_roots(board, tmp_path):
    outside = tmp_path / "user-source-tree"
    outside.mkdir()
    (outside / "keep.txt").write_text("user data", encoding="utf-8")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="scratch on a user tree", workspace_kind="scratch")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_path=? WHERE id=?", (str(outside), tid),
            )
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (outside / "keep.txt").exists()


def test_gc_keeps_the_workspaces_root_itself(board):
    root = kb.workspaces_root()
    sibling = root / "t_other"
    sibling.mkdir(parents=True)
    (sibling / "work.txt").write_text("another task's scratch", encoding="utf-8")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="done on the root", workspace_kind="scratch")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_path=? WHERE id=?", (str(root), tid),
            )
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (sibling / "work.txt").exists()


def test_gc_keeps_a_scratch_dir_a_live_process_is_working_in(board):
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "busy", status="done")
    proc = _live_process_in(ws)
    try:
        assert kanban_ops._cmd_gc(_gc_args()) == 0
        assert (ws / "work.txt").exists(), "a live process keeps its tree"
        with kbc.connect_closing() as conn:
            assert [e["reason"] for e in _events(conn, tid, "workspace_cleanup_deferred_busy")] == ["busy"]
    finally:
        _stop(proc)
    # The deferral is retryable: the same standard command now collects it.
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert not ws.exists()


def test_completion_defers_while_another_process_works_in_the_workspace(board):
    """The worker that just completed is this process tree; a third party is not."""
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "still being read")
        proc = _live_process_in(ws)
        try:
            assert kb.complete_task(conn, tid, result="done") is True
            assert (ws / "work.txt").exists()
            assert _events(conn, tid, "workspace_cleanup_deferred_busy")
        finally:
            _stop(proc)
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert not ws.exists()


def test_completion_removes_the_workspace_it_is_not_its_own_blocker(board, tmp_path):
    """A dispatched worker runs *inside* its workspace: that is not a third party."""
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "worker cwd")
        cwd = os.getcwd()
        try:
            os.chdir(ws)
            assert kb.complete_task(conn, tid, result="done") is True
        finally:
            os.chdir(cwd)
    assert not ws.exists()


def test_cleanup_records_a_failed_removal_instead_of_swallowing_it(board, monkeypatch):
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, "unremovable")

        def _refuse(_path, **_kwargs):
            raise OSError(16, "Device or resource busy")

        monkeypatch.setattr(shutil, "rmtree", _refuse)
        kbw._cleanup_workspace(conn, tid)

        assert ws.is_dir(), "a failed removal keeps the tree for the gc retry"
        payload = _events(conn, tid, "workspace_cleanup_failed")
        assert payload and payload[0]["reason"] == "EBUSY"
        assert payload[0]["path"] == str(ws)
        assert "work.txt" not in json.dumps(payload), "no deleted file name may leak"


# --------------------------------------------------------------------------- worktrees


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _repo_with_worktree(tmp_path, name):
    """A repo with an ``origin``-backed worktree checked out on its own branch."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git("init", "--bare", "-q", "--initial-branch=main", str(origin), cwd=tmp_path)
    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "-q", "--initial-branch=main", str(seed), cwd=tmp_path)
    _git("config", "user.email", "test@example.invalid", cwd=seed)
    _git("config", "user.name", "test", cwd=seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=seed)
    _git("commit", "-qm", "seed", cwd=seed)
    _git("remote", "add", "origin", str(origin), cwd=seed)
    _git("push", "-q", "-u", "origin", "main", cwd=seed)

    tree = tmp_path / "tree" / name
    tree.parent.mkdir()
    _git("worktree", "add", "-q", "-b", f"wt/{name}", str(tree), "HEAD", cwd=seed)
    _git("push", "-q", "-u", "origin", f"wt/{name}", cwd=tree)
    return tree


def test_gc_removes_a_clean_pushed_done_worktree(board, tmp_path):
    tree = _repo_with_worktree(tmp_path, "t_clean")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="clean worktree")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='worktree', workspace_path=?, "
                "branch_name=? WHERE id=?",
                (str(tree), "wt/t_clean", tid),
            )
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert not tree.exists()


def test_gc_keeps_a_dirty_done_worktree(board, tmp_path):
    tree = _repo_with_worktree(tmp_path, "t_dirty")
    (tree / "uncommitted.txt").write_text("scribble", encoding="utf-8")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="dirty worktree")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='worktree', workspace_path=?, "
                "branch_name=? WHERE id=?",
                (str(tree), "wt/t_dirty", tid),
            )
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (tree / "uncommitted.txt").exists()


def test_gc_keeps_a_done_worktree_with_an_unpushed_commit(board, tmp_path):
    tree = _repo_with_worktree(tmp_path, "t_unpushed")
    (tree / "local.txt").write_text("local only", encoding="utf-8")
    _git("config", "user.email", "test@example.invalid", cwd=tree)
    _git("config", "user.name", "test", cwd=tree)
    _git("add", "-A", cwd=tree)
    _git("commit", "-qm", "never pushed", cwd=tree)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="unpushed worktree")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='worktree', workspace_path=?, "
                "branch_name=? WHERE id=?",
                (str(tree), "wt/t_unpushed", tid),
            )
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    assert (tree / "local.txt").exists()
