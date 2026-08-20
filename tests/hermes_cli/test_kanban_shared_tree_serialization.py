"""Regression tests for shared-tree commit isolation (audit 2026-07-26).

When a board has a git-repo ``default_workdir``, at most ONE scratch task
may run at a time. Prevents concurrent scratch workers from editing the
same shared checkout simultaneously — the shared-tree collision that
intermixed 3 tasks' uncommitted work in ~/projects/NOESIS.

Worktree tasks are always exempt (they get isolated checkouts at
``<repo>/.worktrees/<id>``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_shared_tree_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _make_git_repo(tmp_path) -> str:
    """Create a minimal git repo and return its absolute path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        [
            "git", "init", "-b", "main", str(repo),
        ],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "user.name=Test",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            "commit", "--allow-empty", "-m", "init",
        ],
        check=True, capture_output=True, text=True,
    )
    return str(repo)


def _fake_spawn(*args, **kwargs):
    return 12345


def test_no_default_workdir_no_guard(isolated_kanban_home_with_profiles):
    """Without a board default_workdir, all scratch tasks dispatch freely."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="pre-running", assignee="alpha")
        # Set first task to running to simulate in-flight scratch worker
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1'"
            )
        for i in range(3):
            kb.create_task(conn, title=f"task-{i}", assignee="alpha")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 3
    assert not res.skipped_shared_tree


def test_scratch_serialized_when_default_workdir_set(
    isolated_kanban_home_with_profiles, tmp_path
):
    """With a git-repo default_workdir, a pre-running scratch task defers
    all other scratch tasks."""
    kb = isolated_kanban_home_with_profiles
    repo = _make_git_repo(tmp_path)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test", default_workdir=repo)
        # One scratch task already running
        running_id = kb.create_task(conn, title="running", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' "
                "WHERE id = ?",
                (running_id,),
            )
        # Two more scratch tasks ready
        t1 = kb.create_task(conn, title="ready-1", assignee="alpha")
        t2 = kb.create_task(conn, title="ready-2", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 0
    assert len(res.skipped_shared_tree) == 2
    deferred_ids = {entry[0] for entry in res.skipped_shared_tree}
    assert deferred_ids == {t1, t2}


def test_worktree_tasks_exempt_from_guard(
    isolated_kanban_home_with_profiles, tmp_path
):
    """Worktree tasks dispatch freely even when a scratch task is running
    on the same shared tree."""
    kb = isolated_kanban_home_with_profiles
    repo = _make_git_repo(tmp_path)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test", default_workdir=repo)
        # Scratch task already running
        running_id = kb.create_task(conn, title="running-scratch", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' "
                "WHERE id = ?",
                (running_id,),
            )
        # Worktree task ready — should NOT be deferred
        wt_task = kb.create_task(
            conn, title="worktree-task", assignee="beta",
            workspace_kind="worktree",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == wt_task
    assert not res.skipped_shared_tree


def test_first_scratch_dispatches_second_defers(
    isolated_kanban_home_with_profiles, tmp_path
):
    """No pre-running tasks: first scratch task dispatches, second is
    deferred in the SAME tick (intra-tick tracking)."""
    kb = isolated_kanban_home_with_profiles
    repo = _make_git_repo(tmp_path)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test", default_workdir=repo)
        t1 = kb.create_task(conn, title="first", assignee="alpha")
        t2 = kb.create_task(conn, title="second", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == t1
    assert len(res.skipped_shared_tree) == 1
    assert res.skipped_shared_tree[0][0] == t2


def test_deferred_task_dispatches_after_running_completes(
    isolated_kanban_home_with_profiles, tmp_path
):
    """A deferred scratch task should dispatch on the next tick once the
    running scratch worker completes."""
    kb = isolated_kanban_home_with_profiles
    repo = _make_git_repo(tmp_path)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test", default_workdir=repo)
        running_id = kb.create_task(conn, title="running", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' "
                "WHERE id = ?",
                (running_id,),
            )
        ready_id = kb.create_task(conn, title="ready", assignee="alpha")
    # Tick 1: ready task deferred
    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res1.skipped_shared_tree) == 1
    # Simulate running task completing
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL "
                "WHERE id = ?",
                (running_id,),
            )
    # Tick 2: previously deferred task now dispatches
    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res2.spawned) == 1
    assert res2.spawned[0][0] == ready_id
    assert not res2.skipped_shared_tree


def test_dispatch_result_has_skipped_shared_tree_field():
    """Schema-level invariant: DispatchResult exposes the
    skipped_shared_tree field as a list of (task_id, running_ids) tuples."""
    from hermes_cli.kanban_db import DispatchResult
    r = DispatchResult()
    assert hasattr(r, "skipped_shared_tree")
    assert r.skipped_shared_tree == []


def test_nonexistent_default_workdir_no_guard(
    isolated_kanban_home_with_profiles
):
    """A board default_workdir that doesn't exist on disk should NOT trigger
    the guard (fails open — no serialization)."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(
            slug="default", name="Test",
            default_workdir="/nonexistent/path/that/does/not/exist",
        )
        running_id = kb.create_task(conn, title="running", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' "
                "WHERE id = ?",
                (running_id,),
            )
        t1 = kb.create_task(conn, title="ready", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 1
    assert not res.skipped_shared_tree
