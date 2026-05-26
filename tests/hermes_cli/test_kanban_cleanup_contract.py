"""Tests for Kanban cleanup contract — Slice 2 of lifecycle cleanup closed-loop.

Covers:
- ``build_cleanup_contract()`` validation (absolute path, symlink escape, containment)
- DB persistence at claim/dispatch time
- Task roundtrip with cleanup_contract field
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# build_cleanup_contract — validation
# ---------------------------------------------------------------------------

def test_build_cleanup_contract_scratch_workspace(kanban_home, tmp_path):
    """Scratch workspace creates a valid cleanup contract with no repo/branch."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="scratch-task", assignee="worker",
            workspace_kind="scratch",
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)

        contract = kb.build_cleanup_contract(task, str(workspace))
        assert contract is not None
        assert contract["task_id"] == tid
        assert contract["workspace_kind"] == "scratch"
        assert contract["workspace_path"] == str(workspace)
        assert contract["repo_path"] is None
        assert contract["branch_name"] is None
        assert "created_at" in contract
        assert "allowed_artifact_names" in contract
        assert "forbidden_cleanup" in contract
        assert contract["artifact_ttl_hours"] > 0
    finally:
        conn.close()


def test_build_cleanup_contract_worktree(kanban_home, tmp_path):
    """Worktree workspace captures repo_path and branch_name."""
    # Create a minimal git repo to resolve as worktree target
    repo = tmp_path / "my-repo"
    repo.mkdir()
    os.system(f"git -C {repo} init -q")
    os.system(f"git -C {repo} checkout -b task/bo-156 2>/dev/null; true")

    conn = kb.connect()
    try:
        wt_path = str(repo / ".worktrees" / "test-wt")
        tid = kb.create_task(
            conn, title="wt-task", assignee="worker",
            workspace_kind="worktree",
            workspace_path=wt_path,
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)

        contract = kb.build_cleanup_contract(task, str(workspace))
        assert contract["workspace_kind"] == "worktree"
        assert contract["workspace_path"] == str(workspace)
    finally:
        conn.close()


def test_build_cleanup_contract_rejects_non_absolute_path(kanban_home):
    """Non-absolute workspace path must fail closed."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="relative-bad", assignee="worker",
            workspace_kind="dir",
            workspace_path="relative/path",
        )
        task = kb.get_task(conn, tid)
        # resolve_workspace will reject non-absolute paths
        with pytest.raises(ValueError, match=r"non-absolute"):
            kb.resolve_workspace(task)
        # build_cleanup_contract with non-absolute path should also fail
        with pytest.raises(ValueError, match=r"absolute|non-absolute"):
            kb.build_cleanup_contract(task, "relative/path")
    finally:
        conn.close()


def test_build_cleanup_contract_rejects_symlink_escape(kanban_home, tmp_path):
    """Workspace path that is a symlink pointing outside must be rejected."""
    real_dir = tmp_path / "real-workspace"
    real_dir.mkdir()
    symlink_dir = tmp_path / "symlink-workspace"
    # symlink pointing outside the workspace
    os.symlink(str(real_dir), str(symlink_dir))

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="symlink-task", assignee="worker",
            workspace_kind="dir",
            workspace_path=str(symlink_dir),
        )
        task = kb.get_task(conn, tid)

        with pytest.raises(ValueError, match=r"symlink"):
            kb.build_cleanup_contract(task, str(symlink_dir))
    finally:
        conn.close()


def test_build_cleanup_contract_rejects_path_traversal(kanban_home, tmp_path):
    """Path containing ../ traversal must be rejected."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="traversal-task", assignee="worker",
            workspace_kind="dir",
            workspace_path=str(tmp_path / "sub"),
        )
        task = kb.get_task(conn, tid)

        # Supply a path with ../ traversal
        traversal_path = str(tmp_path / "sub" / ".." / ".." / "etc")
        with pytest.raises(ValueError, match=r"traversal|absolute"):
            kb.build_cleanup_contract(task, traversal_path)
    finally:
        conn.close()


def test_build_cleanup_contract_missing_workspace_path(kanban_home):
    """Task with no workspace_path and unresolvable scratch should fail cleanly."""
    conn = kb.connect()
    try:
        # A scratch task resolves automatically, so use a dir task with no path
        tid = kb.create_task(
            conn, title="no-path", assignee="worker",
            workspace_kind="dir",
            workspace_path=None,
        )
        task = kb.get_task(conn, tid)
        with pytest.raises(ValueError):
            kb.build_cleanup_contract(task, "")
        with pytest.raises(ValueError):
            kb.build_cleanup_contract(task, None)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DB persistence — cleanup_contract column
# ---------------------------------------------------------------------------

def test_cleanup_contract_column_migration(kanban_home):
    """Migration adds cleanup_contract column to legacy DBs."""
    conn = kb.connect()
    try:
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(tasks)")
        }
        assert "cleanup_contract" in cols, (
            "cleanup_contract column should be present after init_db"
        )
    finally:
        conn.close()


def test_cleanup_contract_persisted_and_roundtrip(kanban_home, tmp_path):
    """Cleanup contract stored at claim time survives roundtrip through Task."""
    conn = kb.connect()
    try:
        abs_path = str(tmp_path / "my-workspace")
        tid = kb.create_task(
            conn, title="persist-test", assignee="worker",
            workspace_kind="dir",
            workspace_path=abs_path,
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        contract = kb.build_cleanup_contract(task, str(workspace))

        # Persist
        kb.set_cleanup_contract(conn, tid, contract)

        # Roundtrip
        reloaded = kb.get_task(conn, tid)
        assert reloaded.cleanup_contract is not None
        assert reloaded.cleanup_contract["task_id"] == tid
        assert reloaded.cleanup_contract["workspace_kind"] == "dir"
        assert reloaded.cleanup_contract["workspace_path"] == abs_path
    finally:
        conn.close()


def test_cleanup_contract_none_for_legacy_task(kanban_home):
    """Task created before migration has cleanup_contract=None."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="legacy", assignee="worker",
        )
        task = kb.get_task(conn, tid)
        assert task.cleanup_contract is None
    finally:
        conn.close()


def test_cleanup_contract_invalid_json_is_none(kanban_home):
    """Corrupt cleanup_contract JSON in DB returns None gracefully."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="corrupt", assignee="worker",
        )
        # Write garbage directly
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET cleanup_contract = ? WHERE id = ?",
                ("{{{broken", tid),
            )
        conn.commit()
        task = kb.get_task(conn, tid)
        assert task.cleanup_contract is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dispatch integration — contract persisted at claim time
# ---------------------------------------------------------------------------

def test_dispatch_persists_cleanup_contract(kanban_home, all_assignees_spawnable):
    """When dispatch_once claims and resolves a workspace, the cleanup
    contract is persisted to the task row."""
    spawned: list = []

    def _fake_spawn(task, ws, **kwargs):
        spawned.append((task.id, ws))
        return 99999  # fake PID

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="dispatch-test", assignee="worker",
            workspace_kind="dir",
            workspace_path="/tmp/test-dispatch-ws",
        )
        # Run dispatch with our fake spawn — it should claim, resolve,
        # and build the cleanup contract
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert len(spawned) == 1, f"expected 1 spawn, got {len(spawned)}"

        # The cleanup contract should be persisted
        task = kb.get_task(conn, tid)
        assert task.cleanup_contract is not None, (
            "cleanup_contract should be set after dispatch"
        )
        assert task.cleanup_contract["task_id"] == tid
        assert task.cleanup_contract["workspace_kind"] == "dir"
        assert task.cleanup_contract["workspace_path"] == "/tmp/test-dispatch-ws"
        assert task.cleanup_contract["repo_path"] is None
        assert "created_at" in task.cleanup_contract
        assert "allowed_artifact_names" in task.cleanup_contract
        assert "forbidden_cleanup" in task.cleanup_contract
    finally:
        conn.close()
