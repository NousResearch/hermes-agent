"""Regression test: _cleanup_workspace must NOT rmtree the scratch dir inline.

The inline rmtree in complete_task was deleting the cwd out from under
the live worker, causing FileNotFoundError storms in os.getcwd() that
crashed the worker mid-completion. The fix moves rmtree to a
dispatcher-driven gc function that runs out-of-process.

See PR sahilm-ti/hermes-agent#4 (symptom belt) and the followup PR for
the full context.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_cleanup_workspace_does_not_rmtree(tmp_path, monkeypatch):
    """complete_task -> _cleanup_workspace must NOT remove the scratch dir."""
    from hermes_cli import kanban_db

    # Create a fake scratch workspace
    scratch = tmp_path / "fake-scratch"
    scratch.mkdir()
    sentinel = scratch / "sentinel.txt"
    sentinel.write_text("alive")

    # Mock _cleanup_worker_tmux so we don't shell out
    monkeypatch.setattr(kanban_db, "_cleanup_worker_tmux", lambda conn, tid: None)

    conn = kanban_db.connect(tmp_path / "test.db")
    try:
        # Insert a fake task with this scratch path
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO tasks (id, title, assignee, status, workspace_kind, workspace_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))",
                ("t_test", "test", "tester", "running", "scratch", str(scratch)),
            )

        kanban_db._cleanup_workspace(conn, "t_test")

        # CRITICAL: scratch dir + sentinel must still exist
        assert scratch.is_dir(), "scratch dir was removed — the self-delete bug is back"
        assert sentinel.exists(), "scratch contents were removed"
    finally:
        conn.close()


def test_gc_scratch_workspaces_reaps_done_tasks(tmp_path, monkeypatch):
    """gc_scratch_workspaces removes scratch dirs for settled tasks."""
    from hermes_cli import kanban_db

    monkeypatch.setattr(kanban_db, "_cleanup_worker_tmux", lambda conn, tid: None)

    done_scratch = tmp_path / "done-scratch"
    done_scratch.mkdir()
    (done_scratch / "old.txt").write_text("stale")

    running_scratch = tmp_path / "running-scratch"
    running_scratch.mkdir()
    (running_scratch / "active.txt").write_text("live")

    conn = kanban_db.connect(tmp_path / "test.db")
    try:
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO tasks (id, title, assignee, status, workspace_kind, workspace_path, claim_lock, created_at) "
                "VALUES (?, ?, ?, 'done', 'scratch', ?, NULL, strftime('%s','now'))",
                ("t_done", "done task", "tester", str(done_scratch)),
            )
            conn.execute(
                "INSERT INTO tasks (id, title, assignee, status, workspace_kind, workspace_path, claim_lock, created_at) "
                "VALUES (?, ?, ?, 'running', 'scratch', ?, 'host:123', strftime('%s','now'))",
                ("t_running", "running task", "tester", str(running_scratch)),
            )

        reaped = kanban_db.gc_scratch_workspaces(conn)

        assert reaped == 1
        assert not done_scratch.exists(), "done scratch should be reaped"
        assert running_scratch.exists(), "running scratch must NOT be reaped"
    finally:
        conn.close()


def test_gc_scratch_workspaces_skips_claimed_tasks(tmp_path, monkeypatch):
    """A task in done status but with a claim still held must not be reaped."""
    from hermes_cli import kanban_db

    scratch = tmp_path / "claimed-scratch"
    scratch.mkdir()

    conn = kanban_db.connect(tmp_path / "test.db")
    try:
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO tasks (id, title, assignee, status, workspace_kind, workspace_path, claim_lock, created_at) "
                "VALUES (?, ?, ?, 'done', 'scratch', ?, 'host:123', strftime('%s','now'))",
                ("t_claimed", "claimed task", "tester", str(scratch)),
            )

        reaped = kanban_db.gc_scratch_workspaces(conn)

        assert reaped == 0
        assert scratch.is_dir()
    finally:
        conn.close()
