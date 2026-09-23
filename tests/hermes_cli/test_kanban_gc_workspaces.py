"""Workspace recovery invariants for ``hermes kanban gc``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _gc() -> int:
    return kanban_ops._cmd_gc(
        argparse.Namespace(event_retention_days=0, log_retention_days=0)
    )


def _workspace(conn, *, title: str, status: str) -> tuple[str, Path]:
    task_id = kb.create_task(conn, title=title, workspace_kind="scratch")
    workspace = kb.workspaces_root() / task_id
    workspace.mkdir(parents=True)
    (workspace / "residue.txt").write_text("residue\n", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status=?, workspace_path=? WHERE id=?",
            (status, str(workspace), task_id),
        )
    return task_id, workspace


def test_gc_reaps_terminal_scratch_residue_without_deleting_active_parent(
    board: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Done residue is recoverable, archived behavior remains, and a parent
    stays intact while a non-terminal child still needs its handoff artifacts."""
    with kbc.connect_closing() as conn:
        _done_id, done_workspace = _workspace(conn, title="done residue", status="done")
        _archived_id, archived_workspace = _workspace(
            conn, title="archived residue", status="archived"
        )
        parent_id = kb.create_task(conn, title="done parent", workspace_kind="scratch")
        child_id = kb.create_task(conn, title="active child", workspace_kind="scratch")
        kb.link_tasks(conn, parent_id, child_id)
        parent_workspace = kb.workspaces_root() / parent_id
        parent_workspace.mkdir(parents=True)
        (parent_workspace / "handoff.txt").write_text("needed\n", encoding="utf-8")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent_id,))

        foreign_id = kb.create_task(
            conn, title="other board residue", workspace_kind="scratch"
        )
        foreign_workspace = (
            board / "kanban" / "boards" / "other" / "workspaces" / foreign_id
        )
        foreign_workspace.mkdir(parents=True)
        (foreign_workspace / "foreign.txt").write_text("keep\n", encoding="utf-8")

        done_dir = board / "persistent-dir"
        done_dir.mkdir()
        done_dir_id = kb.create_task(
            conn,
            title="done persistent dir",
            workspace_kind="dir",
            workspace_path=str(done_dir),
        )

        worktree_id = kb.create_task(
            conn,
            title="preserved archived worktree",
            workspace_kind="worktree",
            workspace_path=str(kb.workspaces_root() / "preserved-worktree"),
        )
        preserved_worktree = kb.workspaces_root() / "preserved-worktree"
        preserved_worktree.mkdir(parents=True)

        file_id = kb.create_task(
            conn, title="scratch path is a file", workspace_kind="scratch"
        )
        file_workspace = kb.workspaces_root() / file_id
        file_workspace.write_text("not a directory\n", encoding="utf-8")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', workspace_path=? WHERE id=?",
                (str(foreign_workspace), foreign_id),
            )
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done_dir_id,))
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (worktree_id,))
            conn.execute(
                "UPDATE tasks SET status='archived', workspace_path=? WHERE id=?",
                (str(file_workspace), file_id),
            )

    assert _gc() == 0

    captured = capsys.readouterr()
    assert not done_workspace.exists()
    assert not archived_workspace.exists()
    assert (parent_workspace / "handoff.txt").is_file()
    assert (foreign_workspace / "foreign.txt").is_file()
    assert done_dir.is_dir()
    assert preserved_worktree.is_dir()
    assert file_workspace.is_file()
    assert "2 workspace(s)" in captured.out
    assert "4 workspace(s) retained/skipped" in captured.out
    assert parent_id in captured.err
    assert foreign_id in captured.err
    assert worktree_id in captured.err
    assert file_id in captured.err


def test_gc_does_not_count_or_log_a_surviving_workspace_as_removed(
    board: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A no-op/failed rmtree leaves truthful CLI and completion-cleanup reports."""
    with kbc.connect_closing() as conn:
        task_id, workspace = _workspace(conn, title="undeletable residue", status="done")

    def denied_rmtree(*_args, **_kwargs):
        raise PermissionError("simulated permission failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(shutil, "rmtree", denied_rmtree)
        patcher.setattr(
            kbw.os.path,
            "lexists",
            lambda _path: False,
            raising=True,
        )
        assert kbw._rmtree_workspace(workspace) is False

    monkeypatch.setattr(shutil, "rmtree", lambda *_args, **_kwargs: None)

    assert _gc() == 0

    captured = capsys.readouterr()
    assert workspace.is_dir()
    assert "0 workspace(s)" in captured.out
    assert task_id in captured.err

    caplog.clear()
    with kbc.connect_closing() as conn:
        kbw._cleanup_workspace(conn, task_id)
    assert workspace.is_dir()
    assert any(
        task_id in record.getMessage() and "failed" in record.getMessage().lower()
        for record in caplog.records
    )
    assert not any(
        "Removed scratch workspace" in record.getMessage() for record in caplog.records
    )
