"""Invariant tests for dispatcher-owned worktree base assignment."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_assigned_worktree_base_cannot_change_on_retry(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "base-test@example.com")
    _git(repo, "config", "user.name", "Base Test")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    original_sha = _git(repo, "rev-parse", "HEAD")
    branch = "codex/immutable-base"
    _git(repo, "checkout", "-b", branch)
    prefix = f"branch.{branch}.hermes-kanban-base"
    _git(repo, "config", prefix + "-ref", "refs/heads/main")
    _git(repo, "config", prefix + "-sha", original_sha)

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Immutable base",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
        )
        kbw.set_worktree_base(conn, task_id, repo, branch)
        kbw.set_worktree_base(conn, task_id, repo, branch)

        changed_sha = "f" * 40
        _git(repo, "config", prefix + "-ref", "refs/remotes/origin/stable")
        _git(repo, "config", prefix + "-sha", changed_sha)
        with pytest.raises(RuntimeError, match="cannot change"):
            kbw.set_worktree_base(conn, task_id, repo, branch)

        task = kb.get_task(conn, task_id)
        assert task.workspace_base_ref == "refs/heads/main"
        assert task.workspace_base_sha == original_sha
