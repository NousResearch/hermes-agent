"""Tests for WorktreeService."""

import subprocess
import time
from pathlib import Path

from hermes_cli.code.worktree_service import WorktreeService
from hermes_state import SessionDB


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True, capture_output=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


def test_detect_capabilities_outside_git_repo(tmp_path):
    caps = WorktreeService.detect_git_capabilities(tmp_path)
    assert caps["is_git_repo"] is False
    assert caps["supports_worktree"] is False


def test_detect_capabilities_git_repo_and_dirty_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    caps = WorktreeService.detect_git_capabilities(repo)
    assert caps["is_git_repo"] is True
    assert caps["has_commits"] is True
    assert caps["is_dirty"] is False

    (repo / "dirty.txt").write_text("x", encoding="utf-8")
    dirty_caps = WorktreeService.detect_git_capabilities(repo)
    assert dirty_caps["is_dirty"] is True


def test_workspace_capabilities_fallback_when_missing_workspace(tmp_path):
    service = WorktreeService(db_path=tmp_path / "state.db")
    caps = service.detect_capabilities_for_workspace("missing")
    assert caps["is_git_repo"] is False


def test_checkpoint_metadata_creation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        now = time.time()
        db._conn.execute(
            """
            INSERT INTO code_workspaces
                (id, name, owner, repo, path, git_remote, repo_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ws-1", "repo", "owner", "repo", str(repo), "", "", now, now),
        )
        db._conn.commit()
    finally:
        db.close()

    service = WorktreeService(db_path=tmp_path / "state.db")
    checkpoint = service.create_checkpoint_metadata("ws-1", name="before-change")
    assert checkpoint["workspace_id"] == "ws-1"
    assert checkpoint["name"] == "before-change"
    assert checkpoint["git_commit"]
