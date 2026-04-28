"""Tests for WorktreeService and GitCapabilities detection."""

import subprocess
import pytest
from pathlib import Path


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo with one commit."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)
    (path / "README.md").write_text("# test")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


class TestGitCapabilitiesDetection:
    def test_non_git_directory(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        non_git = tmp_path / "non_git"
        non_git.mkdir()
        caps = detect_git_capabilities(non_git)
        assert caps.is_git_repo is False
        assert caps.supports_worktree is False
        assert caps.current_branch is None

    def test_non_existent_path(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        caps = detect_git_capabilities(tmp_path / "does_not_exist")
        assert caps.is_git_repo is False

    def test_git_repo_detection(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        repo = tmp_path / "myrepo"
        repo.mkdir()
        _init_git_repo(repo)

        caps = detect_git_capabilities(repo)
        assert caps.is_git_repo is True
        assert caps.has_commits is True
        assert caps.current_branch is not None  # main or master
        assert caps.toplevel is not None

    def test_clean_repo_not_dirty(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        repo = tmp_path / "clean"
        repo.mkdir()
        _init_git_repo(repo)

        caps = detect_git_capabilities(repo)
        assert caps.is_dirty is False

    def test_dirty_repo_is_dirty(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        repo = tmp_path / "dirty"
        repo.mkdir()
        _init_git_repo(repo)
        (repo / "new_file.py").write_text("x = 1")

        caps = detect_git_capabilities(repo)
        assert caps.is_dirty is True

    def test_to_dict(self, tmp_path):
        from hermes_cli.code.worktree_service import detect_git_capabilities
        repo = tmp_path / "dicttest"
        repo.mkdir()
        _init_git_repo(repo)
        caps = detect_git_capabilities(repo)
        d = caps.to_dict()
        assert "is_git_repo" in d
        assert "supports_worktree" in d
        assert "current_branch" in d
        assert d["is_git_repo"] is True


class TestWorktreeService:
    @pytest.fixture()
    def repo_workspace(self, tmp_path):
        """Create a workspace in a git repo."""
        repo = tmp_path / "myrepo"
        repo.mkdir()
        _init_git_repo(repo)
        return repo

    @pytest.fixture()
    def service_with_workspace(self, tmp_path, repo_workspace):
        from hermes_state import WorkspaceDB
        from hermes_cli.code.worktree_service import WorktreeService

        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        ws = wdb.upsert_workspace(
            path=str(repo_workspace),
            name="myrepo",
            is_git_repo=True,
            branch="main",
            detected_stack=["python"],
        )
        wdb.close()
        svc = WorktreeService(db_path=db_path)
        return svc, ws["id"]

    def test_detect_capabilities(self, service_with_workspace):
        svc, ws_id = service_with_workspace
        caps = svc.detect_capabilities(ws_id)
        assert caps["is_git_repo"] is True
        assert caps["has_commits"] is True

    def test_detect_capabilities_missing_workspace_returns_degraded(self, tmp_path):
        from hermes_cli.code.worktree_service import WorktreeService
        svc = WorktreeService(db_path=tmp_path / "state.db")
        # No workspace registered → should degrade gracefully
        caps = svc.detect_capabilities("nonexistent-workspace-id")
        assert caps["is_git_repo"] is False

    def test_create_checkpoint(self, service_with_workspace):
        svc, ws_id = service_with_workspace
        checkpoint = svc.create_checkpoint(ws_id, label="before-refactor")
        assert checkpoint["id"]
        assert checkpoint["label"] == "before-refactor"
        assert checkpoint["git"]["is_git_repo"] is True
        assert "head_sha" in checkpoint  # has commits

    def test_list_worktrees_returns_list(self, service_with_workspace):
        svc, ws_id = service_with_workspace
        worktrees = svc.list_worktrees(ws_id)
        assert isinstance(worktrees, list)
        # At minimum the main worktree is listed
        assert len(worktrees) >= 1

    def test_list_worktrees_non_git_returns_empty(self, tmp_path):
        from hermes_state import WorkspaceDB
        from hermes_cli.code.worktree_service import WorktreeService

        non_git = tmp_path / "notgit"
        non_git.mkdir()
        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        ws = wdb.upsert_workspace(
            path=str(non_git),
            name="notgit",
            is_git_repo=False,
            branch=None,
            detected_stack=[],
        )
        wdb.close()
        svc = WorktreeService(db_path=db_path)
        worktrees = svc.list_worktrees(ws["id"])
        assert worktrees == []

    def test_prepare_task_branch_creates_branch(self, service_with_workspace):
        svc, ws_id = service_with_workspace
        result = svc.prepare_task_branch(ws_id, branch_name="task/test-feature")
        # WorktreeService wraps prepare_branch which returns branch_name key
        branch = result.get("branch") or result.get("branch_name")
        assert branch == "task/test-feature"

    def test_prepare_task_branch_non_git_raises(self, tmp_path):
        from hermes_state import WorkspaceDB
        from hermes_cli.code.worktree_service import WorktreeService

        non_git = tmp_path / "notgit"
        non_git.mkdir()
        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        ws = wdb.upsert_workspace(
            path=str(non_git),
            name="notgit",
            is_git_repo=False,
            branch=None,
            detected_stack=[],
        )
        wdb.close()
        svc = WorktreeService(db_path=db_path)
        with pytest.raises(ValueError, match="not a Git repository"):
            svc.prepare_task_branch(ws["id"], branch_name="task/x")


class TestParseWorktreeList:
    def test_parse_single_worktree(self):
        from hermes_cli.code.worktree_service import _parse_worktree_list
        output = "worktree /home/user/myrepo\nHEAD abc123\nbranch refs/heads/main\n\n"
        result = _parse_worktree_list(output)
        assert len(result) == 1
        assert result[0]["path"] == "/home/user/myrepo"
        assert result[0]["head"] == "abc123"
        assert result[0]["branch"] == "refs/heads/main"

    def test_parse_multiple_worktrees(self):
        from hermes_cli.code.worktree_service import _parse_worktree_list
        output = (
            "worktree /main\nHEAD abc\nbranch refs/heads/main\n\n"
            "worktree /wt1\nHEAD def\nbranch refs/heads/feature\n\n"
        )
        result = _parse_worktree_list(output)
        assert len(result) == 2
