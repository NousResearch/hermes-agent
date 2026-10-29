"""Tests for HERMES_PROTECTED_GIT_ROOTS injection in _default_spawn.

Verifies that the dispatcher sets HERMES_PROTECTED_GIT_ROOTS in the worker
env when the board has a default_workdir that is a git repo, and that the
worker's own .worktrees/<id> workspace is NOT listed as a protected root.
"""
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_task(workspace_path: str, project_id=None) -> MagicMock:
    task = MagicMock()
    task.id = "t_test0001"
    task.assignee = "kamiyu"
    task.tenant = None
    task.branch_name = None
    task.workspace_kind = "worktree"
    task.workspace_path = workspace_path
    task.project_id = project_id
    task.goal_mode = False
    task.goal_max_turns = None
    task.max_runtime_seconds = None
    task.model_override = None
    task.provider_override = None
    task.claim_lock = None
    task.current_run_id = 1
    task.skills = []
    return task


# ---------------------------------------------------------------------------
# Unit tests for the _pg_is_inside_protected_root helper
# ---------------------------------------------------------------------------

class TestPgIsInsideProtectedRoot:
    def setup_method(self, tmp_path=None):
        from tools.approval import _pg_is_inside_protected_root
        self._fn = _pg_is_inside_protected_root

    def test_exact_match_is_protected(self, tmp_path):
        root = str(tmp_path)
        hit, r = self._fn(root, [root])
        assert hit is True
        assert r == root

    def test_child_of_root_is_protected(self, tmp_path):
        subdir = tmp_path / "subdir" / "deep"
        subdir.mkdir(parents=True)
        hit, r = self._fn(str(subdir), [str(tmp_path)])
        assert hit is True

    def test_unrelated_path_not_protected(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        root = tmp_path / "repo"
        root.mkdir()
        hit, _ = self._fn(str(other), [str(root)])
        assert hit is False

    def test_worktrees_child_is_exempt(self, tmp_path):
        # .worktrees/<id> inside a protected root should be EXEMPT
        root = tmp_path / "repo"
        worktree_dir = root / ".worktrees" / "t_abc123"
        worktree_dir.mkdir(parents=True)
        hit, _ = self._fn(str(worktree_dir), [str(root)])
        assert hit is False

    def test_worktrees_child_deep_is_exempt(self, tmp_path):
        root = tmp_path / "repo"
        deep = root / ".worktrees" / "t_abc123" / "src" / "foo"
        deep.mkdir(parents=True)
        hit, _ = self._fn(str(deep), [str(root)])
        assert hit is False

    def test_empty_roots_list(self, tmp_path):
        hit, r = self._fn(str(tmp_path), [])
        assert hit is False
        assert r is None


# ---------------------------------------------------------------------------
# Unit tests for _pg_effective_cwd
# ---------------------------------------------------------------------------

class TestPgEffectiveCwd:
    def setup_method(self, tmp_path=None):
        from tools.approval import _pg_effective_cwd
        self._fn = _pg_effective_cwd

    def test_no_leading_cd_returns_session_cwd(self):
        result = self._fn("git log --oneline", "/home/user/repo")
        assert result == "/home/user/repo"

    def test_leading_cd_absolute_overrides_session_cwd(self):
        result = self._fn("cd /tmp/work && git log", "/home/user/repo")
        assert result == "/tmp/work"

    def test_leading_cd_relative_joins_session_cwd(self):
        result = self._fn("cd src && git status", "/home/user/repo")
        assert result == "/home/user/repo/src"

    def test_non_cd_first_segment_keeps_session_cwd(self):
        result = self._fn("echo foo; cd /tmp && git reset --hard", "/home/user")
        # The first segment is `echo foo`, not a cd — guard sees session cwd
        assert result == "/home/user"

    def test_empty_command_returns_session_cwd(self):
        result = self._fn("", "/home/user")
        assert result == "/home/user"


# ---------------------------------------------------------------------------
# Unit tests for check_protected_git_roots_guard
# ---------------------------------------------------------------------------

class TestCheckProtectedGitRootsGuard:
    def setup_method(self, tmp_path=None):
        from tools.approval import check_protected_git_roots_guard
        self._fn = check_protected_git_roots_guard

    # --- Guard inactive when env var is absent ---

    def test_no_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("HERMES_PROTECTED_GIT_ROOTS", raising=False)
        assert self._fn("git reset --hard HEAD~1", "/some/repo") is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", "")
        assert self._fn("git reset --hard", "/some/repo") is None

    # --- Safe commands are allowed even when guard is active ---

    def test_git_log_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git log --oneline", str(tmp_path)) is None

    def test_git_status_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git status", str(tmp_path)) is None

    def test_git_stash_list_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git stash list", str(tmp_path)) is None

    def test_git_stash_show_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git stash show stash@{0}", str(tmp_path)) is None

    def test_git_reset_soft_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git reset --soft HEAD~1", str(tmp_path)) is None

    def test_git_reset_mixed_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git reset --mixed HEAD~1", str(tmp_path)) is None

    def test_git_checkout_new_branch_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git checkout -b feat/new", str(tmp_path)) is None

    def test_git_commit_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        assert self._fn("git commit -m 'wip'", str(tmp_path)) is None

    # --- Destructive commands are BLOCKED in protected root cwd ---

    def test_git_reset_hard_blocked_in_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git reset --hard HEAD~1", str(tmp_path))
        assert result is not None
        assert result["approved"] is False
        assert "BLOCKED" in result["message"]

    def test_git_stash_push_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git stash push -m 'wip'", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_bare_git_stash_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git stash", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_stash_pop_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git stash pop", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_checkout_branch_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git checkout main", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_checkout_file_restore_blocked(self, tmp_path, monkeypatch):
        # git checkout -- <path> discards working-tree changes
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git checkout -- src/foo.py", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_clean_force_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git clean -fd", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_branch_D_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git branch -D old-feature", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_push_force_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git push --force origin main", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_git_switch_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn("git switch main", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    # --- cd <safe_path> && dangerous git → allowed ---

    def test_cd_to_safe_path_allows_dangerous(self, tmp_path, monkeypatch):
        protected = tmp_path / "repo"
        protected.mkdir()
        safe = tmp_path / "safe"
        safe.mkdir()
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))
        # session cwd is protected root, but cd moves to /tmp/safe first
        result = self._fn(f"cd {safe} && git reset --hard", str(protected))
        assert result is None  # not blocked — effective cwd is safe

    # --- .worktrees child cwd → allowed ---

    def test_worktree_child_cwd_allowed(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        worktree = root / ".worktrees" / "t_abc123"
        worktree.mkdir(parents=True)
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
        # Session cwd is inside .worktrees — should be ALLOWED
        result = self._fn("git reset --hard HEAD~1", str(worktree))
        assert result is None

    # --- cd <protected_root> + dangerous → blocked ---

    def test_cd_to_protected_root_blocks(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        safe_start = tmp_path / "safe"
        safe_start.mkdir()
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
        # Session cwd is safe, but cd moves into the protected root
        result = self._fn(f"cd {root} && git reset --hard", str(safe_start))
        assert result is not None
        assert result["approved"] is False

    # --- Bypass patterns ---

    def test_git_dash_C_bypass_blocked(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
        # Session cwd is /tmp, but git -C <root> targets the protected root
        result = self._fn(f"git -C {root} reset --hard", "/tmp")
        assert result is not None
        assert result["approved"] is False

    def test_git_work_tree_bypass_blocked(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
        result = self._fn(
            f"git --work-tree={root} reset --hard",
            "/tmp",
        )
        assert result is not None
        assert result["approved"] is False

    def test_semicolon_chain_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        # Semicolon chain: shell still runs in the same (protected) cwd
        result = self._fn(f"echo foo; git reset --hard", str(tmp_path))
        assert result is not None
        assert result["approved"] is False

    def test_sh_c_wrapping_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(tmp_path))
        result = self._fn(
            f'sh -c "cd {tmp_path} && git reset --hard"',
            str(tmp_path),
        )
        # The pattern fires because the command text contains the pattern
        assert result is not None
        assert result["approved"] is False
