"""Tests for worktree base-ref resolution — branch from the fresh remote tip.

A worktree created off the standalone clone's local ``HEAD`` roots the new
branch on a stale base when that clone lags the remote. ``_resolve_worktree_base``
fetches and branches from the remote tip instead so the worktree starts current.

These tests exercise the REAL ``cli._resolve_worktree_base`` /
``cli._setup_worktree`` against a real local "remote" repo (so ``git fetch``
works offline in the hermetic sandbox), proving the worktree includes commits
that exist on the remote but not on the stale local HEAD.
"""

import subprocess
from pathlib import Path

import pytest

import cli


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)


def _commit(repo, name, msg):
    (Path(repo) / name).write_text(msg + "\n")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", msg], repo)


def _head(repo):
    return _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()


@pytest.fixture
def remote_and_clone(tmp_path):
    """A bare 'remote' + a clone that is intentionally BEHIND the remote.

    Returns (clone_path, remote_head_sha, stale_local_head_sha).
    """
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _run(["git", "init"], seed)
    _run(["git", "config", "user.email", "t@t.com"], seed)
    _run(["git", "config", "user.name", "T"], seed)
    # Pin the seed repo's branch name so push + remote default are 'main'.
    _run(["git", "checkout", "-b", "main"], seed)
    _commit(seed, "README.md", "base commit")
    _run(["git", "init", "--bare", str(remote)], tmp_path)
    _run(["git", "remote", "add", "origin", str(remote)], seed)
    _run(["git", "push", "origin", "main"], seed)
    # Set the bare remote's default branch so a clone gets origin/HEAD ->
    # origin/main and a tracking branch (mirrors a real GitHub remote).
    _run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], remote)

    # Clone it (this clone tracks origin/main).
    clone = tmp_path / "clone"
    _run(["git", "clone", str(remote), str(clone)], tmp_path)
    _run(["git", "config", "user.email", "t@t.com"], clone)
    _run(["git", "config", "user.name", "T"], clone)
    stale_local_head = _head(clone)

    # Advance the REMOTE past the clone (simulating other merges landing on
    # main while this clone sat stale).
    _commit(seed, "feature.txt", "remote-only commit")
    _run(["git", "push", "origin", "main"], seed)
    remote_head = _head(seed)

    assert remote_head != stale_local_head
    return clone, remote_head, stale_local_head


class TestResolveWorktreeBase:
    def test_resolves_to_fetched_upstream(self, remote_and_clone):
        clone, remote_head, stale_local_head = remote_and_clone
        base_ref, label = cli._resolve_worktree_base(str(clone))
        # Should resolve to the upstream tracking ref and have fetched it.
        assert base_ref == "origin/main"
        assert "fetched" in label
        # The fetched ref now points at the remote tip, not the stale local HEAD.
        resolved = _run(["git", "rev-parse", base_ref], clone).stdout.strip()
        assert resolved == remote_head
        assert resolved != stale_local_head

    def test_falls_back_to_head_without_remote(self, tmp_path):
        repo = tmp_path / "no-remote"
        repo.mkdir()
        _run(["git", "init"], repo)
        _run(["git", "config", "user.email", "t@t.com"], repo)
        _run(["git", "config", "user.name", "T"], repo)
        _commit(repo, "README.md", "only commit")
        base_ref, label = cli._resolve_worktree_base(str(repo))
        assert base_ref == "HEAD"
        assert "HEAD" in label


class TestSetupWorktreeSyncBase:
    def test_sync_true_branches_from_remote_tip(self, remote_and_clone, monkeypatch):
        clone, remote_head, stale_local_head = remote_and_clone
        info = cli._setup_worktree(str(clone), sync_base=True)
        assert info is not None
        # The new worktree's HEAD must be the REMOTE tip, not the stale local one.
        wt_head = _head(info["path"])
        assert wt_head == remote_head, "worktree should start from the fetched remote tip"
        assert wt_head != stale_local_head
        # And it must contain the remote-only file.
        assert (Path(info["path"]) / "feature.txt").exists()

    def test_sync_false_branches_from_local_head(self, remote_and_clone):
        clone, remote_head, stale_local_head = remote_and_clone
        info = cli._setup_worktree(str(clone), sync_base=False)
        assert info is not None
        # Opted out -> branch from the stale local HEAD (old behavior).
        wt_head = _head(info["path"])
        assert wt_head == stale_local_head
        assert not (Path(info["path"]) / "feature.txt").exists()

    def test_default_is_sync_true(self, remote_and_clone):
        """The default path (no sync_base arg) branches from the remote tip."""
        clone, remote_head, _ = remote_and_clone
        info = cli._setup_worktree(str(clone))
        assert info is not None
        assert _head(info["path"]) == remote_head

    def test_workspace_name_is_reflected_in_worktree_and_branch(self, remote_and_clone):
        clone, _, _ = remote_and_clone
        info = cli._setup_worktree(
            str(clone), sync_base=False, workspace_name="JARVIS Workspace!"
        )
        assert info is not None

        leaf = Path(info["path"]).name
        assert leaf.startswith("hermes-jarvis-workspace-")
        assert info["branch"].startswith("hermes/hermes-jarvis-workspace-")
        assert info["workspace_slug"] == "jarvis-workspace"

    def test_workspace_name_falls_back_to_repo_directory(self, remote_and_clone):
        clone, _, _ = remote_and_clone
        info = cli._setup_worktree(str(clone), sync_base=False)
        assert info is not None

        leaf = Path(info["path"]).name
        assert leaf.startswith("hermes-clone-")
        assert info["branch"].startswith("hermes/hermes-clone-")
        assert info["workspace_slug"] == "clone"

    def test_workspace_name_lookup_does_not_create_projects_db(self, remote_and_clone):
        from hermes_cli import projects_db

        clone, _, _ = remote_and_clone
        db_path = projects_db.projects_db_path()
        assert not db_path.exists()

        assert cli._workspace_name_for_worktree(str(clone)) == "clone"
        assert not db_path.exists()

    def test_workspace_name_uses_owning_project_slug(self, remote_and_clone):
        from hermes_cli import projects_db

        clone, _, _ = remote_and_clone
        with projects_db.connect_closing() as conn:
            projects_db.create_project(
                conn,
                name="Bob Workspace",
                slug="bob-workspace",
                primary_path=str(clone),
            )

        info = cli._setup_worktree(str(clone), sync_base=False)
        assert info is not None

        leaf = Path(info["path"]).name
        assert leaf.startswith("hermes-bob-workspace-")
        assert info["branch"].startswith("hermes/hermes-bob-workspace-")
        assert info["workspace_slug"] == "bob-workspace"
