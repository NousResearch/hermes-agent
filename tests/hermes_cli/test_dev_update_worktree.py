"""Tests for worktree-based ejected update (task 3.4).

Tests ``hermes_cli.dev_update`` against fixture git repos created with
real git in temp directories.

Per AGENTS.md: tests do NOT read source code files. They exercise behavior
through the public API and assert on observable outcomes (git status,
filesystem state, return values).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.dev_update import (
    WorktreeUpdateResult,
    _create_worktree,
    _is_dirty,
    _git_porcelain_status,
    gc_worktrees,
    list_worktrees,
    run_dev_update,
    should_use_worktree_update,
)


# ---------------------------------------------------------------------------
# Fixture: create a git repo with a remote and commits
# ---------------------------------------------------------------------------

def _git(*args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def _make_repo(path: Path) -> Path:
    """Initialize a git repo at *path* with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    (path / "README.md").write_text("# Test Repo\n")
    (path / ".gitignore").write_text(".worktrees/\n")
    _git("add", ".", cwd=path)
    _git("commit", "-qm", "initial commit", cwd=path)
    return path


def _make_remote(remote_path: Path, checkout_path: Path) -> None:
    """Create a bare remote from *checkout_path* and add it as origin."""
    _git("init", "--bare", "-q", "-b", "main", str(remote_path), cwd=checkout_path)
    _git("remote", "add", "origin", str(remote_path), cwd=checkout_path)
    _git("push", "-q", "-u", "origin", "main", cwd=checkout_path)


def _add_commit(path: Path, filename: str, content: str, msg: str = "update") -> None:
    """Add a commit to the repo at *path*."""
    (path / filename).write_text(content)
    _git("add", filename, cwd=path)
    _git("commit", "-qm", msg, cwd=path)


def _make_upstream(remote_path: Path, dest: Path) -> Path:
    """Clone the remote into *dest* and configure it for pushing."""
    _git("clone", "-q", "-b", "main", str(remote_path), str(dest), cwd=Path.cwd())
    _git("config", "user.email", "test@example.com", cwd=dest)
    _git("config", "user.name", "Test", cwd=dest)
    return dest


def _fetch_origin(checkout_path: Path, branch: str = "main") -> None:
    """Fetch origin/<branch> into the checkout."""
    _git("fetch", "origin", branch, cwd=checkout_path)


@pytest.fixture
def clean_repo(tmp_path):
    """A clean git checkout with a remote that has one new commit ahead."""
    checkout = _make_repo(tmp_path / "checkout")
    remote = tmp_path / "remote.git"
    _make_remote(remote, checkout)

    # Add a new commit on a clone of the remote (simulating upstream)
    upstream = tmp_path / "upstream"
    _make_upstream(remote, upstream)
    _add_commit(upstream, "feature.py", "print('new')\n", "add feature")
    _git("push", "-q", "origin", "main", cwd=upstream)

    # Fetch in the checkout so origin/main is available
    _fetch_origin(checkout)

    return checkout


@pytest.fixture
def dirty_repo(tmp_path):
    """A dirty git checkout (uncommitted changes) with a remote ahead."""
    checkout = _make_repo(tmp_path / "checkout")
    remote = tmp_path / "remote.git"
    _make_remote(remote, checkout)

    # Add a new commit upstream
    upstream = tmp_path / "upstream"
    _make_upstream(remote, upstream)
    _add_commit(upstream, "feature.py", "print('new')\n", "add feature")
    _git("push", "-q", "origin", "main", cwd=upstream)

    # Fetch in the checkout
    _fetch_origin(checkout)

    # Dirty the checkout
    (checkout / "local_change.py").write_text("# my local change\n")
    _git("add", "local_change.py", cwd=checkout)
    # Leave it staged but uncommitted (dirty)

    return checkout


@pytest.fixture
def dirty_repo_unstaged(tmp_path):
    """A dirty git checkout with unstaged changes."""
    checkout = _make_repo(tmp_path / "checkout")
    remote = tmp_path / "remote.git"
    _make_remote(remote, checkout)

    upstream = tmp_path / "upstream"
    _make_upstream(remote, upstream)
    _add_commit(upstream, "feature.py", "print('new')\n", "add feature")
    _git("push", "-q", "origin", "main", cwd=upstream)

    _fetch_origin(checkout)

    # Dirty the checkout with an unstaged change
    (checkout / "README.md").write_text("# Modified\n")
    # Don't stage it

    return checkout


# ---------------------------------------------------------------------------
# Tests: should_use_worktree_update
# ---------------------------------------------------------------------------

class TestShouldUseWorktreeUpdate:
    """Tests for the routing decision: should we use the worktree path?"""

    def test_returns_false_for_in_place_flag(self, clean_repo):
        """--in-place forces the legacy flow."""
        assert should_use_worktree_update(clean_repo, in_place=True) is False

    def test_returns_true_for_checkout_with_git(self, clean_repo):
        """A checkout (has .git) with git worktree available should route here."""
        assert should_use_worktree_update(clean_repo, in_place=False) is True

    def test_returns_false_for_slot(self, tmp_path):
        """A slot (has manifest.json) should not use worktree update."""
        slot = tmp_path / "slot"
        slot.mkdir()
        (slot / "manifest.json").write_text('{"version": "1.0"}')
        assert should_use_worktree_update(slot, in_place=False) is False


# ---------------------------------------------------------------------------
# Tests: clean tree fast-forward
# ---------------------------------------------------------------------------

class TestCleanTreeFastForward:
    """Clean tree, target newer → fast-forward in place (no worktree)."""

    def test_clean_tree_fast_forwards(self, clean_repo):
        """A clean tree should fast-forward without creating a worktree."""
        result = run_dev_update(
            clean_repo,
            "main",
            in_place=False,
            choose=None,
            input_fn=lambda prompt, default: "1",
        )

        assert result.success is True
        assert result.fast_forwarded is True
        assert result.worktree_path is None
        # No worktree should have been created
        assert not (clean_repo / ".worktrees").exists()

    def test_clean_tree_no_choice_prompted(self, clean_repo):
        """Even with choose=None, clean tree doesn't prompt (fast-forwards)."""
        # Track if prompt was called
        prompt_calls = []

        def fake_input(prompt, default):
            prompt_calls.append(prompt)
            return "1"

        result = run_dev_update(
            clean_repo,
            "main",
            in_place=False,
            choose=None,
            input_fn=fake_input,
        )

        assert result.success is True
        assert result.fast_forwarded is True
        assert len(prompt_calls) == 0  # no prompt for clean tree


# ---------------------------------------------------------------------------
# Tests: dirty tree → 3-option choice
# ---------------------------------------------------------------------------

class TestDirtyTreeChoice:
    """Dirty tree → returns the 3-option choice."""

    def test_dirty_tree_returns_choice_when_cancel(self, dirty_repo):
        """Dirty tree with choose='cancel' returns cancelled."""
        result = run_dev_update(
            dirty_repo,
            "main",
            in_place=False,
            choose="cancel",
        )

        assert result.success is False
        assert result.choice is not None
        assert result.choice.action == "cancel"
        # Original tree status should be byte-identical
        assert _is_dirty(dirty_repo) is True

    def test_dirty_tree_prompt_returns_choice(self, dirty_repo):
        """Dirty tree with choose=None prompts and returns the user's choice."""
        result = run_dev_update(
            dirty_repo,
            "main",
            in_place=False,
            choose=None,
            input_fn=lambda prompt, default: "3",  # cancel
        )

        assert result.success is False
        assert result.choice is not None
        assert result.choice.action == "cancel"


# ---------------------------------------------------------------------------
# Tests: choose="switch" creates worktree
# ---------------------------------------------------------------------------

class TestSwitchCreatesWorktree:
    """choose='switch' creates worktree, original tree byte-identical."""

    def test_switch_creates_worktree(self, dirty_repo):
        """Switching creates .worktrees/<target> and provisions it."""
        provision_calls = []

        def fake_provision(wt_path: Path):
            provision_calls.append(wt_path)
            # Create a bin/hermes stub so symlink target exists
            bin_dir = wt_path / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "hermes").write_text("#!/bin/sh\necho hermes\n")
            (bin_dir / "hermes").chmod(0o755)

        symlink_calls = []

        def fake_symlink(symlink_path: Path, target: Path):
            symlink_calls.append((symlink_path, target))

        # Capture status before
        status_before = _git_porcelain_status(dirty_repo)

        result = run_dev_update(
            dirty_repo,
            "main",
            in_place=False,
            choose="switch",
            dev_sync_fn=fake_provision,
            symlink_fn=fake_symlink,
        )

        assert result.success is True
        assert result.choice is not None
        assert result.choice.action == "switch"
        assert result.worktree_path is not None
        assert result.worktree_path.exists()

        # Worktree was created under .worktrees/
        wt_dir = dirty_repo / ".worktrees"
        assert wt_dir.exists()
        created = [d for d in wt_dir.iterdir() if d.is_dir()]
        assert len(created) >= 1

        # Provisioning was called
        assert len(provision_calls) == 1

        # Symlink was repointed
        assert len(symlink_calls) == 1

        # Original tree's git status is byte-identical
        status_after = _git_porcelain_status(dirty_repo)
        assert status_before == status_after, (
            "Original tree's git status must be byte-identical before/after"
        )

    def test_switch_original_tree_unchanged_unstaged(self, dirty_repo_unstaged):
        """Switch with unstaged changes also preserves byte-identical status."""
        def fake_provision(wt_path: Path):
            bin_dir = wt_path / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "hermes").write_text("#!/bin/sh\necho hermes\n")
            (bin_dir / "hermes").chmod(0o755)

        status_before = _git_porcelain_status(dirty_repo_unstaged)

        result = run_dev_update(
            dirty_repo_unstaged,
            "main",
            in_place=False,
            choose="switch",
            dev_sync_fn=fake_provision,
            symlink_fn=lambda s, t: None,  # mock symlink
        )

        assert result.success is True
        status_after = _git_porcelain_status(dirty_repo_unstaged)
        assert status_before == status_after


# ---------------------------------------------------------------------------
# Tests: choose="merge" → fetch+merge, stop on conflict
# ---------------------------------------------------------------------------

class TestMergeInPlace:
    """choose='merge' → fetch + merge in place, stop on conflict."""

    def test_merge_succeeds_when_no_conflict(self, dirty_repo):
        """Merge succeeds when there are no conflicts."""
        result = run_dev_update(
            dirty_repo,
            "main",
            in_place=False,
            choose="merge",
        )

        assert result.choice is not None
        assert result.choice.action == "merge"
        assert result.choice.merge_exit_code == 0
        assert result.success is True

    def test_merge_stops_on_conflict(self, tmp_path):
        """Merge stops with non-zero exit code when there's a conflict."""
        checkout = _make_repo(tmp_path / "checkout")
        remote = tmp_path / "remote.git"
        _make_remote(remote, checkout)

        # Upstream makes a conflicting change
        upstream = tmp_path / "upstream"
        _make_upstream(remote, upstream)
        (upstream / "README.md").write_text("# Upstream Changed\n")
        _git("add", "README.md", cwd=upstream)
        _git("commit", "-qm", "upstream change", cwd=upstream)
        _git("push", "-q", "origin", "main", cwd=upstream)

        # Fetch in checkout
        _fetch_origin(checkout)

        # Make a conflicting LOCAL commit (so the tree diverges)
        (checkout / "README.md").write_text("# Local Changed\n")
        _git("add", "README.md", cwd=checkout)
        _git("commit", "-qm", "local change", cwd=checkout)

        # Dirty the tree with an uncommitted change so it's not "clean"
        # (run_dev_update checks _is_dirty first)
        (checkout / "uncommitted.py").write_text("# dirty\n")

        result = run_dev_update(
            checkout,
            "main",
            in_place=False,
            choose="merge",
        )

        assert result.choice is not None
        assert result.choice.action == "merge"
        # Merge should fail (conflict)
        assert result.choice.merge_exit_code != 0
        assert result.success is False
        # Verify conflict markers exist
        readme = (checkout / "README.md").read_text()
        assert "<<<<<<" in readme or ">>>>" in readme


# ---------------------------------------------------------------------------
# Tests: worktree creation failure → fallback
# ---------------------------------------------------------------------------

class TestWorktreeCreationFailure:
    """Worktree creation failure → falls back to legacy autostash."""

    def test_falls_back_on_worktree_failure(self, dirty_repo, monkeypatch):
        """When git worktree add fails, falls back to legacy autostash."""
        # Monkeypatch _create_worktree to raise RuntimeError
        from hermes_cli import dev_update

        def fake_create(tree_root, target_name, target_ref):
            raise RuntimeError("simulated worktree creation failure")

        monkeypatch.setattr(dev_update, "_create_worktree", fake_create)

        result = run_dev_update(
            dirty_repo,
            "main",
            in_place=False,
            choose="switch",
            dev_sync_fn=lambda p: None,
            symlink_fn=lambda s, t: None,
        )

        assert result.fell_back is True
        assert result.success is False
        assert len(result.errors) > 0
        assert "worktree" in result.errors[0].lower()


# ---------------------------------------------------------------------------
# Tests: gc — list/remove old version-worktrees
# ---------------------------------------------------------------------------

class TestGcWorktrees:
    """hermes dev gc — list/remove old version-worktrees."""

    def test_list_worktrees_empty(self, tmp_path):
        """No .worktrees dir → empty list."""
        checkout = _make_repo(tmp_path / "checkout")
        assert list_worktrees(checkout) == []

    def test_list_worktrees_finds_created(self, dirty_repo):
        """After creating a worktree, list_worktrees finds it."""
        wt_path = _create_worktree(
            dirty_repo, "v1.0.0", "origin/main"
        )
        wts = list_worktrees(dirty_repo)
        assert len(wts) == 1
        assert wts[0].resolve() == wt_path.resolve()

    def test_gc_removes_old_worktrees(self, dirty_repo):
        """gc removes old worktrees, keeping the most recent N."""
        # Create 3 worktrees
        for i in range(3):
            # Each needs a unique ref — use commits
            _add_commit(dirty_repo, f"file{i}.py", f"# {i}\n", f"commit {i}")
            _git("fetch", "origin", "main", cwd=dirty_repo)

        # Create worktrees at different commits
        commits = _git("log", "--format=%H", cwd=dirty_repo).stdout.strip().split("\n")
        for i, commit in enumerate(commits[:3]):
            _create_worktree(dirty_repo, f"v1.{i}", commit)

        wts = list_worktrees(dirty_repo)
        assert len(wts) == 3

        # GC with keep=1 (should remove 2, keep 1)
        removed = gc_worktrees(dirty_repo, keep_n=1)
        assert len(removed) == 2

        wts_after = list_worktrees(dirty_repo)
        assert len(wts_after) == 1

    def test_gc_never_removes_active_symlink_target(self, dirty_repo, monkeypatch):
        """gc never removes the worktree the PATH symlink points at."""
        wt_path = _create_worktree(dirty_repo, "v1.0.0", "origin/main")

        # Mock the symlink to point at this worktree
        from hermes_cli import dev_update

        def fake_resolve():
            return wt_path.resolve()

        monkeypatch.setattr(
            dev_update,
            "_resolve_active_symlink_target",
            fake_resolve,
        )

        # Create a second worktree
        wt_path2 = _create_worktree(dirty_repo, "v2.0.0", "origin/main")

        # GC with keep=1 — should remove v2 but NOT v1 (the active one)
        removed = gc_worktrees(dirty_repo, keep_n=1)
        removed_names = [p.name for p in removed]

        # The active one must not be in removed
        assert "v1.0.0" not in removed_names

        # The active one must still exist
        assert wt_path.exists()

    def test_gc_dry_run(self, dirty_repo):
        """Dry run lists but doesn't remove."""
        _create_worktree(dirty_repo, "v1.0.0", "origin/main")
        _create_worktree(dirty_repo, "v2.0.0", "origin/main")

        removed = gc_worktrees(dirty_repo, keep_n=1, dry_run=True)
        assert len(removed) == 1

        # Nothing actually removed
        wts = list_worktrees(dirty_repo)
        assert len(wts) == 2


# ---------------------------------------------------------------------------
# Tests: _is_dirty and _git_porcelain_status
# ---------------------------------------------------------------------------

class TestGitStatusHelpers:
    """Tests for git status helper functions."""

    def test_is_dirty_returns_false_for_clean(self, clean_repo):
        assert _is_dirty(clean_repo) is False

    def test_is_dirty_returns_true_for_staged(self, dirty_repo):
        assert _is_dirty(dirty_repo) is True

    def test_is_dirty_returns_true_for_unstaged(self, dirty_repo_unstaged):
        assert _is_dirty(dirty_repo_unstaged) is True

    def test_porcelain_status_returns_raw_output(self, dirty_repo):
        status = _git_porcelain_status(dirty_repo)
        assert "local_change.py" in status

    def test_porcelain_status_empty_for_clean(self, clean_repo):
        status = _git_porcelain_status(clean_repo)
        assert status.strip() == ""
