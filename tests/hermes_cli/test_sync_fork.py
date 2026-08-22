"""Tests for ``hermes sync-fork`` — merging upstream into a *diverged* fork.

``hermes update`` deliberately refuses this case. ``_sync_upstream`` compares
``origin/main`` with ``upstream/main`` and returns early when the fork carries
commits upstream does not have::

    if origin_ahead > 0:
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        return

The refusal is correct — it uses ``pull --ff-only``, which cannot express a
merge — but it leaves every fork that has ever committed with no supported
update path. The gap is silent and compounds: a fork can sit hundreds of
commits behind while ``hermes update`` reports success, because the fast-forward
sync it skipped is the only upstream check it performs.

``sync-fork`` fills that gap: it merges rather than fast-forwards, and it
merges *incrementally* so a large backlog fails in small, reviewable pieces
instead of one unresolvable conflict.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import sync_fork


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commit(repo: Path, path: str, body: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    _git(repo, "add", path)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def forked(tmp_path: Path) -> Path:
    """A fork that has diverged: shared base, then commits on both sides."""
    up = tmp_path / "upstream"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "shared.py", "BASE = 1\n", "base")
    _commit(up, "upstream_only.py", "UP = 1\n", "upstream feature")

    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    # Diverge: fork gains a commit upstream does not have.
    _commit(fork, "local_only.py", "LOCAL = 1\n", "local feature")
    # Upstream moves on afterwards.
    _commit(up, "upstream_later.py", "LATER = 1\n", "upstream later")
    _git(fork, "fetch", "-q", "upstream", "main")
    return fork


# ---------------------------------------------------------------------------
# divergence detection — the state `hermes update` refuses to act on
# ---------------------------------------------------------------------------

def test_detects_divergence(forked: Path) -> None:
    state = sync_fork.inspect(forked, upstream_ref="upstream/main")
    assert state.diverged is True
    assert state.behind == 1
    assert state.ahead == 1


def test_clean_fork_is_not_diverged(tmp_path: Path) -> None:
    up = tmp_path / "u"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "a.py", "A = 1\n", "a")
    fork = tmp_path / "f"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    _git(fork, "fetch", "-q", "upstream", "main")
    state = sync_fork.inspect(fork, upstream_ref="upstream/main")
    assert state.diverged is False
    assert state.behind == 0


# ---------------------------------------------------------------------------
# the merge itself — what `pull --ff-only` cannot do
# ---------------------------------------------------------------------------

def test_merges_diverged_fork_keeping_both_sides(forked: Path) -> None:
    result = sync_fork.sync(forked, upstream_ref="upstream/main")
    assert result.ok is True
    assert (forked / "local_only.py").exists(), "local work must survive"
    assert (forked / "upstream_later.py").exists(), "upstream work must arrive"
    assert sync_fork.inspect(forked, upstream_ref="upstream/main").behind == 0


def test_refuses_to_run_on_a_dirty_worktree(forked: Path) -> None:
    (forked / "shared.py").write_text("BASE = 999\n")
    result = sync_fork.sync(forked, upstream_ref="upstream/main")
    assert result.ok is False
    assert "uncommitted" in result.reason.lower()
    # The dirty edit must still be there — refusing must not discard work.
    assert (forked / "shared.py").read_text() == "BASE = 999\n"


# ---------------------------------------------------------------------------
# conflict handling — must restore, never leave a half-merged tree
# ---------------------------------------------------------------------------

def test_conflict_aborts_and_restores_original_head(tmp_path: Path) -> None:
    up = tmp_path / "u"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "f.py", "VALUE = 1\n", "base")
    fork = tmp_path / "f"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    # Both sides edit the same line — a guaranteed conflict.
    _commit(fork, "f.py", "VALUE = 'fork'\n", "fork edit")
    _commit(up, "f.py", "VALUE = 'upstream'\n", "upstream edit")
    _git(fork, "fetch", "-q", "upstream", "main")

    before = _git(fork, "rev-parse", "HEAD")
    result = sync_fork.sync(fork, upstream_ref="upstream/main")

    assert result.ok is False
    assert result.conflicted_paths == ["f.py"]
    assert _git(fork, "rev-parse", "HEAD") == before, "HEAD must be restored"
    assert not (fork / ".git" / "MERGE_HEAD").exists(), "no half-merged state"
    assert _git(fork, "status", "--porcelain") == "", "worktree must be clean"


# ---------------------------------------------------------------------------
# incremental strategy — the part that makes a large backlog tractable
# ---------------------------------------------------------------------------

def test_plan_splits_at_commits_touching_conflict_prone_files(forked: Path) -> None:
    """A backlog is merged in steps, not one jump.

    Upstream commits that touch a file the fork has also modified are the ones
    that can conflict. Stopping at each of them means a conflict names one
    upstream commit instead of the whole range, which is what makes it
    reviewable.
    """
    plan = sync_fork.plan(forked, upstream_ref="upstream/main")
    assert plan.steps, "a behind fork must produce at least one step"
    assert plan.steps[-1] == "upstream/main", "the plan must finish at the tip"


def test_plan_is_empty_when_already_current(tmp_path: Path) -> None:
    up = tmp_path / "u"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "a.py", "A = 1\n", "a")
    fork = tmp_path / "f"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    _git(fork, "fetch", "-q", "upstream", "main")
    assert sync_fork.plan(fork, upstream_ref="upstream/main").steps == []


def test_dry_run_does_not_move_head(forked: Path) -> None:
    before = _git(forked, "rev-parse", "HEAD")
    result = sync_fork.sync(forked, upstream_ref="upstream/main", dry_run=True)
    assert result.ok is True
    assert _git(forked, "rev-parse", "HEAD") == before
    assert sync_fork.inspect(forked, upstream_ref="upstream/main").behind == 1


# ---------------------------------------------------------------------------
# committer identity — a merge writes a commit
# ---------------------------------------------------------------------------

def test_syncs_on_a_host_with_no_git_identity(tmp_path: Path, monkeypatch) -> None:
    """A merge commit needs a committer, and plenty of hosts have none.

    Fresh containers, service accounts and CI runners have no global git
    identity. Without one, ``git merge`` exits non-zero with "Committer identity
    unknown" and *zero* conflicted paths, so reporting it as a conflict sends
    the user looking for a merge conflict that does not exist. Caught by the
    Docker e2e run, which is why this test isolates git's config lookup instead
    of trusting the developer machine's global config.
    """
    home = tmp_path / "empty-home"
    home.mkdir()
    # Point every git config lookup at an empty home so no identity is visible.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "nonexistent-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(home / "nonexistent-system"))
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)

    up = tmp_path / "u"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "shared.py", "V = 1\n", "base")
    fork = tmp_path / "f"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    _commit(fork, "local.py", "L = 1\n", "local")
    _commit(up, "later.py", "X = 1\n", "upstream later")
    _git(fork, "fetch", "-q", "upstream", "main")

    result = sync_fork.sync(fork, upstream_ref="upstream/main")

    assert result.ok is True, result.reason
    assert (fork / "local.py").exists()
    assert (fork / "later.py").exists()


def test_non_conflict_failure_is_not_reported_as_a_conflict(tmp_path: Path) -> None:
    """A merge can fail for reasons other than conflicting content.

    Unrelated histories, a bad ref, a rejecting hook — all exit non-zero with no
    unmerged paths. Those must surface git's own message, because calling them
    "conflict" points the reader at the wrong problem.
    """
    a = tmp_path / "a"
    a.mkdir()
    _git(a, "init", "-q", "-b", "main")
    _commit(a, "a.py", "A = 1\n", "a")
    # A second repo with no shared history at all.
    b = tmp_path / "b"
    b.mkdir()
    _git(b, "init", "-q", "-b", "main")
    _commit(b, "b.py", "B = 1\n", "b")
    _git(a, "remote", "add", "upstream", str(b))
    _git(a, "fetch", "-q", "upstream", "main")

    before = _git(a, "rev-parse", "HEAD")
    result = sync_fork.sync(a, upstream_ref="upstream/main")

    assert result.ok is False
    assert result.conflicted_paths == []
    assert "conflict" not in result.reason.lower()
    assert _git(a, "rev-parse", "HEAD") == before
    assert _git(a, "status", "--porcelain") == ""
# ---------------------------------------------------------------------------
# an unreadable position is not a healthy one
# ---------------------------------------------------------------------------


def _fork_with_unfetched_upstream(tmp_path: Path) -> Path:
    """A fork whose ``upstream`` remote exists but has never been fetched.

    This is not a contrived case: it is the state of every fresh clone where
    someone ran ``git remote add upstream ...`` and then went straight to
    ``hermes sync-fork``. The ``forked`` fixture above deliberately fetches, which
    is why the rest of this suite never exercised it.
    """
    up = tmp_path / "upstream"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "shared.py", "BASE = 1\n", "base")
    _commit(up, "upstream_only.py", "UP = 1\n", "upstream feature")

    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", "-q", str(up), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(up))
    # Upstream moves on, so there IS something to be behind by.
    _commit(up, "upstream_later.py", "LATER = 1\n", "upstream later")
    # No fetch. upstream/main does not exist in the fork.
    return fork


def test_an_unfetched_upstream_is_an_error_not_up_to_date(tmp_path: Path) -> None:
    """The state must say it could not be read, rather than reading as current.

    ``git rev-list --count HEAD..upstream/main`` fails when the ref is absent;
    that failure became an empty string and then the integer 0, so the fork was
    certified up to date while being two commits behind. A silent no-op forever
    is exactly what this command exists to prevent one level up.
    """
    fork = _fork_with_unfetched_upstream(tmp_path)
    state = sync_fork.inspect(fork)

    assert state.error, "an unresolvable upstream ref reported no error"
    assert "upstream/main" in state.error
    # And the numbers must not be mistakable for a healthy reading.
    assert state.behind == 0 and state.diverged is False
    # Proof the fork really was behind, i.e. the 0 above was a lie without the guard.
    _git(fork, "fetch", "-q", "upstream", "main")
    assert sync_fork.inspect(fork).behind > 0


def test_a_garbage_upstream_ref_is_an_error(forked: Path) -> None:
    """A mistyped --upstream-ref is the same class of fault, not a clean repo."""
    state = sync_fork.inspect(forked, "@@not-a-ref@@")
    assert state.error and "@@not-a-ref@@" in state.error


def test_a_resolvable_ref_still_reports_no_error(forked: Path) -> None:
    """The guard must not fire on the happy path."""
    state = sync_fork.inspect(forked)
    assert state.error is None
    assert state.diverged is True
