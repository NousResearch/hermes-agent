"""Regression: the orphan-branch pruner keys on the owning session, not a clock.

``_prune_orphaned_branches`` deletes local ``hermes/hermes-*`` and ``pr-*`` branches
that have no worktree. A branch is worktree-less for a while before its work is
finished — the ref is created before its tree, and a tree can be reaped on an earlier
pass — so reaping on sight destroys in-flight work and takes the only durable record of
it.

The rule under test: a branch carrying a recorded owner is reaped when that session is
gone from the session store, whatever its age; a branch with no recorded owner falls
back to the age gate. Reaping is destructive, so every unknown fails toward "keep".
"""

import os
import subprocess
import time

import pytest

from hermes_cli import worktree_ops

STALE_DATE = "2026-01-01T00:00:00+00:00"


def _git(repo, *args, env=None):
    merged = dict(os.environ)
    merged.update(env or {})
    subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, env=merged)


def _branches(repo):
    out = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    return {b.strip() for b in out.splitlines() if b.strip()}


@pytest.fixture
def repo(tmp_path):
    """A branch cut now from a days-old commit, and one untouched for days.

    The fresh branch's tip commit is old: only its ref update is new, as for a lane
    branched off a quiet repo's main or a ``pr-*`` head fetched from an old commit.
    """
    old = {"GIT_AUTHOR_DATE": STALE_DATE, "GIT_COMMITTER_DATE": STALE_DATE}
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@test.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "commit", "-q", "--allow-empty", "-m", "base", env=old)
    _git(root, "branch", "hermes/hermes-fresh")
    _git(root, "checkout", "-q", "-b", "hermes/hermes-stale", "main")
    _git(root, "commit", "-q", "--allow-empty", "-m", "stale", env=old)
    _git(root, "checkout", "-q", "main")
    stale_log = root / ".git" / "logs" / "refs" / "heads" / "hermes" / "hermes-stale"
    days_ago = time.time() - 5 * 86400
    os.utime(stale_log, (days_ago, days_ago))
    return root


@pytest.fixture
def live():
    """The real session store under the hermetic HERMES_HOME: ``live.add(sid)`` records a
    live session, ``live.archive(sid)`` records one and archives it."""
    from hermes_state import SessionDB

    db = SessionDB()

    class Store:
        def add(self, sid):
            db.create_session(sid, source="cli")

        def archive(self, sid):
            self.add(sid)
            db.set_session_archived(sid, True)

    yield Store()
    db.close()


def _own(repo, branch, session_id, monkeypatch):
    """Record *session_id* as the owner of *branch*, as creation would."""
    monkeypatch.setenv("HERMES_SESSION_ID", session_id)
    worktree_ops._record_branch_owner(str(repo), branch)


# ── the association itself ────────────────────────────────────────────


def test_the_creator_records_itself_on_the_branch(repo, monkeypatch):
    _own(repo, "hermes/hermes-fresh", "sess-created", monkeypatch)

    assert worktree_ops._branch_owner_sessions(str(repo)) == {
        "hermes/hermes-fresh": "sess-created"
    }


def test_no_session_in_the_environment_records_no_owner(repo, monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    worktree_ops._record_branch_owner(str(repo), "hermes/hermes-fresh")

    assert worktree_ops._branch_owner_sessions(str(repo)) == {}


# ── session-first: the branch lives as long as its session ────────────


def test_a_branch_whose_session_is_gone_is_reaped_however_young(repo, live, monkeypatch):
    """The in-flight branch the age gate would spare, once its session has gone."""
    _own(repo, "hermes/hermes-fresh", "sess-done", monkeypatch)
    live.archive("sess-done")

    worktree_ops._prune_orphaned_branches(str(repo))

    assert "hermes/hermes-fresh" not in _branches(repo)


def test_a_branch_whose_session_is_live_is_kept_however_old(repo, live, monkeypatch):
    """The stale branch the age gate would reap, while its session is still live."""
    _own(repo, "hermes/hermes-stale", "sess-live", monkeypatch)
    live.add("sess-live")

    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=0)

    assert "hermes/hermes-stale" in _branches(repo)


def test_an_owner_the_store_has_no_row_for_falls_back_to_the_age_gate(repo, live, monkeypatch):
    """No row is not "gone": the row is written on the first turn and the store is per
    profile, so a fresh lane (or another profile's) must not be reaped on sight."""
    _own(repo, "hermes/hermes-fresh", "sess-before-first-turn", monkeypatch)
    _own(repo, "hermes/hermes-stale", "sess-other-profile", monkeypatch)

    worktree_ops._prune_orphaned_branches(str(repo))

    assert "hermes/hermes-fresh" in _branches(repo)
    assert "hermes/hermes-stale" not in _branches(repo)


# ── the age gate, as the fallback for branches nothing owns ───────────


def test_a_branch_younger_than_the_cutoff_is_kept(repo, live):
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=24)

    assert "hermes/hermes-fresh" in _branches(repo)


def test_a_branch_older_than_the_cutoff_is_reaped(repo, live):
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=24)

    assert "hermes/hermes-stale" not in _branches(repo)


def test_the_default_call_keeps_a_branch_created_seconds_ago(repo, live):
    """No arguments: the startup pass must not reap in-flight work."""
    worktree_ops._prune_orphaned_branches(str(repo))

    assert "hermes/hermes-fresh" in _branches(repo)


def test_the_cutoff_and_not_the_name_decides(repo, live):
    """With the cutoff set to now, the branch the name rule would also spare goes."""
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=0)

    assert _branches(repo) == {"main"}


def test_an_undatable_unowned_branch_is_kept(repo, live, monkeypatch):
    """The age gate must never fire on a branch it cannot date."""
    monkeypatch.setattr(worktree_ops, "_branch_tip_dates", lambda _root: {})

    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=0)

    assert {"hermes/hermes-fresh", "hermes/hermes-stale"} <= _branches(repo)
