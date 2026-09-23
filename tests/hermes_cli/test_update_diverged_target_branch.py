"""Regression: `hermes update` must not destroy local commits on the target branch.

Live incident (2026-08-31): the source checkout sat on ``main`` carrying two
local-only commits (a Grok prompt-guidance feature and an Anthropic auxiliary
auth fix). ``hermes update`` found the history diverged, took the
"true upstream force-push" path and ran ``git reset --hard origin/main``,
silently destroying both commits. Only the reflog made recovery possible.

The parked-branch path already got this right: on a CUSTOM branch the updater
merges origin/<target> instead of resetting, "so local commits survive". The
gap was that the guard only ran when ``current_branch != branch`` — the exact
same work sitting on the target branch itself was unprotected.

``_plan_diverged_target_update`` closes that gap. It answers one question
against real git state: does HEAD carry commits that are not contained in
origin/<branch>?

- yes -> ("merge", n) : local work exists; merging preserves it, and a
  conflict can stop the update cleanly with nothing lost.
- no  -> ("reset", 0) : nothing local to lose, so the original force-push
  recovery behaviour (hard reset onto the remote) still applies.

These tests run against REAL git repositories rather than mocked
``subprocess.run``, so they exercise the actual ``git cherry`` semantics the
planner depends on.
"""

import subprocess

import pytest

from hermes_cli import update_cmd


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(
        GIT + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def _init_origin(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")
    return origin


def _clone(tmp_path, origin):
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    return clone


@pytest.fixture()
def diverged_with_local_commits(tmp_path):
    """Clone on main with local commits; origin/main moved on independently.

    This is the 2026-08-31 shape: the user's own work committed on the target
    branch, upstream advanced, ff-only impossible.
    """
    origin = _init_origin(tmp_path)
    clone = _clone(tmp_path, origin)

    # Local work committed straight onto main.
    (clone / "local_feature.txt").write_text("grok guidance\n")
    _git(clone, "add", "local_feature.txt")
    _git(clone, "commit", "-qm", "local: feature the user cares about")

    # Upstream advances with unrelated work.
    (origin / "b.txt").write_text("upstream\n")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "c2")

    _git(clone, "fetch", "-q", "origin", "main")
    return clone


@pytest.fixture()
def diverged_force_push_only(tmp_path):
    """Clone on main with NO local commits; origin/main was rewritten.

    The genuine force-push case the reset path was written for.
    """
    origin = _init_origin(tmp_path)
    clone = _clone(tmp_path, origin)

    # Upstream rewrites history (amend = new sha, same lineage position).
    (origin / "a.txt").write_text("rewritten\n")
    _git(origin, "commit", "-aq", "--amend", "-m", "c1 rewritten")

    _git(clone, "fetch", "-q", "origin", "main")
    return clone


def test_local_commits_on_target_branch_plan_a_merge(diverged_with_local_commits):
    """Local-only commits on the target branch must never be hard-reset away."""
    action, count = update_cmd._plan_diverged_target_update(
        GIT, diverged_with_local_commits, "main"
    )

    assert action == "merge"
    assert count == 1


def test_force_push_without_local_commits_still_resets(diverged_force_push_only):
    """No local work to lose → keep the original force-push recovery."""
    action, count = update_cmd._plan_diverged_target_update(
        GIT, diverged_force_push_only, "main"
    )

    assert action == "reset"
    assert count == 0


def test_commit_already_upstream_by_patch_does_not_force_a_merge(tmp_path):
    """A commit whose patch already landed upstream is not local work.

    ``git cherry`` marks it '-'; resetting loses nothing, so the planner must
    not treat a cherry-picked-upstream commit as a reason to merge.
    """
    origin = _init_origin(tmp_path)
    clone = _clone(tmp_path, origin)

    (clone / "shared.txt").write_text("same change\n")
    _git(clone, "add", "shared.txt")
    _git(clone, "commit", "-qm", "shared change")

    # The identical patch lands upstream independently.
    (origin / "shared.txt").write_text("same change\n")
    _git(origin, "add", "shared.txt")
    _git(origin, "commit", "-qm", "shared change")

    _git(clone, "fetch", "-q", "origin", "main")

    action, count = update_cmd._plan_diverged_target_update(GIT, clone, "main")

    assert action == "reset"
    assert count == 0


def test_unverifiable_git_state_prefers_merge(tmp_path):
    """If the local/remote relationship cannot be read, do not destroy work.

    An unreadable comparison is exactly when a hard reset is least defensible,
    so the planner fails safe toward the non-destructive path.
    """
    origin = _init_origin(tmp_path)
    clone = _clone(tmp_path, origin)

    action, count = update_cmd._plan_diverged_target_update(
        GIT, clone, "branch-that-does-not-exist"
    )

    assert action == "merge"
    assert count == 0
