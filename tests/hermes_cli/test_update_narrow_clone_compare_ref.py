"""Regression for #125112: the updater must resolve its compare ref on narrow clones.

A tag-pinned ``git clone --depth 1 --single-branch --branch <tag>`` configures
``remote.origin.fetch`` as a tag-only refspec. Fetching a branch by name then
writes only ``FETCH_HEAD`` — no ``refs/remotes/origin/<branch>`` tracking ref —
so the updater's ``rev-parse origin/<branch>`` failed with "Branch not found"
even though the remote has the branch. The fix fetches by explicit refspec, which
always writes the tracking ref.

Tests run the REAL fetch helpers against a real local ``file://`` origin and a
real narrow clone (git behavior is the thing under test; mocks would prove nothing).
"""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_check

git_cmd = ["git"]


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


@pytest.fixture
def narrow_clone(tmp_path: Path) -> Path:
    """Bare origin with ``main`` + one tag, and a tag-pinned --single-branch clone of it."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    _run(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _run(tmp_path, "clone", "-q", f"file://{origin}", str(seed))
    (seed / "f.txt").write_text("a\n", encoding="utf-8")
    _run(seed, "add", "f.txt")
    _run(seed, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    _run(seed, "tag", "v1")
    assert _run(seed, "push", "-q", "origin", "HEAD:refs/heads/main", "v1").returncode == 0
    work = tmp_path / "work"
    _run(tmp_path, "clone", "-q", "--depth", "1", "--single-branch", "--branch", "v1", f"file://{origin}", str(work))
    # Prove the clone is the narrow shape from the issue before testing anything.
    assert _run(work, "config", "--get-all", "remote.origin.fetch").stdout.strip() == "+refs/tags/v1:refs/tags/v1"
    return work


def test_forced_refspec_updates_shallow_clone_to_new_tip(narrow_clone: Path) -> None:
    """The ``+`` prefix is load-bearing on depth-1 shallow clones (the installer's shape).

    A shallow boundary makes the new tip a non-descendant of the old one, so a NON-forced
    refspec fetch is rejected as non-fast-forward and the tracking ref never advances —
    the update check would keep comparing against the stale tip forever. Only the forced
    form lands the new tip, so a refactor to the non-forced refspec fails here instead of
    silently breaking every shallow installer update check.
    """
    # The prior update check left a tracking ref at the tip the clone was pinned to:
    # seed it the same way (non-forced fetch while origin is still at the old tip).
    adv = narrow_clone.parent / "advance"
    assert _run(narrow_clone.parent, "clone", "-q", f"file://{narrow_clone.parent / 'origin.git'}", str(adv)).returncode == 0
    seed_fetch = _run(
        narrow_clone, "fetch", "--depth", "1", "origin",
        "refs/heads/main:refs/remotes/origin/main",
    )
    assert seed_fetch.returncode == 0, seed_fetch.stderr
    stale = _run(narrow_clone, "rev-parse", "--verify", "--quiet", "origin/main").stdout.strip()
    assert stale

    # Advance origin/main past the pinned tag.
    (adv / "f.txt").write_text("b\n", encoding="utf-8")
    _run(adv, "add", "f.txt")
    assert _run(adv, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "second").returncode == 0
    assert _run(adv, "push", "-q", "origin", "main").returncode == 0
    new_tip = _run(adv, "rev-parse", "HEAD").stdout.strip()
    assert new_tip and new_tip != stale

    # A non-forced refspec is rejected here as non-fast-forward; the production one lands.
    fetch_result, compare_branch = update_cmd_check.fetch_compare_branch(
        git_cmd, narrow_clone, "main", ["--depth", "1"],
    )
    assert fetch_result.returncode == 0, fetch_result.stderr
    assert compare_branch == "origin/main"
    landed = _run(narrow_clone, "rev-parse", "--verify", "--quiet", "origin/main").stdout.strip()
    assert landed == new_tip


def test_detached_narrow_clone_counts_from_the_pre_checkout_head(narrow_clone: Path, monkeypatch) -> None:
    """A narrow clone is detached with no local branch, so the updater runs ``checkout -B main
    origin/main``, which lands ON the target. Counting ``HEAD..origin/main`` afterwards read 0 and
    the update finished as "Already up to date!" with dependencies and migrations skipped."""
    from hermes_cli import update_cmd

    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", narrow_clone)
    monkeypatch.setattr("hermes_cli.source_check._github_compare_behind", lambda *a, **k: None)
    pinned = _run(narrow_clone, "rev-parse", "HEAD").stdout.strip()

    def plan():
        fetch_result, _ = update_cmd_check.fetch_compare_branch(git_cmd, narrow_clone, "main", ["--depth", "1"])
        assert fetch_result.returncode == 0, fetch_result.stderr
        return update_cmd._prepare_checkout_for_update(
            git_cmd, "main", update_cmd._current_branch_name(git_cmd, check=True), is_fork=False,
            assume_yes=True, gateway_mode=False, gw_input_fn=None, switch_branch=False,
            _windows_gateway_resume=None)

    # Pinned tag == origin/main: genuinely up to date.
    assert plan().commit_count == 0

    # Back to the narrow shape, then advance origin/main past the pinned tag.
    assert _run(narrow_clone, "checkout", "-q", "--detach", "v1").returncode == 0
    assert _run(narrow_clone, "branch", "-q", "-D", "main").returncode == 0
    adv = narrow_clone.parent / "advance"
    assert _run(narrow_clone.parent, "clone", "-q", f"file://{narrow_clone.parent / 'origin.git'}", str(adv)).returncode == 0
    assert _run(adv, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "b").returncode == 0
    assert _run(adv, "push", "-q", "origin", "main").returncode == 0

    behind = plan()
    assert behind.commit_count != 0  # -1 here: shallow, and the compare API is stubbed out
    # _pull_updates' did-HEAD-move guard compares against this, not the already-moved HEAD.
    assert behind.pre_sync_sha == pinned
