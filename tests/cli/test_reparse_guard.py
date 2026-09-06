"""Regression tests for the 2026-09-06 data loss: a junction inside a worktree.

The scenario, exactly as it happened in production:

    repo/app/node_modules/@real3d/stage  --junction-->  victim_repo/
    repo/.wt/w1/app/node_modules         --junction-->  repo/app/node_modules

``git worktree remove repo/.wt/w1`` walked junction -> junction -> victim and
deleted the victim's contents. Two source repositories were emptied.

The Windows-specific tests skip elsewhere; the ``unknown``-state and
ownership tests are platform-independent and always run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.reparse_guard import (  # noqa: E402
    UnknownEdge,
    is_reparse_point,
    remove_tree,
)

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="junction semantics are Windows-only"
)


def _mklink_junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot create junction here: {result.stderr or result.stdout}")


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture()
def scenario(tmp_path: Path):
    """The production shape: worktree -> primary node_modules -> victim repo."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("PRECIOUS SOURCE")

    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    _git("init", "-q", ".", cwd=repo)
    _git("config", "user.email", "t@t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    (repo / "app" / "f.txt").write_text("x")
    (repo / ".gitignore").write_text("node_modules/\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)

    nm = repo / "app" / "node_modules" / "@real3d"
    nm.mkdir(parents=True)
    _mklink_junction(nm / "stage", victim)

    wt = repo / ".wt" / "w1"
    _git("worktree", "add", str(wt), "-b", "wt/w1", "-q", cwd=repo)
    _mklink_junction(wt / "app" / "node_modules", repo / "app" / "node_modules")

    return {"repo": repo, "wt": wt, "victim": victim}


# --------------------------------------------------------------------------
# The blind spot, and the damage
# --------------------------------------------------------------------------

@windows_only
def test_junction_is_not_seen_by_islink(scenario):
    """os.path.islink answers False for a junction — the whole blind spot.

    Every ordinary "don't follow links" guard walked straight in because of
    this. If it ever fails, CPython changed and the guard can be simplified.
    """
    junction = scenario["wt"] / "app" / "node_modules"
    assert os.path.isdir(junction)
    assert os.path.islink(junction) is False       # the trap
    assert is_reparse_point(junction) is True      # what actually detects it


@windows_only
def test_git_worktree_remove_destroys_through_a_junction(scenario):
    """Proves the damage is real, and that it is git's traversal doing it.

    This is why the fix cannot be "call git worktree remove more carefully".
    """
    victim_file = scenario["victim"] / "important.txt"
    assert victim_file.exists()

    _git("worktree", "remove", str(scenario["wt"]), "--force", cwd=scenario["repo"])

    assert not victim_file.exists(), (
        "expected git to delete through the junction; if this now passes, git "
        "fixed the behaviour and our own walker is belt-and-braces"
    )


# --------------------------------------------------------------------------
# The fix: our own walker owns the deletion
# --------------------------------------------------------------------------

@windows_only
def test_remove_tree_saves_the_victim_and_the_junction_target(scenario):
    victim_file = scenario["victim"] / "important.txt"

    remove_tree(scenario["wt"])

    assert not scenario["wt"].exists(), "the worktree itself must be gone"
    assert victim_file.exists(), "the victim repo must survive"
    assert victim_file.read_text() == "PRECIOUS SOURCE"
    # And the primary tree the junction pointed at is untouched.
    assert (scenario["repo"] / "app" / "node_modules" / "@real3d" / "stage").exists()


@windows_only
def test_remove_tree_on_a_junction_unlinks_it_rather_than_following_it(tmp_path):
    """Asking to delete a junction deletes the junction, never its contents."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    _mklink_junction(link, target)

    assert remove_tree(link) == 1
    assert not link.exists()
    assert (target / "keep.txt").read_text() == "keep"


def test_remove_tree_deletes_an_ordinary_nested_tree(tmp_path):
    """The boring case still has to work, on every platform."""
    root = tmp_path / "tree"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "deep.txt").write_text("x")
    (root / "top.txt").write_text("y")

    removed = remove_tree(root)

    assert not root.exists()
    assert removed == 5  # deep.txt, b, a, top.txt, root


# --------------------------------------------------------------------------
# `unknown` must refuse — the review's first gate
# --------------------------------------------------------------------------

def test_unclassifiable_entry_refuses_the_whole_removal(tmp_path, monkeypatch):
    """A metadata failure must abort, never be treated as an ordinary edge.

    This is the exact hiding place the guard exists to close: if lstat fails on
    the one entry that IS a junction and we shrug it off, we delete through it.
    """
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    victim = root / "sub" / "mystery"
    victim.write_text("?")

    real_lstat = os.lstat

    def flaky_lstat(path, *args, **kwargs):
        if str(path).endswith("mystery"):
            raise PermissionError(5, "Access is denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", flaky_lstat)

    with pytest.raises(UnknownEdge) as excinfo:
        remove_tree(root)

    assert "mystery" in str(excinfo.value.path)
    # Nothing beyond the refusal: the tree is still there to be inspected.
    assert root.exists()
    assert victim.exists()


def test_unreadable_directory_refuses_rather_than_skipping(tmp_path, monkeypatch):
    """A directory we cannot enumerate is unknown, not empty."""
    root = tmp_path / "tree"
    (root / "locked").mkdir(parents=True)

    real_scandir = os.scandir

    def flaky_scandir(path, *args, **kwargs):
        if str(path).endswith("locked"):
            raise PermissionError(5, "Access is denied")
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky_scandir)

    with pytest.raises(UnknownEdge):
        remove_tree(root)
    assert root.exists()


def test_is_reparse_point_raises_on_an_unclassifiable_path(tmp_path, monkeypatch):
    """The classifier itself must never answer False for "I could not tell"."""
    p = tmp_path / "thing"
    p.write_text("x")

    def boom(path, *args, **kwargs):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(os, "lstat", boom)
    with pytest.raises(UnknownEdge):
        is_reparse_point(p)


def test_missing_path_is_not_unknown(tmp_path):
    """Absent is a definite answer, not an unclassifiable one."""
    assert is_reparse_point(tmp_path / "nope") is False
    assert remove_tree(tmp_path / "nope") == 0


# --------------------------------------------------------------------------
# The generation gap — the review's second gate
# --------------------------------------------------------------------------

@windows_only
def test_a_junction_created_mid_walk_is_still_not_followed(tmp_path, monkeypatch):
    """A link that appears AFTER the walk starts must still not be traversed.

    The old shape (scan, then hand the tree to `git worktree remove`) could not
    survive this: the scan's verdict was already stale by the time the deleter
    ran. Because the walker classifies each edge in the same pass that deletes
    it, an entry created mid-walk is classified when it is reached.
    """
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("PRECIOUS")

    root = tmp_path / "tree"
    (root / "early").mkdir(parents=True)
    (root / "early" / "f.txt").write_text("x")
    later = root / "later"
    later.mkdir()

    real_scandir = os.scandir
    planted = {"done": False}

    def scandir_then_plant(path, *args, **kwargs):
        entries = list(real_scandir(path, *args, **kwargs))
        # Plant the junction after `root` has been enumerated once — i.e. after
        # any "pre-flight scan" would have declared the tree clean.
        if not planted["done"] and Path(str(path)) == root:
            planted["done"] = True
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(later / "nm"), str(victim)],
                capture_output=True, text=True,
            )
        return iter(entries)

    monkeypatch.setattr(os, "scandir", scandir_then_plant)

    remove_tree(root)

    assert planted["done"], "the test must actually have planted the junction"
    assert (victim / "precious.txt").exists(), (
        "a junction created mid-removal was traversed — the walker is not "
        "classifying edges in the same generation it deletes them"
    )


# --------------------------------------------------------------------------
# Ownership — the review's third gate
# --------------------------------------------------------------------------

def test_no_caller_invokes_git_worktree_remove_directly():
    """`git worktree remove` must have exactly one owner.

    Five call sites each ran their own removal, and every one of them could
    delete through a junction. If a new one appears, this fails and points at
    the module to use instead.
    """
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for path in list(repo.glob("hermes_cli/**/*.py")) + [repo / "cli.py"]:
        if path.name in {"worktree_removal.py"} or "test" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            if '"worktree", "remove"' in line or '"remove", str(' in line and "worktree" in line:
                offenders.append(f"{path.relative_to(repo)}:{lineno}")
    assert not offenders, (
        "these call git worktree remove directly instead of using "
        "hermes_cli.worktree_removal.remove_worktree: " + ", ".join(offenders)
    )
