"""Repair of shallow boundaries dropped by the stale-graft prune (#108286).

``prune_stale_shallow_grafts()`` dropped ``.git/shallow`` grafts that reflog-only
commits still needed, leaving shallow installer checkouts with commits whose parent
objects were never downloaded — ``git gc`` / ``fsck`` / ``fetch`` all fail, and the
corruption cannot self-heal because reflog expiry happens during ``git gc``, which is
exactly what broke. ``repair_broken_shallow_boundaries()`` re-appends the missing
boundaries; the updater calls it before the prune at both call sites.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import hermes_cli.gitlock as gitlock


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=check,
    )


def fixture(tmp_path):
    origin = tmp_path / "origin"; origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "config", "user.email", "t@example.com"); git(origin, "config", "user.name", "t")
    git(origin, "commit", "--allow-empty", "-qm", "c0")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--depth", "1", origin.as_uri(), str(clone)], check=True)
    git(origin, "commit", "--allow-empty", "-qm", "c1")
    git(origin, "commit", "--allow-empty", "-qm", "c2")
    git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    git(origin, "commit", "--allow-empty", "-qm", "c3")
    git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    return clone


def corrupt_fixture(clone):
    path = clone / ".git" / "shallow"
    lines = path.read_text(encoding="utf-8").splitlines()
    for removed in lines:
        path.write_text("\n".join(x for x in lines if x != removed) + "\n", encoding="utf-8")
        fsck = git(clone, "fsck", "--connectivity-only", check=False)
        if "broken link" in (fsck.stdout + fsck.stderr) or "missing commit" in (fsck.stdout + fsck.stderr):
            return removed
    raise AssertionError("fixture did not create a broken shallow boundary")


def _walks(clone):
    return git(clone, "rev-list", "--count", "--all", "--reflog", check=False).returncode == 0


def test_repair_restores_boundary_for_reflog_only_commit_with_unfetched_parent(tmp_path):
    clone = fixture(tmp_path)
    corrupt_fixture(clone)
    assert not _walks(clone)
    assert "broken link" in git(clone, "fsck", "--connectivity-only", check=False).stdout
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert _walks(clone)
    fsck = git(clone, "fsck", "--connectivity-only")
    assert "broken link" not in (fsck.stdout + fsck.stderr)
    assert git(clone, "gc", "-q").returncode == 0


def test_repair_then_prune_leaves_repo_walkable(tmp_path):
    """The production updater sequence: repair runs, then the prune immediately after.

    Without the prune's ``--reflog`` fail-safe walk, the prune dropped the boundary
    repair had just restored, re-breaking the repo on every ``hermes update`` run.
    """
    clone = fixture(tmp_path)
    corrupt_fixture(clone)
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert _walks(clone)
    gitlock.prune_stale_shallow_grafts(clone)
    assert _walks(clone)
    fsck = git(clone, "fsck", "--connectivity-only", check=False)
    assert "broken link" not in (fsck.stdout + fsck.stderr)


def test_repair_is_noop_on_healthy_shallow_checkout(tmp_path):
    clone = fixture(tmp_path); path = clone / ".git" / "shallow"; before = path.read_bytes()
    assert gitlock.repair_broken_shallow_boundaries(clone) == 0
    assert path.read_bytes() == before


def test_repair_is_noop_on_full_clone_without_shallow_file(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir(); git(repo, "init", "-q")
    assert gitlock.repair_broken_shallow_boundaries(repo) == 0


def test_repair_never_raises_on_broken_repo(tmp_path):
    assert gitlock.repair_broken_shallow_boundaries(tmp_path / "missing") == 0


def test_repair_does_not_touch_reflogs(tmp_path):
    clone = fixture(tmp_path); corrupt_fixture(clone)
    before = git(clone, "reflog", "show", "--all").stdout
    gitlock.repair_broken_shallow_boundaries(clone)
    assert git(clone, "reflog", "show", "--all").stdout == before


def test_repair_is_idempotent(tmp_path):
    clone = fixture(tmp_path); corrupt_fixture(clone)
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert gitlock.repair_broken_shallow_boundaries(clone) == 0


def test_repair_ignores_parent_lines_inside_commit_messages(tmp_path):
    """A ``parent <sha>`` line in a commit *message body* is prose, not an edge:
    the parent parser must read only the commit header, so a healthy commit whose
    message mentions ``parent <sha>`` is never shallow-marked."""
    origin = tmp_path / "origin"; origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "config", "user.email", "t@example.com"); git(origin, "config", "user.name", "t")
    git(origin, "commit", "--allow-empty", "-qm", "c0")
    git(origin, "commit", "--allow-empty", "-qm", "c1")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--depth", "2", origin.as_uri(), str(clone)], check=True)
    git(clone, "config", "user.email", "t@example.com"); git(clone, "config", "user.name", "t")
    git(clone, "commit", "--allow-empty", "-m", "subject\n\nbody line\n\nparent ffffffffffffffffffffffffffffffffffffffff")
    head = git(clone, "rev-parse", "HEAD").stdout.strip()
    # HEAD's real parent exists locally; the message-body "parent" line names an
    # absent object and must NOT register as a missing edge (header-only parsing).
    assert gitlock._batch_missing_parents(clone, [head]) == set()
    assert git(clone, "rev-list", "--count", "HEAD").stdout.strip() == "3"


def test_repair_does_not_mask_unrelated_object_loss(tmp_path):
    """Missing objects that are NOT reflog-only boundary commits must stay visible
    to fsck — repair must not relabel arbitrary object loss as shallow history."""
    clone = fixture(tmp_path)
    git(clone, "config", "user.email", "t@example.com"); git(clone, "config", "user.name", "t")
    git(clone, "commit", "--allow-empty", "-qm", "local1")
    git(clone, "commit", "--allow-empty", "-qm", "local2")
    # Drop HEAD's parent object from the object store.
    victim = git(clone, "rev-parse", "HEAD~1").stdout.strip()
    loose = clone / ".git" / "objects" / victim[:2] / victim[2:]
    assert loose.is_file(), "expected a loose object to delete"
    loose.unlink()
    assert gitlock.repair_broken_shallow_boundaries(clone) == 0
    fsck = git(clone, "fsck", "--connectivity-only", check=False)
    assert "missing" in (fsck.stdout + fsck.stderr) or "broken link" in (fsck.stdout + fsck.stderr)
