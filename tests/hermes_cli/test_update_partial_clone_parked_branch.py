"""The parked-branch liveness probe must not need the promisor remote.

Live incident (2026-09-28/29, macOS): a patch branch parked on
``fix/gateway-identity-file-unlink`` reported

    ⚠ CODE UPDATE SKIPPED — checkout is parked on '<branch>'
      Not auto-switching to main: the branch state could not be verified
      against origin/main.

with ``updates.parked_branch_strategy: update_in_place`` already set, so the
configured in-place merge never ran and the checkout stayed ~1700 commits
behind. The pre-flight's own ``git fetch`` had succeeded, which is what made it
look like a false alarm.

Cause: ``_assess_parked_branch_switch`` asks ``git cherry origin/<target>``,
whose patch-id needs each commit's TREE **and the blobs it touches**. This
checkout is a partial clone (``remote.origin.partialclonefilter=tree:0``), so
those objects are fetched on demand from the promisor remote: a pre-update
*liveness* check silently became a full-history download. On an intermittent
link the lazy fetch fails, cherry exits non-zero, the assessment reports
``unverifiable``, and the caller SKIPs the update before
``parked_branch_strategy`` is ever read.

The clone below reproduces that state faithfully: ``origin/main`` is advanced
AFTER the clone exists and pulled with the tree:0 filter (commits without
trees), and the promisor remote is then pointed at a path that does not exist,
so any lazy fetch fails immediately.

Temp directories are managed with :mod:`tempfile` rather than pytest's
``tmp_path`` so the fixture behaves the same under every runner.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from hermes_cli.update_cmd_git import _assess_parked_branch_switch

GIT = ["git"]


def _git(cwd, *args, check=True):
    """Run git with proxy variables stripped: these remotes are local paths."""
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    return subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, text=True,
                          check=check, env=env)


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _build_partial_clone(root: Path) -> Path:
    src = root / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    (src / "a.txt").write_text("x" * 4000 + "\nbase\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "c1")

    bare = root / "origin.git"
    _git(root, "init", "-q", "--bare", "-b", "main", str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(src, "remote", "add", "origin", str(bare))
    _git(src, "push", "-q", "origin", "main")

    clone = root / "clone"
    _git(root, "clone", "-q", "--filter=tree:0", "--single-branch", "--branch", "main",
         bare.as_uri(), str(clone))
    _git(clone, "config", "user.email", "t@example.com")
    _git(clone, "config", "user.name", "t")

    # Advance origin/main AFTER the clone exists, then tree:0-fetch: the commits arrive
    # without their trees, which is what makes cherry reach for the promisor remote.
    for i in range(2, 14):
        (src / f"f{i}.txt").write_text(("y" * 4000) + f"\nfile {i}\n")
        _git(src, "add", ".")
        _git(src, "commit", "-qm", f"c{i}")
    _git(src, "push", "-q", "origin", "main")
    _git(clone, "fetch", "-q", "origin", "main")

    # Park the branch with a commit of its own.
    _git(clone, "checkout", "-q", "-b", "feature", "main")
    (clone / "parked.txt").write_text("parked work\n")
    _git(clone, "add", ".")
    _git(clone, "commit", "-qm", "parked commit")

    # Any lazy fetch now fails immediately (no network, no fixture to hang on).
    _git(clone, "remote", "set-url", "origin", str(root / "GONE.git"))
    return clone


@pytest.fixture()
def partial_clone():
    """A real ``tree:0`` partial clone parked on ``feature``, promisor remote broken."""
    if not _git_available():
        pytest.skip("git unavailable")
    root = Path(tempfile.mkdtemp(prefix="hermes-partial-clone-"))
    try:
        yield _build_partial_clone(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_partial_clone_is_detected(partial_clone):
    """The clone reports the partial filter the fix keys off."""
    from hermes_cli.update_cmd_git import _is_partial_clone

    assert _is_partial_clone(GIT, partial_clone) is True


def test_parked_branch_assessed_without_the_promisor_remote(partial_clone):
    """RED on base (``unverifiable`` -> the update SKIPs); GREEN with the fix.

    This is the regression that stranded a real checkout: the assessment must
    resolve from local data even when the remote is unreachable.
    """
    switch_safe, reason = _assess_parked_branch_switch(GIT, partial_clone, "feature", "main")

    assert switch_safe is True, (
        f"assessment must not require the promisor remote (got {reason!r}); "
        "'unverifiable' makes the caller SKIP the whole update"
    )
    assert reason == "unmerged:1", f"expected one unmerged parked commit, got {reason!r}"


def test_unmerged_count_is_offline_and_counts_only_the_branch_commits(partial_clone):
    """The count comes from reachability, and merge commits are not double-counted."""
    from hermes_cli.update_cmd_git import _count_unmerged_parked_commits

    assert _count_unmerged_parked_commits(GIT, partial_clone, "main") == 1


def test_unmerged_count_is_none_when_the_repo_cannot_be_read():
    """A genuinely unreadable repository still reports None -> ``unverifiable``."""
    from hermes_cli.update_cmd_git import _count_unmerged_parked_commits

    root = Path(tempfile.mkdtemp(prefix="hermes-not-a-repo-"))
    try:
        assert _count_unmerged_parked_commits(GIT, root, "main") is None
    finally:
        shutil.rmtree(root, ignore_errors=True)
