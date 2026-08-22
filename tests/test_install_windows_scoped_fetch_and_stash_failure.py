"""Regression for the Windows installer's repository-update parity fix.

Upstream #32384 ("`hermes update` Corrupts Git Repo and Breaks Installation")
was caused by a bare `git fetch origin` pulling every remote-tracking ref —
on a repo with thousands of auto-generated branches, a stale/rewritten one
can make the remote respond with "did not send all necessary objects" /
"bad object". `scripts/install.sh` already scopes the remote to just the
target branch (`git remote set-branches origin "$BRANCH"`) before fetching.
`scripts/install.ps1` did not — this file proves the Windows installer now
has the same protection, and that a `git stash push` failure during the
repository stage is reported as a STASH failure (not misattributed to the
next command that happens to also fail once the repo is already dirty).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (candidate for candidate in ("pwsh", "powershell") if shutil.which(candidate)),
    None,
)

pytestmark = [
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(
        shutil.which("git") is None or POWERSHELL is None,
        reason="needs git and PowerShell",
    ),
]


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _run_repository_stage(managed: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-File",
            str(INSTALL_PS1),
            "-Stage",
            "repository",
            "-NonInteractive",
            "-InstallDir",
            str(managed),
            "-HermesHome",
            str(tmp_path / "hermes-home"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _make_basic_managed_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A bare origin with two branches, plus a full (unscoped) local clone."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init")
    (seed / "tracked.txt").write_text("base\n", encoding="utf-8")
    (seed / "other.txt").write_text("unrelated\n", encoding="utf-8")
    _git(seed, "add", "tracked.txt", "other.txt")
    _git(seed, "commit", "-m", "base")
    _git(seed, "branch", "-M", "main")

    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")

    # A second branch so a full-mirror fetch has something extra to pull —
    # this is the branch we'll later make unreachable to simulate #32384.
    _git(seed, "checkout", "-b", "feature-x")
    (seed / "tracked.txt").write_text("feature edit\n", encoding="utf-8")
    _git(seed, "commit", "-am", "feature commit")
    _git(seed, "push", "origin", "feature-x")
    _git(seed, "checkout", "main")

    managed = tmp_path / "hermes-agent"
    # Full (unscoped) clone, like a pre-fix Windows install would have —
    # this is what lets a later full-mirror fetch touch feature-x at all.
    _git(tmp_path, "clone", str(remote), str(managed))
    _git(managed, "checkout", "main")

    return managed, remote


def test_install_ps1_scopes_origin_to_target_branch(tmp_path: Path) -> None:
    """After the repository stage, origin's fetch refspec must be branch-scoped —
    not the default wildcard mirror that let #32384 happen."""
    managed, _ = _make_basic_managed_checkout(tmp_path)

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, result.stderr
    refspecs = _git(managed, "config", "--get-all", "remote.origin.fetch").stdout.strip()
    assert refspecs == "+refs/heads/main:refs/remotes/origin/main", refspecs
    assert "+refs/heads/*:refs/remotes/origin/*" not in refspecs


def test_install_ps1_stale_unrelated_branch_cannot_poison_update(tmp_path: Path) -> None:
    """Reproduces the #32384 shape: a branch unrelated to the update target
    becomes unresolvable on the remote (simulating a force-push/rewrite/
    deletion of history the local full-mirror clone still references).
    A scoped update must succeed regardless — it never needs that branch."""
    managed, remote = _make_basic_managed_checkout(tmp_path)

    # Simulate the remote's feature-x becoming unreachable the way a
    # rewritten/deleted upstream branch does: force it to point at a
    # dangling, ungraspable commit id that was never pushed.
    _git(
        remote,
        "update-ref",
        "refs/heads/feature-x",
        "0" * 40,
        check=False,
    )

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "did not send all necessary objects" not in result.stdout
    assert "bad object" not in result.stdout


def test_install_ps1_stash_failure_stops_before_fetch_and_reports_stash(tmp_path: Path) -> None:
    """A `git stash push` that cannot save the index (the real #32384-class
    failure signature: "invalid object ... Cannot save the current index
    state") must be reported as a STASH failure and must never reach fetch."""
    managed, _ = _make_basic_managed_checkout(tmp_path)

    # Dirty the working tree so the installer's autostash path triggers.
    (managed / "tracked.txt").write_text("locally modified\n", encoding="utf-8")

    # Reproduce "Cannot save the current index state" as a hard stash
    # failure that creates NO stash entry: a stuck index.lock (the artifact
    # a crashed/killed prior git process leaves behind) makes `git stash
    # push` fail immediately with "Unable to create '.../index.lock': File
    # exists" — while `git status`/`rev-parse` (the installer's earlier
    # repo-validity precheck) don't take that lock and still succeed, so
    # this reproduces the failure at exactly the right point: past the
    # "is this a usable repo" gate, inside the stash step itself.
    lock_path = managed / ".git" / "index.lock"
    lock_path.write_text("", encoding="utf-8")

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "git stash failed" in combined, combined
    assert "git fetch failed" not in combined, combined
    # Nothing was fetched: origin's refspec was never touched, no new stash
    # was created (the failed push saved nothing), and the working tree is
    # exactly as this test left it dirty.
    assert _git(managed, "stash", "list").stdout.strip() == ""
    assert (managed / "tracked.txt").read_text(encoding="utf-8") == "locally modified\n"


def test_install_ps1_stash_created_despite_nonzero_exit_still_continues(tmp_path: Path) -> None:
    """`git stash push` can exit nonzero yet still have created the stash
    entry (e.g. it saved everything but failed to delete a swept file from
    the working tree). That must NOT be treated as the hard "nothing was
    saved" failure -- the update should continue, with the working tree
    reset to a clean state since the changes are already safe in the stash."""
    managed, _ = _make_basic_managed_checkout(tmp_path)
    (managed / "tracked.txt").write_text("locally modified\n", encoding="utf-8")

    # Reproduce the real-world trigger: an untracked file `--include-untracked`
    # must sweep into the stash, held open with a Windows exclusive lock so
    # git's post-save cleanup can delete everything else but not this one --
    # `stash push` saves the entry successfully, then exits nonzero only
    # because that one delete failed. Hold the handle for the whole install
    # call so the failure is genuinely still live when stash runs.
    untracked = managed / "locked-untracked.txt"
    untracked.write_text("swept but undeletable\n", encoding="utf-8")

    handle = open(untracked, "r+b")
    try:
        result = _run_repository_stage(managed, tmp_path)
    finally:
        handle.close()

    assert result.returncode == 0, result.stderr
    # Must NOT take the "nothing was saved" abort path.
    assert "git stash failed" not in result.stdout + result.stderr
    # Must have continued into (and completed) the update.
    assert '"ok":true' in result.stdout, result.stdout


def test_install_ps1_fetch_failure_reports_fetch_not_stash(tmp_path: Path) -> None:
    """When stash succeeds but fetch genuinely fails, the reported failure
    must name fetch — proving the fix didn't just relabel every failure as
    stash, only the ones that actually are."""
    managed, _ = _make_basic_managed_checkout(tmp_path)
    (managed / "tracked.txt").write_text("locally modified\n", encoding="utf-8")

    # Point origin at a URL that cannot be fetched from.
    _git(managed, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "git fetch failed" in combined, combined
    assert "git stash failed" not in combined, combined
    # The stash from before the (failed) fetch must be preserved, not lost.
    assert _git(managed, "stash", "list").stdout.strip() != ""


def test_install_ps1_preserves_preexisting_user_stashes(tmp_path: Path) -> None:
    """A stash the user already had before running the installer must
    survive an update untouched — the installer must only ever touch its
    own autostash entry."""
    managed, _ = _make_basic_managed_checkout(tmp_path)

    # A pre-existing user stash, unrelated to the installer's own autostash.
    (managed / "tracked.txt").write_text("user's own wip\n", encoding="utf-8")
    _git(managed, "stash", "push", "-m", "users-own-wip")
    assert _git(managed, "stash", "list").stdout.strip() != ""

    # New dirty state for the installer's own autostash to pick up.
    (managed / "other.txt").write_text("installer-triggering edit\n", encoding="utf-8")

    result = _run_repository_stage(managed, tmp_path)

    assert result.returncode == 0, result.stderr
    stash_list = _git(managed, "stash", "list").stdout
    assert "users-own-wip" in stash_list, stash_list
