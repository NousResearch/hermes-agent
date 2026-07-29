"""Regression: managed installs must be able to see the repo's other branches.

``git clone --depth 1 --branch main`` implies ``--single-branch``, which pins
``remote.origin.fetch`` to ``+refs/heads/main:refs/remotes/origin/main``. On
such a checkout ``git fetch origin <other>`` updates FETCH_HEAD but never
creates ``origin/<other>``, so other branches are invisible to ``git branch
-r`` / ``gh pr checkout`` and ``hermes update --branch <other>`` fails with
"does not exist locally or on origin".

The installer restores the wildcard refspec. That only costs bandwidth if
something issues a bare ``git fetch``, so the startup update check has to keep
naming its branch.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_upstream(tmp_path: Path) -> Path:
    """A bare remote with `main` plus a couple of side branches."""
    upstream = tmp_path / "upstream.git"
    seed = tmp_path / "seed"
    _git(tmp_path, "init", "--bare", "-b", "main", str(upstream))
    _git(tmp_path, "clone", upstream.as_uri(), str(seed))
    (seed / "a.txt").write_text("one\n")
    _git(seed, "add", "a.txt")
    _git(seed, "commit", "-m", "init")
    for branch in ("feat-1", "feat-2"):
        _git(seed, "checkout", "-b", branch, "main")
        (seed / f"{branch}.txt").write_text(branch)
        _git(seed, "add", ".")
        _git(seed, "commit", "-m", branch)
    _git(seed, "checkout", "main")
    _git(seed, "push", "--all", "origin")
    return upstream


def _narrow_clone(tmp_path: Path, upstream: Path) -> Path:
    """Clone exactly the way the installer does: shallow and single-branch."""
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "--depth", "1", "--branch", "main",
         upstream.as_uri(), str(checkout))
    assert _git(checkout, "config", "--get-all", "remote.origin.fetch") == (
        "+refs/heads/main:refs/remotes/origin/main"
    )
    return checkout


def _widen(checkout: Path) -> None:
    """Run install.sh's widen_remote_branches() against a real checkout."""
    match = re.search(r"widen_remote_branches\(\) \{.*?\n\}", INSTALL_SH.read_text(),
                      re.DOTALL)
    assert match is not None, "widen_remote_branches() not found in install.sh"
    script = f"{match.group(0)}\nwiden_remote_branches {shlex.quote(str(checkout))}\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_narrow_clone_cannot_resolve_other_branches(tmp_path: Path) -> None:
    """The premise: without the fix, fetching a branch by name is not enough."""
    checkout = _narrow_clone(tmp_path, _make_upstream(tmp_path))

    _git(checkout, "fetch", "origin", "feat-1")
    assert _git(checkout, "branch", "-r").split() == ["origin/main"]

    # This is the exact pair of commands `hermes update --branch feat-1` runs.
    for args in (("checkout", "feat-1"), ("checkout", "-B", "feat-1", "origin/feat-1")):
        assert subprocess.run(["git", *args], cwd=checkout,
                              capture_output=True).returncode != 0


def test_widening_makes_other_branches_resolvable(tmp_path: Path) -> None:
    checkout = _narrow_clone(tmp_path, _make_upstream(tmp_path))
    _widen(checkout)

    assert _git(checkout, "config", "--get-all", "remote.origin.fetch") == (
        "+refs/heads/*:refs/remotes/origin/*"
    )
    _git(checkout, "fetch", "--depth", "1", "origin", "feat-1")
    assert _git(checkout, "branch", "-r").split() == ["origin/feat-1", "origin/main"]
    _git(checkout, "checkout", "-B", "feat-1", "origin/feat-1")
    assert _git(checkout, "rev-parse", "--abbrev-ref", "HEAD") == "feat-1"


def test_widening_downloads_nothing_by_itself(tmp_path: Path) -> None:
    """Rewriting the refspec must not pull refs the user didn't ask for."""
    checkout = _narrow_clone(tmp_path, _make_upstream(tmp_path))
    before = _git(checkout, "rev-list", "--all", "--objects", "--count")

    _widen(checkout)

    assert _git(checkout, "branch", "-r").split() == ["origin/main"]
    assert _git(checkout, "rev-list", "--all", "--objects", "--count") == before


def test_widening_is_idempotent(tmp_path: Path) -> None:
    """Re-running the installer over an already-widened checkout is a no-op."""
    checkout = _narrow_clone(tmp_path, _make_upstream(tmp_path))
    _widen(checkout)
    _widen(checkout)

    assert _git(checkout, "config", "--get-all", "remote.origin.fetch") == (
        "+refs/heads/*:refs/remotes/origin/*"
    )


def test_update_check_stays_scoped_on_a_widened_checkout(tmp_path: Path) -> None:
    """The startup banner check must not pull every branch once it can see them.

    A bare ``git fetch origin`` on a widened checkout would drag in every
    branch — hundreds of MB on the real repo, behind a 10s timeout, on every
    launch. Naming main keeps the transfer to one ref.
    """
    import hermes_cli.banner as banner

    upstream = _make_upstream(tmp_path)
    checkout = _narrow_clone(tmp_path, upstream)
    _widen(checkout)

    # Move main forward so the check has something to report.
    seed = tmp_path / "seed"
    (seed / "a.txt").write_text("two\n")
    _git(seed, "commit", "-am", "advance main")
    _git(seed, "push", "origin", "main")

    assert banner._check_via_local_git(checkout) == banner.UPDATE_AVAILABLE_NO_COUNT
    assert _git(checkout, "branch", "-r").split() == ["origin/main"]
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "true"
