"""Regression: a failed update fetch must not be reported as a ready repository.

The repository stage runs ``git fetch origin $BRANCH`` unguarded. The staged
entry point runs the stage body inside a subshell that inherits ``set +e``
(scripts/install.sh, run_stage), so the failure is ignored: the stage continues,
prints "Repository ready" and exits 0. The checkout is left on whatever it
already had, and callers consuming the JSON contract see {ok: true} for an
update that fetched nothing.

Any fetch failure triggers this. The one observed in the wild is HTTP 429 from
github.com, which this repo's network is large enough to hit in bursts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _managed_checkout(tmp_path: Path, *, local_changes: bool) -> Path:
    """A managed checkout whose origin cannot be fetched.

    Moving the bare remote aside after cloning makes ``git fetch origin main``
    fail the way a transient network error or a rate limit does, without having
    to reproduce one.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init")
    (seed / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "tracked.txt")
    _git(seed, "commit", "-m", "base")
    _git(seed, "branch", "-M", "main")

    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")

    managed = tmp_path / "hermes-agent"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(managed))

    if local_changes:
        (managed / "tracked.txt").write_text("local edit\n", encoding="utf-8")

    remote.rename(tmp_path / "origin.git.gone")
    return managed


def _run_repository_stage(tmp_path: Path, managed: Path) -> subprocess.CompletedProcess:
    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }
    return subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "repository", "--non-interactive"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


requires_shell = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_failed_fetch_fails_the_stage(tmp_path: Path) -> None:
    """The headline bug: an unreachable origin still reports a ready repository."""
    managed = _managed_checkout(tmp_path, local_changes=False)

    result = _run_repository_stage(tmp_path, managed)
    output = result.stdout + result.stderr

    assert "Repository ready" not in output, output
    assert result.returncode != 0, output


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_failed_fetch_is_named_as_such(tmp_path: Path) -> None:
    """The user is told what failed, not left with a bare git error."""
    managed = _managed_checkout(tmp_path, local_changes=False)

    result = _run_repository_stage(tmp_path, managed)
    output = result.stdout + result.stderr

    assert "Failed to fetch" in output, output


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_stashed_changes_are_pointed_at_on_failure(tmp_path: Path) -> None:
    """Local work stashed before the fetch must not go unmentioned."""
    managed = _managed_checkout(tmp_path, local_changes=True)

    result = _run_repository_stage(tmp_path, managed)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert _git(managed, "stash", "list").stdout.strip(), "stash must survive"
    assert "git stash apply" in output, output
    assert "hermes-install-autostash-" in output, output


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_no_stash_hint_when_nothing_was_stashed(tmp_path: Path) -> None:
    """The recovery hint must not misfire on a clean checkout."""
    managed = _managed_checkout(tmp_path, local_changes=False)

    result = _run_repository_stage(tmp_path, managed)
    output = result.stdout + result.stderr

    assert "git stash apply" not in output, output
