"""Regression coverage for updating an existing install to a remote tag.

Some Git versions leave ``git fetch origin <tag>`` only in ``FETCH_HEAD``.
The installer must materialize the requested tag before checking it out while
keeping the normal branch-update path attached to its branch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
REAL_GIT = shutil.which("git")
pytestmark = [
    pytest.mark.linux_only,
    pytest.mark.skipif(
        REAL_GIT is None or shutil.which("bash") is None,
        reason="needs git and bash",
    ),
]


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [REAL_GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _make_remote_with_unfetched_tag(tmp_path: Path) -> tuple[Path, Path, str]:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    (seed / "version.txt").write_text("old\n", encoding="utf-8")
    _git(seed, "add", "version.txt")
    _git(seed, "commit", "-qm", "old")

    _git(seed, "checkout", "-qb", "tag-source")
    (seed / "version.txt").write_text("tagged\n", encoding="utf-8")
    _git(seed, "commit", "-qam", "tagged")
    tag_commit = _git(seed, "rev-parse", "HEAD").stdout.strip()
    _git(seed, "tag", "release-test")
    _git(seed, "checkout", "-q", "main")

    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(
        seed,
        "push",
        "-q",
        "origin",
        "main",
        "main:refs/heads/nested/release-test",
        "refs/tags/release-test",
    )
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    managed = tmp_path / "hermes-agent"
    # Force the transport path: local-clone optimization copies every local
    # ref even with --no-tags, which would pre-materialize the regression tag.
    _git(
        tmp_path,
        "clone",
        "-q",
        "--no-tags",
        "--branch",
        "main",
        remote.as_uri(),
        str(managed),
    )
    # Local-clone transports may still opportunistically copy tag refs. The
    # reported update starts from a checkout where this target ref is absent.
    _git(managed, "update-ref", "-d", "refs/tags/release-test")
    assert _git(
        managed, "show-ref", "--verify", "refs/tags/release-test", check=False
    ).returncode != 0
    return managed, seed, tag_commit


def _git_wrapper(tmp_path: Path) -> Path:
    """Emulate Git versions that keep a fetched tag only in FETCH_HEAD."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text(
        """#!/usr/bin/env bash
"$REAL_GIT" "$@"
status=$?
if [ "$status" -eq 0 ] && [ "$1" = fetch ] && [ "$2" = origin ] && [ "$3" = release-test ]; then
    "$REAL_GIT" update-ref -d refs/tags/release-test
fi
exit "$status"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return bin_dir


def _run_repository_stage(
    tmp_path: Path, managed: Path, branch: str, *, emulate_fetch_head_only: bool = False
) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_INSTALL_DIR": str(managed),
    }
    if emulate_fetch_head_only:
        env["REAL_GIT"] = Path(REAL_GIT).as_posix()
        env["PATH"] = f"{_git_wrapper(tmp_path)}:{env['PATH']}"
    return subprocess.run(
        [
            "bash",
            str(INSTALL_SH),
            "--stage",
            "repository",
            "--branch",
            branch,
            "--non-interactive",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.live_system_guard_bypass
def test_existing_install_materializes_remote_tag_before_checkout(tmp_path: Path) -> None:
    managed, _seed, tag_commit = _make_remote_with_unfetched_tag(tmp_path)

    result = _run_repository_stage(
        tmp_path, managed, "release-test", emulate_fetch_head_only=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == tag_commit
    assert _git(
        managed, "show-ref", "--verify", "refs/tags/release-test"
    ).returncode == 0
    assert _git(managed, "symbolic-ref", "-q", "HEAD", check=False).returncode != 0


@pytest.mark.live_system_guard_bypass
def test_existing_install_branch_update_remains_attached(tmp_path: Path) -> None:
    managed, seed, _tag_commit = _make_remote_with_unfetched_tag(tmp_path)
    (seed / "version.txt").write_text("new branch\n", encoding="utf-8")
    _git(seed, "commit", "-qam", "new branch")
    branch_commit = _git(seed, "rev-parse", "HEAD").stdout.strip()
    _git(seed, "push", "-q", "origin", "main")

    result = _run_repository_stage(tmp_path, managed, "main")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == branch_commit
    assert _git(managed, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main"
