from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hermes_cli import update_cmd


GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Hermes Test",
    "GIT_AUTHOR_EMAIL": "hermes-test@example.invalid",
    "GIT_COMMITTER_NAME": "Hermes Test",
    "GIT_COMMITTER_EMAIL": "hermes-test@example.invalid",
}


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=GIT_ENV,
        text=True,
        capture_output=True,
        check=check,
    )


def make_diverged_fork(tmp_path: Path) -> tuple[Path, Path]:
    upstream_bare = tmp_path / "upstream.git"
    fork_bare = tmp_path / "fork.git"
    seed = tmp_path / "seed"
    work = tmp_path / "work"

    git(tmp_path, "init", "--bare", str(upstream_bare))
    git(tmp_path, "init", "seed")
    git(seed, "checkout", "-b", "main")
    (seed / "shared.txt").write_text("base\n", encoding="utf-8")
    git(seed, "add", "shared.txt")
    git(seed, "commit", "-m", "base")
    git(seed, "remote", "add", "origin", str(upstream_bare))
    git(seed, "push", "-u", "origin", "main")

    git(tmp_path, "clone", "--bare", str(upstream_bare), str(fork_bare))
    git(tmp_path, "clone", str(fork_bare), str(work))
    git(work, "checkout", "main")
    git(work, "remote", "add", "upstream", str(upstream_bare))

    (work / "repair.txt").write_text("bounded repair\n", encoding="utf-8")
    git(work, "add", "repair.txt")
    git(work, "commit", "-m", "repair")
    git(work, "push", "origin", "main")

    (seed / "upstream.txt").write_text("new upstream\n", encoding="utf-8")
    git(seed, "add", "upstream.txt")
    git(seed, "commit", "-m", "upstream update")
    git(seed, "push", "origin", "main")
    git(work, "fetch", "origin", "main")
    git(work, "fetch", "upstream", "main")
    return work, fork_bare


def test_customized_fork_merges_upstream_and_pushes_without_force(tmp_path: Path) -> None:
    work, fork_bare = make_diverged_fork(tmp_path)

    assert update_cmd._merge_diverged_fork_with_upstream(["git"], work) is True

    git(work, "fetch", "origin", "main")
    assert git(work, "merge-base", "--is-ancestor", "upstream/main", "HEAD").returncode == 0
    assert (work / "repair.txt").read_text(encoding="utf-8") == "bounded repair\n"
    assert (work / "upstream.txt").read_text(encoding="utf-8") == "new upstream\n"
    assert git(fork_bare, "rev-parse", "main").stdout.strip() == git(work, "rev-parse", "HEAD").stdout.strip()


def test_customized_fork_sync_fails_closed_on_dirty_tree(tmp_path: Path) -> None:
    work, _ = make_diverged_fork(tmp_path)
    before = git(work, "rev-parse", "HEAD").stdout.strip()
    (work / "dirty.txt").write_text("do not overwrite\n", encoding="utf-8")

    assert update_cmd._merge_diverged_fork_with_upstream(["git"], work) is False

    assert git(work, "rev-parse", "HEAD").stdout.strip() == before
    assert (work / "dirty.txt").read_text(encoding="utf-8") == "do not overwrite\n"
