"""Flat-install database-adjacent runtime artifacts must survive update autostash."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def flat_install_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "flat-install-checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copyfile(REPO_ROOT / ".gitignore", repo / ".gitignore")
    (repo / "app.py").write_text("print('hermes')\n")
    _git(repo, "add", ".gitignore", "app.py")
    _git(
        repo,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-qm",
        "init",
    )
    return repo


def test_database_adjacent_runtime_artifacts_are_ignored(flat_install_repo: Path):
    artifacts = (
        "state.db.quarantine.lock",
        "state.db.repair.lock",
        "state.db.fts_rebuild.lock",
        "state.db.auto-maintenance.lock",
        "state.db.repair-attempts.json",
        "state.db.malformed-backup-20260915_060000",
        "state.db.pre-update-emergency-2026-09-15T06-00-00-000Z.bak",
        "kanban.db.init.lock",
        "kanban.db.dispatch.lock",
        "kanban.db.corrupt.20260915.bak",
    )
    for name in artifacts:
        (flat_install_repo / name).write_bytes(b"runtime state")

    status = _git(
        flat_install_repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    assert status.stdout == "", status.stdout


@pytest.mark.skipif(os.name == "nt", reason="fcntl is POSIX-only")
def test_autostash_cannot_split_live_database_lock_inode(flat_install_repo: Path):
    import fcntl

    lock_path = flat_install_repo / "state.db.quarantine.lock"
    with lock_path.open("a+b") as first:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        first_stat = os.fstat(first.fileno())
        first_identity = first_stat.st_dev, first_stat.st_ino

        (flat_install_repo / "app.py").write_text("print('changed')\n")
        _git(
            flat_install_repo,
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "hermes-update-autostash",
        )

        assert lock_path.exists()
        assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == first_identity
        with lock_path.open("a+b") as second:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
