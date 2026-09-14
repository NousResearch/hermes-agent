"""Flat-install runtime state must be gitignored so ``hermes update``'s untracked
autostash cannot sweep the live state.db (#110648).

On a flat install (checkout root == $HERMES_HOME) the profile's runtime files
live inside the repo as untracked paths. ``git stash push --include-untracked``
(hermes_cli/update_cmd_stash.py) moves the whole untracked set into the stash and
unlinks it from the working tree under the running gateway, silently stranding
every transcript when the restore is declined or fails its health check. The
tracked .gitignore must cover the runtime state set, mirroring the
.hermes-bootstrap-complete / .install_method precedent (#38529 / #66189).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime state that lives at $HERMES_HOME's root on a flat install, exactly as
# ``hermes update`` would sweep it: the session store plus its SQLite sidecars
# (gateway/platforms/base.py _ROOT_CREDENTIAL_PATHS enumerates the same set) and
# retired-WAL capture dirs, quick snapshots, the legacy transcript dir, the
# default kanban board, the cron executions ledger, gateway lock/pid files and
# the cache/spill directories.
FLAT_INSTALL_RUNTIME_STATE = (
    "state.db",
    "state.db-wal",
    "state.db-shm",
    "state.db-journal",
    "state.db.retired-wal-20260914T000000Z-1234/manifest.json",
    "kanban.db",
    "kanban.db-wal",
    "kanban.db-shm",
    "kanban.db-journal",
    "state-snapshots/2026-09-14T06-00-00-pre-update/state.db",
    "sessions/2026-09-14_06-00-00_abcd123d.jsonl",
    "browser-profile/Cookies",
    "cron/executions.db",
    "gateway.lock",
    "gateway.pid",
    "hook_outputs/2026-09-14_06-00-00/tool.json",
    "cache/banner_snapshot.json",
)


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


@pytest.fixture
def flat_install_repo(tmp_path: Path) -> Path:
    """A real git repo standing in for a flat install, with the tracked .gitignore.

    Built in a subdirectory of tmp_path: the suite-wide HERMES_HOME isolation
    fixture (tests/conftest.py) materialises its own ``hermes_test/`` tree in
    tmp_path itself, which is not part of this repo's story.
    """
    repo = tmp_path / "flat-install-checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copyfile(REPO_ROOT / ".gitignore", repo / ".gitignore")
    (repo / "app.py").write_text("print('hermes')\n")
    _run_git(repo, "add", ".gitignore", "app.py")
    _run_git(
        repo,
        "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "init",
    )
    for rel in FLAT_INSTALL_RUNTIME_STATE:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime state")
    return repo


def test_flat_install_runtime_state_is_ignored(flat_install_repo):
    """`git status --porcelain` must stay empty with the full runtime state present,
    so `hermes update` never enters its stash step for runtime state alone."""
    status = _run_git(
        flat_install_repo, "status", "--porcelain", "--untracked-files=all"
    )
    assert status.stdout == "", status.stdout


def test_untracked_autostash_cannot_sweep_runtime_state(flat_install_repo):
    """The exact ``git stash push --include-untracked`` the updater runs must leave
    every runtime state file in the working tree (issue repro, step 6)."""
    # A tracked local change proves the stash really ran: it must be swept away
    # while the runtime state survives.
    (flat_install_repo / "app.py").write_text("print('changed')\n")
    _run_git(
        flat_install_repo,
        "stash", "push", "--include-untracked", "-m", "hermes-update-autostash",
    )
    assert (flat_install_repo / "app.py").read_text() == "print('hermes')\n"
    missing = [
        rel
        for rel in FLAT_INSTALL_RUNTIME_STATE
        if not (flat_install_repo / rel).exists()
    ]
    assert missing == []
