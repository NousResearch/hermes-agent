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
import sqlite3
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime state that lives at $HERMES_HOME's root on a flat install, exactly as
# ``hermes update`` would sweep it: one representative per ignored class. The
# sidecar names mirror ``_sqlite_files`` in gateway/platforms/base.py; the
# credential entries mirror ``_ROOT_CREDENTIAL_PATHS`` there.
FLAT_INSTALL_RUNTIME_STATE = (
    "state.db",
    "state.db-wal",
    "state.db-shm",
    "state.db-journal",
    "state.db.retired-wal-20260914T000000Z-1234/manifest.json",
    "kanban.db",
    "response_store.db",
    "response_store.db-wal",
    "gateway/discord_message_recovery.db",
    "state-snapshots/2026-09-14T06-00-00-pre-update/state.db",
    "sessions/2026-09-14_06-00-00_abcd123d.jsonl",
    "browser-profile/Cookies",
    "cron/executions.db",
    "cron/executions.db-wal",
    "cron/executions.db-shm",
    "cron/jobs.json",
    "cron.pid",
    "gateway.lock",
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
    "hook_outputs/2026-09-14_06-00-00/tool.json",
    "hooks/on_session_end.sh",
    "cache/banner_snapshot.json",
    "checkpoints/abcd123d/0001.json",
    "pending_messages/telegram.json",
    "plugin-data/example/state.json",
    "kanban/boards/x",
    "config.yaml",
    "auth.json",
    "auth.lock",
    "auth/google_oauth.json",
    ".anthropic_oauth.json",
    "google_token.json",
    "google_oauth_pending.json",
    "webhook_subscriptions.json",
    "channel_directory.json",
    "channel_aliases.json",
    "feishu_comment_pairing.json",
    "memories/MEMORY.md",
    "profiles/work/config.yaml",
    "credentials/github_token",
    "mcp-tokens/server.json",
    "pairing/telegram.json",
    "platforms/pairing/x.json",
    "backups/2026-09-14T06-00-00-pre-update/state.db",
    "vault.key",
    "vault.json.enc",
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


def test_untracked_autostash_leaves_open_wal_database_readable(flat_install_repo):
    """The updater's real stash step must not unlink the -wal/-shm sidecars of a
    WAL-mode database the scheduler holds open (cron/executions.db). With the
    base file ignored but its sidecars swept, the next ``cron.executions._connect()``
    finds a database whose WAL vanished under a live writer and fails with
    ``disk I/O error`` (the review repro on #111175)."""
    from hermes_cli.update_cmd_stash import _stash_local_changes_if_needed

    db_path = flat_install_repo / "cron" / "executions.db"
    db_path.unlink()  # the fixture's placeholder is not a database
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE executions (id INTEGER PRIMARY KEY, status TEXT)")
        conn.execute("INSERT INTO executions (status) VALUES ('ok')")
        conn.commit()
        assert (db_path.parent / "executions.db-wal").exists()
        # A tracked local change makes the updater actually enter its stash step.
        (flat_install_repo / "app.py").write_text("print('changed')\n")

        stash_ref = _stash_local_changes_if_needed(["git"], flat_install_repo)

        assert stash_ref
        # Fresh connection, exactly as every cron.executions call opens one.
        with sqlite3.connect(db_path) as reader:
            assert reader.execute("SELECT status FROM executions").fetchall() == [("ok",)]
    finally:
        conn.close()
