"""Automatic maintenance of a SQLite backend store never VACUUMs under a foreign writer (#110054).

``session_retention.run_auto_prune_and_vacuum`` runs the SAME store rewrite as ``hermes sessions optimize``
(VACUUM + TRUNCATE checkpoint) from CLI and gateway startup. The manual command refuses while a
sibling writer holds the store; the automatic producer must skip for the same reason: a rewritten
generation under a live holder is what leaves every agent refusing turns. It only ever skips.
"""

import subprocess
import sys

import pytest

from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session
from session_retention import run_auto_prune_and_vacuum
from tests.contracts._scopes import MEDIA_ROOT

pytestmark = pytest.mark.platforms("posix")  # holder scan is unavailable on Windows

_HOLDER = (
    "import sqlite3, sys\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT count(*) FROM sqlite_master')\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
)


def _seeded_store(backend, scope):
    store = backend.open(scope)
    db = store.holder()
    create_session(db, "old", "cli")
    append_message(db, "old", "user", "hello")
    end_session(db, "old", "done")
    return store


def _auto_maintenance(store, admin):
    # retention_days=0 makes the ended row prunable; the freelist floor is disabled so only the
    # holder scan can stop the VACUUM.
    return run_auto_prune_and_vacuum(
        store, admin, store_home=store.holder().db_path.parent, media_root=MEDIA_ROOT,
        retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0,
        min_vacuum_freelist_ratio=-1.0)


def test_auto_vacuum_skips_while_a_foreign_process_holds_the_store(backend, scope_a):
    store = _seeded_store(backend, scope_a)
    db = store.holder()
    admin = backend.open_store_admin(scope_a, read_only=False)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db.db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        result = _auto_maintenance(store, admin)
        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result.get("vacuum_skipped_holders")

        # Control: the same call VACUUMs once the holder is gone.
        holder.stdin.close()
        holder.wait(timeout=10)
        db.set_meta("last_auto_prune", "0")
        create_session(db, "old2", "cli")
        end_session(db, "old2", "done")
        again = _auto_maintenance(store, admin)
        assert again["vacuumed"] is True
        assert "vacuum_skipped_holders" not in again
    finally:
        if holder.poll() is None:
            holder.stdin.close()
            holder.wait(timeout=10)
