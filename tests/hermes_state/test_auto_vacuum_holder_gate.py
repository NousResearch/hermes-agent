"""Automatic state.db maintenance never VACUUMs under a foreign writer (#110054).

``maybe_auto_prune_and_vacuum`` runs the SAME store rewrite as ``hermes sessions optimize``
(VACUUM + TRUNCATE checkpoint) from CLI and gateway startup. The manual command refuses while a
sibling writer holds the store; the automatic producer must skip for the same reason — a rewritten
generation under a live holder is what leaves every agent refusing turns. It only ever skips.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="holder scan is unavailable on Windows")

_HOLDER = (
    "import sqlite3, sys\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT count(*) FROM sqlite_master')\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
)


def _seeded_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old", "cli")
    db.append_message("old", "user", "hello")
    db.end_session("old", "done")
    return db


def _auto_maintenance(db):
    # retention_days=0 makes the ended row prunable; the freelist floor is disabled so only the
    # holder scan can stop the VACUUM.
    return db.maybe_auto_prune_and_vacuum(
        retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0,
        min_vacuum_freelist_ratio=-1.0)


def test_auto_vacuum_skips_while_a_foreign_process_holds_the_store(tmp_path):
    db = _seeded_db(tmp_path)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db.db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        result = _auto_maintenance(db)
        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result.get("vacuum_skipped_holders")

        # Control: the same call VACUUMs once the holder is gone.
        holder.stdin.close()
        holder.wait(timeout=10)
        db.set_meta("last_auto_prune", "0")
        db.create_session("old2", "cli")
        db.end_session("old2", "done")
        again = _auto_maintenance(db)
        assert again["vacuumed"] is True
        assert "vacuum_skipped_holders" not in again
    finally:
        if holder.poll() is None:
            holder.stdin.close()
            holder.wait(timeout=10)
        db.close()
