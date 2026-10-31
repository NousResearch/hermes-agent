"""Observation reads work on genuinely read-only storage without repairing it."""
import os
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_observations import read_session_observations_at_path


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="requires unprivileged filesystem permissions")
def test_observation_reader_never_repairs_or_writes_unwritable_storage(tmp_path):
    home = tmp_path / "profile"
    path = home / "state.db"
    with SessionDB(path) as db:
        db.create_session("s", "desktop", profile_name="secondary")
        holder = f"pid={os.getpid()}:turn=readonly"
        assert db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        assert db.open_session_attention("s", turn, "validation")
        assert db.finish_session_observation("s", holder, turn, "complete")
        db.release_session_turn_lease("s", holder)

    # Close every SessionDB connection before changing journal mode. A single-file
    # store remains readable without creating WAL sidecars on unwritable storage.
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"

    def snapshot():
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode) for p in home.iterdir()}

    try:
        for file in home.iterdir():
            file.chmod(0o444)
        home.chmod(0o555)
        assert not os.access(home, os.W_OK) and not os.access(path, os.W_OK)
        before = snapshot()
        row = read_session_observations_at_path(path, ["s"], profile="secondary")[0]
        assert row["execution"] == "idle" and row["turn_id"] == turn
        assert row["last_result"]["status"] == "complete"
        assert row["attention"]["kind"] == "validation"
        assert snapshot() == before, "read modified bytes, permissions, mtime, or sidecar membership"
        path.chmod(0)
        assert not os.access(path, os.R_OK)
        row = read_session_observations_at_path(path, ["s"], profile="secondary")[0]
        assert row["execution"] == "unknown" and row["lineage_tip_id"] is None
        assert path.stat().st_mode & 0o777 == 0, "read repaired inaccessible database permissions"
    finally:
        home.chmod(0o755)
        for file in home.iterdir():
            file.chmod(0o644)
