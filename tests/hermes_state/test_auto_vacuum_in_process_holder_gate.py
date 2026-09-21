"""Automatic VACUUM is refused while THIS process holds another generation of the same store.

``foreign_state_db_holders`` skips ``os.getpid()``, so it can only ever prove other PROCESSES
are away. Under the one-process-per-host multiplexer the dangerous holder is usually a sibling
SessionDB generation inside this very process: the VACUUM's TRUNCATE checkpoint retires the WAL
generation that handle is still bound to.
"""

from __future__ import annotations

import shutil

import hermes_state_registry as registry


def _seed(path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    db.create_session("old", "cli")
    db.append_message("old", "user", "hello")
    db.end_session("old", "done")
    db.close()


def _auto_maintenance(db):
    # retention_days=0 makes the ended row prunable; the freelist floor is disabled so only a
    # holder can stop the VACUUM.
    return db.maybe_auto_prune_and_vacuum(
        retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0,
        min_vacuum_freelist_ratio=-1.0)


def test_auto_vacuum_skips_while_this_process_holds_another_generation(tmp_path):
    db_path = tmp_path / "state.db"
    _seed(db_path)

    first = registry.acquire(db_path)
    try:
        # Snapshot restore / recovery swap shape: the file is replaced, so the next acquire
        # RETIRES `first` (still open, still held) and opens a fresh generation.
        replacement = tmp_path / "replacement.db"
        shutil.copy2(db_path, replacement)
        replacement.replace(db_path)
        second = registry.acquire(db_path)
        try:
            assert second is not first, "inode replacement did not mint a new generation"
            result = _auto_maintenance(second)
            assert result["pruned"] == 1
            assert result["vacuumed"] is False, "VACUUM ran under a live in-process holder"
            assert result.get("vacuum_skipped_holders")
        finally:
            registry.release(second)
    finally:
        registry.release(first)

    # Control: the store is quiet once the other generation is released, so the same call VACUUMs.
    only = registry.acquire(db_path)
    try:
        only.set_meta("last_auto_prune", "0")
        only.create_session("old2", "cli")
        only.end_session("old2", "done")
        again = _auto_maintenance(only)
        assert again["vacuumed"] is True
        assert "vacuum_skipped_holders" not in again
    finally:
        registry.release(only)
