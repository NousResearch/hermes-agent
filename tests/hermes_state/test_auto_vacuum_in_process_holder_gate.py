"""Automatic VACUUM is refused under a LIVE in-process holder — and never under a retired one.

``foreign_state_db_holders`` skips ``os.getpid()``, so it can only ever prove other PROCESSES are
away. Under the one-process-per-host multiplexer the dangerous holder is a sibling SessionDB in
this very process: the VACUUM's TRUNCATE checkpoint retires the WAL generation that handle is
still bound to.

The mirror invariant matters just as much. A RETIRED generation (its file was replaced by a
recovery swap, backup restore or snapshot) is already write-fenced with ``StateDbReplacedError``
and its close-time checkpoint disabled, so it is not the writer this gate protects — and it leaves
the registry only when its holder releases, which a gateway handle does not do before shutdown.
Counting it made ONE inode replacement disable auto-VACUUM for that path for the rest of the
process lifetime, turning the unbounded growth this maintenance exists to bound into a permanent
condition.
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


def _make_prunable(db, session_id):
    db.set_meta("last_auto_prune", "0")
    db.create_session(session_id, "cli")
    db.end_session(session_id, "done")


def test_auto_vacuum_skips_while_a_live_sibling_holds_the_same_store(tmp_path):
    """Invariant: a genuinely LIVE sibling SessionDB for this path defers the VACUUM."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    _seed(db_path)

    live_sibling = registry.acquire(db_path)  # e.g. the gateway's own handle
    maintainer = SessionDB(db_path=db_path)   # a second handle running housekeeping
    try:
        _make_prunable(maintainer, "old2")
        result = _auto_maintenance(maintainer)
        assert result["pruned"] >= 1, result
        assert result["vacuumed"] is False, "VACUUM ran under a live in-process holder"
        assert result.get("vacuum_skipped_holders"), result
    finally:
        maintainer.close()
        registry.release(live_sibling)

    # Control: the same call VACUUMs once that live sibling is gone.
    quiet = SessionDB(db_path=db_path)
    try:
        _make_prunable(quiet, "old3")
        again = _auto_maintenance(quiet)
        assert again["vacuumed"] is True, again
        assert "vacuum_skipped_holders" not in again, again
    finally:
        quiet.close()


def test_auto_vacuum_is_not_starved_by_a_retired_write_fenced_generation(tmp_path):
    """RED on base: one inode replacement stopped auto-VACUUM for the process lifetime.

    The retired handle is never released (a gateway holds its SessionDB until shutdown), so on base
    every later round reported ``vacuum_skipped_holders`` and the store grew without bound.
    """
    db_path = tmp_path / "state.db"
    _seed(db_path)

    parked = registry.acquire(db_path)
    try:
        # Recovery swap / backup restore shape: the file is replaced, so the next acquire RETIRES
        # ``parked`` (still open, still held, already write-fenced) and opens a fresh generation.
        replacement = tmp_path / "replacement.db"
        shutil.copy2(db_path, replacement)
        replacement.replace(db_path)
        current = registry.acquire(db_path)
        try:
            assert current is not parked, "inode replacement did not mint a new generation"
            for round_index in range(3):  # consecutive rounds, fresh prunable rows each time
                _make_prunable(current, f"old-{round_index}")
                result = _auto_maintenance(current)
                assert result["pruned"] >= 1, (round_index, result)
                assert result["vacuumed"] is True, (round_index, result)
                assert "vacuum_skipped_holders" not in result, (round_index, result)
        finally:
            registry.release(current)
    finally:
        registry.release(parked)
