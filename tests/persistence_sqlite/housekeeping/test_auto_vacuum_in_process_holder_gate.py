"""Automatic VACUUM on a SQLite backend store is never refused under a RETIRED in-process generation.

``SQLiteStoreAdmin.vacuum_blockers`` pairs the foreign-process scan with the registry's in-process arm
(``_registry.other_generations_for_path``). A RETIRED generation (its file was replaced by a recovery swap,
backup restore or snapshot) is already write-fenced with ``StateDbReplacedError`` and its close-time
checkpoint disabled, so it is not the writer this gate protects, and it leaves the registry only when its
holder releases, which a gateway handle does not do before shutdown. Counting it made ONE inode replacement
disable auto-VACUUM for that path for the rest of the process lifetime, turning the unbounded growth this
maintenance exists to bound into a permanent condition.
"""

from __future__ import annotations

import shutil

from persistence.sqlite import SQLiteBackend
from persistence.sqlite import _registry as registry
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session
from session_retention import run_auto_prune_and_vacuum
from tests.contracts._scopes import MEDIA_ROOT


def _seed(path):
    db = StoreHolder(db_path=path)
    create_session(db, "old", "cli")
    append_message(db, "old", "user", "hello")
    end_session(db, "old", "done")
    db.close()


def _auto_maintenance(store, admin):
    # retention_days=0 makes the ended row prunable; the freelist floor is disabled so only a
    # holder can stop the VACUUM.
    return run_auto_prune_and_vacuum(
        store, admin, store_home=store.holder().db_path.parent, media_root=MEDIA_ROOT,
        retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0,
        min_vacuum_freelist_ratio=-1.0)


def _make_prunable(db, session_id):
    db.set_meta("last_auto_prune", "0")
    create_session(db, session_id, "cli")
    end_session(db, session_id, "done")


def test_auto_vacuum_is_not_starved_by_a_retired_write_fenced_generation(tmp_path, scope_a):
    """One inode replacement must not stop auto-VACUUM for the process lifetime.

    The retired generation is never released (a gateway holds its store until shutdown), so a gate that
    counted it would report ``vacuum_skipped_holders`` on every later round and the store would grow
    without bound.
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
            backend = SQLiteBackend.lending(scope_a.hermes_home_key, current)
            store = backend.open(scope_a)
            admin = backend.open_store_admin(scope_a, read_only=False)
            for round_index in range(3):  # consecutive rounds, fresh prunable rows each time
                _make_prunable(current, f"old-{round_index}")
                result = _auto_maintenance(store, admin)
                assert result["pruned"] >= 1, (round_index, result)
                assert result["vacuumed"] is True, (round_index, result)
                assert "vacuum_skipped_holders" not in result, (round_index, result)
        finally:
            registry.release(current)
    finally:
        registry.release(parked)
