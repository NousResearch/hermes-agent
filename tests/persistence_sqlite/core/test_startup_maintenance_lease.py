"""Construction-time state.db maintenance must keep the startup watchdog's progress lease alive (#111092).

Archive, prune, orphan sweep and VACUUM are I/O-bound with near-zero CPU, so the watchdog's
CPU fallback cannot tell them from a parked deadlock; only a phase lease can. Leases are
clamped to ``_MAX_LEASE_S`` per call, so each long step renews rather than one lease at entry.
"""

from __future__ import annotations

import time

import hermes_startup_watchdog as sw
from persistence.housekeeping.maintenance import PruneResult
from persistence.sqlite import _registry
from persistence.sqlite.admin import SQLiteStoreAdmin
from persistence.sqlite.housekeeping.maintenance import SQLiteSessionMaintenance
from session_retention import run_auto_archive, run_auto_prune_and_vacuum
from tests.persistence_sqlite.core._lent_store import lent_store

MEDIA_ROOT = "/nonexistent/native-inputs"


def test_maintenance_steps_renew_the_armed_watchdog_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sw._reset_for_tests()
    handle = sw.arm_startup_watchdog(timeout_s=300.0)
    db = _registry.acquire(tmp_path / "state.db")  # the admin's writer, so this handle never blocks VACUUM
    try:
        store, admin = lent_store(db), SQLiteStoreAdmin(db.db_path, read_only=False)
        # Maintenance is throttled by state_meta; a fresh DB has never run it.
        # VACUUM only runs after a real prune
        monkeypatch.setattr(SQLiteSessionMaintenance, "prune_sessions", lambda self, **kw: PruneResult(count=120))
        monkeypatch.setattr(SQLiteSessionMaintenance, "sweep_orphaned_sessions", lambda self, **kw: [])
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 1.0)
        monkeypatch.setattr(admin, "vacuum", lambda: None)
        monkeypatch.setattr(SQLiteSessionMaintenance, "list_idle_archive_candidates", lambda self, *a, **kw: ())
        before = handle._lease_count

        run_auto_archive(store, idle_days=3, min_interval_hours=0)
        run_auto_prune_and_vacuum(
            store, admin, store_home=tmp_path, media_root=MEDIA_ROOT,
            retention_days=90, min_interval_hours=0, vacuum=True, min_vacuum_interval_days=0,
        )

        # One renewal per long step (archive, prune, sweep, vacuum), the last one
        # taken right before VACUUM so a multi-minute rewrite never outlives the clamp.
        assert handle._lease_count - before == 4
        assert handle._lease_phase == "state_db_auto_vacuum"
        assert handle._lease_until > time.monotonic()
    finally:
        _registry.release(db)
        sw._reset_for_tests()
