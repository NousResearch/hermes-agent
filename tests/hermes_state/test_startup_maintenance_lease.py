"""Startup-watchdog progress leases across state.db auto-maintenance.

Regression for #111092: gateway restart live-lock on large installs. The construction-time
maintenance block (auto-archive, auto-prune, orphan sweep, VACUUM) is I/O-bound with
near-zero CPU, which the startup watchdog misreads as a parked deadlock and kills with
exit 75. Each long synchronous step must hold a ``report_startup_progress`` lease; leases
are clamped to 900 s per call, so the lease is renewed per step rather than once at
entry.

Invariant (not a change-detector): every long maintenance step reports a progress lease.
"""

from __future__ import annotations

from pathlib import Path

from hermes_state import SessionDB

import hermes_state_maintenance
import hermes_state_sessions


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def test_prune_sweep_and_vacuum_each_hold_a_lease(tmp_path, monkeypatch):
    leases: list[str] = []
    monkeypatch.setattr(
        hermes_state_maintenance,
        "report_startup_progress",
        lambda expected_s, phase="": leases.append(phase),
    )
    db = _make_db(tmp_path)
    try:
        # Simulate a large-DB sweep where every step does long I/O-bound work.
        monkeypatch.setattr(db, "prune_sessions", lambda **kw: 120)
        monkeypatch.setattr(db, "sweep_orphaned_sessions", lambda **kw: [])
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 1.0)
        monkeypatch.setattr(db, "vacuum", lambda: 0)

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=True,
            min_vacuum_interval_days=0,
        )
        assert result["pruned"] == 120
        assert result["vacuumed"] is True
        # A lease must be held across EACH long step: without them the watchdog
        # fires exit 75 mid-sweep on a large state.db (#111092).
        assert "state_db_auto_prune" in leases
        assert "state_db_auto_sweep" in leases
        assert "state_db_auto_vacuum" in leases
    finally:
        db.close()


def test_auto_archive_holds_a_lease(tmp_path, monkeypatch):
    leases: list[str] = []
    monkeypatch.setattr(
        hermes_state_sessions,
        "report_startup_progress",
        lambda expected_s, phase="": leases.append(phase),
    )
    db = _make_db(tmp_path)
    try:
        monkeypatch.setattr(db, "archive_stale_sessions", lambda *a, **kw: 7)
        result = db.maybe_auto_archive(idle_days=3, min_interval_hours=0)
        assert result["archived"] == 7
        assert "state_db_auto_archive" in leases
    finally:
        db.close()
