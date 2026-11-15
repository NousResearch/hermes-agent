"""Lease-refresh patience vs. lock storms (RCA 2026-09-28, skeptic CONCERN 2).

``refresh_session_turn_lease`` used to write with the routine 20s budget. During a
storm-class hold the refresh write fails, the lease row lapses (TTL 300s), and a
rival same-session writer can legally take it over — killing the turn via
``turn_lease`` instead of ``locked``. The fix: the refresh write rides the
storm-class transcript budget, so a still-owned row keeps being renewed as soon
as ANY write window opens, and never becomes reclaimable behind contention.
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


def _hold_write_lock(db_path, hold_s, started_evt):
    conn = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started_evt.set()
        time.sleep(hold_s)
        conn.execute("COMMIT")
    finally:
        conn.close()


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestLeaseRefreshStormPatience:
    def test_refresh_survives_a_hold_longer_than_the_routine_budget(self, db, monkeypatch):
        """A hold longer than the routine 20s budget (scaled: 0.4s) but inside the
        storm budget (scaled: 2.4s) must not defeat the refresh — pre-fix this
        raised and let the row lapse toward takeover."""
        db.create_session("s1", "cli")
        assert db.acquire_session_turn_lease("s1", "pid=1:turn") or db.try_acquire_session_turn_lease(
            "s1", "pid=1:turn")
        monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.4)
        monkeypatch.setattr(SessionDB, "_TRANSCRIPT_WRITE_PATIENCE_S", 2.4)

        started = threading.Event()
        holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 1.2, started))
        holder.start()
        try:
            assert started.wait(5.0)
            refreshed = db.refresh_session_turn_lease("s1", "pid=1:turn")
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        assert refreshed is True

    def test_refresh_still_fails_when_the_storm_budget_itself_exhausts(self, db, monkeypatch):
        """Polarity: storm patience is tolerance, not immortality — a hold longer
        than the storm budget still raises the classified locked-error (loud
        failure for genuinely wedged stores stays intact)."""
        db.create_session("s1", "cli")
        db.try_acquire_session_turn_lease("s1", "pid=1:turn")
        monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.4)
        monkeypatch.setattr(SessionDB, "_TRANSCRIPT_WRITE_PATIENCE_S", 0.4)

        started = threading.Event()
        holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 2.0, started))
        holder.start()
        try:
            assert started.wait(5.0)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.refresh_session_turn_lease("s1", "pid=1:turn")
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        assert "another Hermes process" in str(excinfo.value)
