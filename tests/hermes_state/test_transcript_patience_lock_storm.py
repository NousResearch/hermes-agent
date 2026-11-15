"""Transcript patience vs. the 2026-09-28 lock-storm class (RCA-state-db-lock-child-death).

Incident receipt (agent.log, 2026-09-28): a single turn-start write txn
(``SessionSessionsMixin.update_system_prompt``) held the shared state.db write lock
276.3s under GIL starvation, and four turns died with
``session_persistence_failed:locked`` when their transcript flush patience (60s)
ran out ~156s before the holder let go. The turns died only because the victim's
budget was shorter than the storm.

These tests pin the victim-side tolerance the 2026-09-14 contention plan promised
("ett skrivlås aldrig ska kunna döda en annan sessions kärnfunktion") for the
transcript path: the budget must (a) cover the observed storm class, and (b) stay
safely under the turn-lease TTL so a patient flush can never outlive its own lease
and trade one kill for another.
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _hold_write_lock(db_path, hold_s, started_evt):
    conn = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started_evt.set()
        time.sleep(hold_s)
        conn.execute("COMMIT")
    finally:
        conn.close()


class TestTranscriptPatienceStormClass:
    def test_patience_floor_covers_the_observed_storm_class(self, db):
        """60s lost today's race (hold 276.3s, victims died ~156s early). The
        budget must cover holds of the storm class we have actually measured
        (8.2s / 9.3s / 15.2s / 276.3s in the 24h before this fix)."""
        assert db._TRANSCRIPT_WRITE_PATIENCE_S >= 240.0

    def test_patience_stays_under_the_turn_lease_ttl(self, db):
        """A flush that waits longer than LEASE_TTL_SECONDS (300s) would survive
        the lock only to find its turn lease expired — trading the
        persistence-kill for a lease-kill. Keep a full patience window plus
        headroom for the turn's remaining work inside the lease: patience must
        stay <= 270s (300s lease - 30s headroom)."""
        from agent.turn_facade_lease import LEASE_TTL_SECONDS

        assert LEASE_TTL_SECONDS == 300.0
        assert db._TRANSCRIPT_WRITE_PATIENCE_S <= 270.0

    def test_transcript_append_survives_hold_longer_than_the_legacy_budget(self, db, monkeypatch):
        """Scaled survival proof of the mechanism this fix relies on: when the
        holder lets go inside the (new) patience window, the append succeeds —
        even for holds longer than the legacy 60s budget would have survived
        when scaled. Scaled per the contention plan's step-0b rule: same boundary
        logic, second-scale values, no minute-long CI runs."""
        db.create_session("s1", "cli")
        # Scale: legacy 60s -> 0.6s would die on a 1.2s hold; the raised budget
        # (240s -> 2.4s scaled) must ride it out.
        monkeypatch.setattr(SessionDB, "_TRANSCRIPT_WRITE_PATIENCE_S", 2.4)

        started = threading.Event()
        holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 1.2, started))
        holder.start()
        try:
            assert started.wait(5.0)
            msg_id = db.append_message(session_id="s1", role="user", content="storm survivor")
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        assert isinstance(msg_id, int)
        msgs = db.get_messages("s1")
        assert any(m["content"] == "storm survivor" for m in msgs)

    def test_transcript_exhausted_patience_still_fails_with_the_named_cause(self, db, monkeypatch):
        """Polarity: patience is tolerance, not immortality. A hold longer than
        the budget must still fail with the explanatory locked-class error
        (classify_persistence_error -> 'locked'), preserving the loud-failure
        contract for genuinely wedged stores."""
        from hermes_state_errors import classify_persistence_error

        db.create_session("s1", "cli")
        monkeypatch.setattr(SessionDB, "_TRANSCRIPT_WRITE_PATIENCE_S", 0.4)

        started = threading.Event()
        holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 2.0, started))
        holder.start()
        try:
            assert started.wait(5.0)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.append_message(session_id="s1", role="user", content="doomed")
        finally:
            holder.join(timeout=10.0)
        assert not holder.is_alive()
        assert "another Hermes process" in str(excinfo.value)
        assert classify_persistence_error(excinfo.value) == "locked"
