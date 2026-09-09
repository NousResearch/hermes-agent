"""Regression tests: a stale non-automatic end stamp on a still-routed session must not
permanently wedge compression (#106459).

Failure shape from the field (support-confirmed agent.log): a gateway-driven session row
carried a non-automatic ``ended_at`` stamp. Compression summarized fine (~60s), then
publish refused with "Compression parent already ended", the oversized history stayed,
and every later turn died with "Context length exceeded ... Cannot compress further" —
manual /compress no-ops through the same path, so the session was unrecoverable.

The heal clears the stamp only when the boundary is contradicted by observable liveness:
no continuation child was published from it and the row still received conversation
traffic inside the heal window. Quiet or forked deliberate boundaries still fail closed.
"""

from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB
from hermes_state_errors import CompressionSessionBusyError

_OLD_ACTIVITY_AGE = 900.0 * 4


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _seed_lock(db: SessionDB, session_id: str, holder: str = "compressor", *, expired: bool = False) -> None:
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (session_id, holder, now, (now - 10.0) if expired else (now + 300.0)),
    )
    db._conn.commit()


def _ended_parent(
    db: SessionDB,
    session_id: str = "parent",
    end_reason: str = "session_switch",
    *,
    activity_age: float = 5.0,
) -> None:
    """Row stamped ended with *end_reason*; its freshest observable activity is *activity_age*
    seconds old (both ``last_activity_at`` and the message timestamp are backdated so the
    freshest-of expression is deterministic)."""
    now = time.time()
    stale_ts = now - activity_age
    db.create_session(session_id, source="test")
    db.append_message(session_id, "user", "before split", timestamp=stale_ts)
    db.end_session(session_id, end_reason)
    db._conn.execute("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (stale_ts, session_id))
    db._conn.commit()


def _publish(db: SessionDB, *, parent: str = "parent", holder: str = "compressor") -> None:
    _seed_lock(db, parent, holder)
    db.publish_compression_child(
        parent_session_id=parent,
        child_session_id=f"{parent}-child",
        source="test",
        messages=[{"role": "user", "content": "compressed summary"}],
        compression_lock_holder=holder,
        require_compression_lease=True,
    )


class TestStaleEndStampHealOnPublish:
    def test_stale_nonautomatic_stamp_with_live_traffic_publishes(self, db: SessionDB) -> None:
        """#106459: the router still drives the row (fresh traffic), nobody forked from the
        boundary, and this writer holds the live lease — the stamp is stale; publish heals
        and completes instead of discarding the compression work."""
        _ended_parent(db, activity_age=5.0)

        _publish(db)

        child = db.get_session("parent-child")
        assert child is not None
        assert child["parent_session_id"] == "parent"
        parent = db.get_session("parent")
        assert parent["end_reason"] == "compression"
        assert parent["ended_at"] is not None

    def test_quiet_deliberate_boundary_still_fails_closed(self, db: SessionDB) -> None:
        """A deliberate boundary nobody acted on since (no traffic inside the heal window)
        must keep refusing publication."""
        _ended_parent(db, activity_age=_OLD_ACTIVITY_AGE)

        with pytest.raises(RuntimeError, match="Compression parent already ended"):
            _publish(db)

        parent = db.get_session("parent")
        assert parent["end_reason"] == "session_switch"
        assert db.get_session("parent-child") is None

    def test_forked_deliberate_boundary_still_fails_closed(self, db: SessionDB) -> None:
        """A continuation child hanging off the boundary means another path owns the
        lineage — recent traffic alone must not widen the heal."""
        _ended_parent(db, activity_age=5.0)
        db.create_session("adopted", source="test", parent_session_id="parent")

        with pytest.raises(RuntimeError, match="Compression parent already ended"):
            _publish(db)

        assert db.get_session("parent")["end_reason"] == "session_switch"

    def test_automatic_stamp_heal_unchanged(self, db: SessionDB) -> None:
        """The #88197 automatic-stamp heal keeps working: an automatic stamp on a live-
        lease rotation clears without the traffic/child guards."""
        _ended_parent(db, end_reason="ws_disconnect", activity_age=_OLD_ACTIVITY_AGE)

        _publish(db)

        assert db.get_session("parent-child") is not None
        assert db.get_session("parent")["end_reason"] == "compression"

    def test_lost_lease_still_fails_busy(self, db: SessionDB) -> None:
        """The heal must not bypass the lease gate: a stale stamp with a foreign/expired
        lease still raises CompressionSessionBusyError before any stamp is touched."""
        _ended_parent(db, activity_age=5.0)
        _seed_lock(db, "parent", holder="someone-else")

        with pytest.raises(CompressionSessionBusyError):
            db.publish_compression_child(
                parent_session_id="parent",
                child_session_id="parent-child",
                source="test",
                messages=[{"role": "user", "content": "compressed summary"}],
                compression_lock_holder="compressor",
                require_compression_lease=True,
            )

        assert db.get_session("parent")["end_reason"] == "session_switch"


class TestReadPredicate:
    def test_contradicted_by_live_traffic_true(self, db: SessionDB) -> None:
        _ended_parent(db, activity_age=5.0)
        assert db.end_stamp_contradicted_by_live_traffic("parent") is True

    def test_contradicted_by_live_traffic_false_when_quiet(self, db: SessionDB) -> None:
        _ended_parent(db, activity_age=_OLD_ACTIVITY_AGE)
        assert db.end_stamp_contradicted_by_live_traffic("parent") is False

    def test_contradicted_by_live_traffic_false_when_forked(self, db: SessionDB) -> None:
        _ended_parent(db, activity_age=5.0)
        db.create_session("adopted", source="test", parent_session_id="parent")
        assert db.end_stamp_contradicted_by_live_traffic("parent") is False

    def test_contradicted_by_live_traffic_false_for_automatic_and_live_rows(self, db: SessionDB) -> None:
        """The predicate targets non-automatic stamps only; live rows and automatic stamps
        take their existing paths."""
        assert db.end_stamp_contradicted_by_live_traffic("missing") is False
        db.create_session("live", source="test")
        assert db.end_stamp_contradicted_by_live_traffic("live") is False
        _ended_parent(db, end_reason="ws_disconnect", activity_age=5.0)
        assert db.end_stamp_contradicted_by_live_traffic("parent") is False

    def test_custom_window_narrows_the_predicate(self, db: SessionDB) -> None:
        _ended_parent(db, activity_age=600.0)
        assert db.end_stamp_contradicted_by_live_traffic("parent", 900.0) is True
        assert db.end_stamp_contradicted_by_live_traffic("parent", 60.0) is False
