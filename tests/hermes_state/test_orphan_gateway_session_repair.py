"""Repair of gateway sessions that lost their routing identity (#82616).

Incident shape (production, Aug 2026): a state.db write-path failure left the
live Telegram DM in a session row that never received its identity columns.
That row is invisible to ``find_latest_gateway_session_for_peer`` — both of
its queries require the very columns the row lacks — so after a gateway
restart the chat resolved to a keyed sibling three days older and the
conversation time-travelled. The real messages were never lost, only
unreachable.

These tests cover the offline repair path: detection must name the
predecessor only when the evidence is unambiguous, and adoption must make
the repaired row win recovery from then on.
"""

import time

import pytest

from hermes_state import SessionDB


PEER = {
    "source": "telegram",
    "user_id": "6308981865",
    "session_key": "agent:main:telegram:dm:6308981865",
    "chat_id": "6308981865",
    "chat_type": "dm",
}


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    yield store
    store.close()


def _mk_session(
    db,
    session_id,
    *,
    keyed=True,
    source="telegram",
    user_id=PEER["user_id"],
    started_at=None,
    last_activity_at=None,
    ended_at=None,
    end_reason=None,
    parent_session_id=None,
    model_config=None,
    messages=0,
):
    kwargs = {"user_id": user_id}
    if keyed:
        kwargs.update(
            session_key=PEER["session_key"],
            chat_id=PEER["chat_id"],
            chat_type=PEER["chat_type"],
        )
    if parent_session_id:
        kwargs["parent_session_id"] = parent_session_id
    if model_config:
        kwargs["model_config"] = model_config
    db.create_session(session_id, source, **kwargs)
    for i in range(messages):
        db.append_message(
            session_id, "user" if i % 2 == 0 else "assistant", f"m{i}"
        )
    with db._lock:
        if keyed:
            # ``create_session`` on main does not carry presentation/origin
            # metadata into the INSERT — set it the way the gateway's
            # per-turn peer refresh does, so the donor row is realistic.
            db._conn.execute(
                "UPDATE sessions SET origin_json = ?, display_name = ? "
                "WHERE id = ?",
                (
                    '{"platform": "telegram", "chat_id": "6308981865"}',
                    "Teknium",
                    session_id,
                ),
            )
        if started_at is not None:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (started_at, session_id),
            )
        if messages:
            # Message timestamps feed the recency expression; pin them to the
            # session window so tests control contiguity deterministically.
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (last_activity_at or started_at, session_id),
            )
        db._conn.execute(
            "UPDATE sessions SET last_activity_at = ?, ended_at = ?, "
            "end_reason = ? WHERE id = ?",
            (last_activity_at, ended_at, end_reason, session_id),
        )
        db._conn.commit()


def _incident(db, *, parent_link=False):
    """Build the reported incident: keyed stale row + unkeyed live row."""
    now = time.time()
    stale_last = now - 3 * 86400
    _mk_session(
        db,
        "20260803_120103_37976afb",
        keyed=True,
        started_at=now - 6 * 86400,
        last_activity_at=stale_last,
        end_reason="agent_close",
        messages=4,
    )
    _mk_session(
        db,
        "20260806_161836_04c1d6c6",
        keyed=False,
        started_at=stale_last + 60,
        last_activity_at=now - 3600,
        parent_session_id=(
            "20260803_120103_37976afb" if parent_link else None
        ),
        messages=6,
    )
    return "20260803_120103_37976afb", "20260806_161836_04c1d6c6"


def _last_activity(db, session_id):
    with db._lock:
        row = db._conn.execute(
            "SELECT last_activity_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row["last_activity_at"]


class TestDetection:
    def test_incident_orphan_is_adoptable_by_contiguity(self, db):
        stale, orphan = _incident(db)
        records = db.find_orphaned_gateway_sessions()
        assert len(records) == 1
        record = records[0]
        assert record["orphan_id"] == orphan
        assert record["donor_id"] == stale
        assert record["session_key"] == PEER["session_key"]
        assert record["evidence"] == "contiguity"
        assert record["adoptable"] is True
        assert record["message_count"] == 6

    def test_parent_link_is_used_without_a_time_window(self, db):
        stale, orphan = _incident(db, parent_link=True)
        # Push the predecessor far outside the contiguity window: a recorded
        # lineage is a fact, not a guess, so it must still resolve.
        long_ago = time.time() - 400 * 86400
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                (long_ago, stale),
            )
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (long_ago, stale),
            )
            db._conn.commit()
        record = db.find_orphaned_gateway_sessions()[0]
        assert record["orphan_id"] == orphan
        assert record["donor_id"] == stale
        assert record["evidence"] == "lineage"
        assert record["adoptable"] is True

    def test_sessions_without_messages_are_not_reported(self, db):
        _mk_session(db, "empty", keyed=False, started_at=time.time())
        assert db.find_orphaned_gateway_sessions() == []

    def test_two_predecessors_in_the_window_fail_closed(self, db):
        stale, _ = _incident(db)
        quiet_at = _last_activity(db, stale)
        # A second chat on the same platform fell quiet at the same moment.
        _mk_session(
            db,
            "20260803_120104_other",
            keyed=True,
            user_id=None,
            started_at=quiet_at - 3600,
            last_activity_at=quiet_at,
            end_reason="agent_close",
            messages=2,
        )
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET session_key = ? WHERE id = ?",
                ("agent:main:telegram:dm:999", "20260803_120104_other"),
            )
            db._conn.commit()
        record = db.find_orphaned_gateway_sessions()[0]
        assert record["adoptable"] is False
        assert "ambiguous" in record["reason"]

    def test_two_orphans_claiming_one_predecessor_fail_closed(self, db):
        stale, _ = _incident(db)
        _mk_session(
            db,
            "20260806_161840_second",
            keyed=False,
            started_at=_last_activity(db, stale) + 90,
            last_activity_at=time.time() - 3600,
            messages=3,
        )
        records = db.find_orphaned_gateway_sessions()
        assert len(records) == 2
        assert not any(r["adoptable"] for r in records)
        assert all("ambiguous" in r["reason"] for r in records)

    def test_delegate_children_are_not_orphans(self, db):
        stale, orphan = _incident(db)
        with db._lock:
            db._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (orphan,)
            )
            db._conn.execute("DELETE FROM sessions WHERE id = ?", (orphan,))
            db._conn.commit()
        _mk_session(
            db,
            "20260806_161836_delegate",
            keyed=False,
            started_at=_last_activity(db, stale) + 60,
            last_activity_at=time.time() - 3600,
            model_config={"_delegate_from": stale},
            messages=3,
        )
        assert db.find_orphaned_gateway_sessions() == []

    def test_a_predecessor_of_another_platform_is_not_a_donor(self, db):
        stale, _ = _incident(db)
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET source = 'discord' WHERE id = ?", (stale,)
            )
            db._conn.commit()
        record = db.find_orphaned_gateway_sessions()[0]
        assert record["adoptable"] is False
        assert record["donor_id"] is None


class TestAdoption:
    def _resolve(self, db):
        return db.find_latest_gateway_session_for_peer(
            source=PEER["source"],
            user_id=PEER["user_id"],
            session_key=PEER["session_key"],
            chat_id=PEER["chat_id"],
            chat_type=PEER["chat_type"],
        )

    def test_adoption_moves_recovery_to_the_live_conversation(self, db):
        stale, orphan = _incident(db)
        # Before: recovery hands the chat to the three-day-old row.
        assert self._resolve(db)["id"] == stale

        assert db.adopt_orphaned_gateway_session(orphan, stale) is True

        after = self._resolve(db)
        assert after["id"] == orphan
        assert after["chat_id"] == PEER["chat_id"]
        assert after["origin_json"]
        assert after["parent_session_id"] == stale

    def test_predecessor_is_retired_under_a_non_resumable_reason(self, db):
        stale, orphan = _incident(db)
        db.adopt_orphaned_gateway_session(orphan, stale)
        row = db.get_session(stale)
        assert row["end_reason"] == "superseded_by_repair"
        assert row["ended_at"] is not None

    def test_adoption_is_idempotent(self, db):
        stale, orphan = _incident(db)
        assert db.adopt_orphaned_gateway_session(orphan, stale) is True
        # The orphan is keyed now, so a replay must not re-retire anything.
        assert db.adopt_orphaned_gateway_session(orphan, stale) is False
        assert db.find_orphaned_gateway_sessions() == []

    def test_existing_columns_are_never_overwritten(self, db):
        stale, orphan = _incident(db)
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET display_name = ? WHERE id = ?",
                ("Renamed", orphan),
            )
            db._conn.commit()
        db.adopt_orphaned_gateway_session(orphan, stale)
        assert db.get_session(orphan)["display_name"] == "Renamed"

    def test_cross_source_adoption_is_refused(self, db):
        stale, orphan = _incident(db)
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET source = 'discord' WHERE id = ?", (orphan,)
            )
            db._conn.commit()
        assert db.adopt_orphaned_gateway_session(orphan, stale) is False
        assert db.get_session(orphan)["session_key"] is None

    def test_unkeyed_donor_is_refused(self, db):
        _, orphan = _incident(db)
        _mk_session(
            db, "no_key_donor", keyed=False, started_at=time.time() - 100
        )
        assert db.adopt_orphaned_gateway_session(orphan, "no_key_donor") is False
