"""Regression tests for gateway session continuity (#82616).

Incident shape (Aug 2026, production): a /new reset's DB writes both failed
silently → the routing index moved to a new session id that had NO row →
the row was later lazily materialized with no identity columns → a gateway
crash pruned sessions.json → restart recovery resolved the chat to the
days-old zombie predecessor (still open, still keyed) → the user's DM
time-traveled three days back.

Four fixes under test:
  1. Identity lands atomically in the session INSERT (origin_json,
     display_name, parent_session_id in db_create_kwargs).
  2. record_gateway_session_peer self-heals a missing row (INSERT on
     absent id) so identity-less lazy writers can never be first.
  3. load_transcript follows the write-side reroute chain + durable
     compression tip; read exceptions are WARNING not silent [].
  4. find_latest_gateway_session_for_peer ranks by last_activity_at and
     returns an empty-but-keyed row instead of None.
"""

import json
import time
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


PEER = dict(
    source="telegram",
    user_id="6308981865",
    session_key="agent:main:telegram:dm:6308981865",
    chat_id="6308981865",
    chat_type="dm",
    thread_id=None,
)


def _mk_session(db, session_id, *, key=True, started_at=None, last_activity_at=None,
                ended_at=None, end_reason=None, msgs=0):
    kwargs = {"user_id": PEER["user_id"]}
    if key:
        kwargs.update(
            session_key=PEER["session_key"], chat_id=PEER["chat_id"],
            chat_type=PEER["chat_type"],
        )
    db.create_session(session_id, "telegram", **kwargs)
    for i in range(msgs):
        db.append_message(session_id, "user" if i % 2 == 0 else "assistant", f"m{i}")
    with db._lock:
        if started_at is not None:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started_at, session_id))
        if last_activity_at is not None:
            db._conn.execute("UPDATE sessions SET last_activity_at=? WHERE id=?", (last_activity_at, session_id))
        if ended_at is not None:
            db._conn.execute(
                "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
                (ended_at, end_reason, session_id),
            )
        db._conn.commit()


class TestIdentityAtInsert:
    def test_insert_session_row_persists_origin_and_display_name(self, db):
        origin = json.dumps({"platform": "telegram", "chat_id": "6308981865"})
        db.create_session(
            "s1", "telegram",
            session_key=PEER["session_key"], chat_id=PEER["chat_id"],
            chat_type="dm", origin_json=origin, display_name="Teknium",
        )
        row = db.get_session("s1")
        assert row["origin_json"] == origin
        assert row["display_name"] == "Teknium"
        assert row["session_key"] == PEER["session_key"]

    def test_conflict_backfills_origin_without_overwriting(self, db):
        db.create_session("s2", "telegram", session_key=PEER["session_key"])
        # Second insert (agent-side upsert) backfills origin_json on the
        # existing row...
        db.create_session("s2", "telegram", origin_json='{"a":1}', display_name="X")
        row = db.get_session("s2")
        assert row["origin_json"] == '{"a":1}'
        # ...but never overwrites a value already set.
        db.create_session("s2", "telegram", origin_json='{"b":2}', display_name="Y")
        row = db.get_session("s2")
        assert row["origin_json"] == '{"a":1}'
        assert row["display_name"] == "X"


class TestPeerRecorderSelfHeal:
    def test_recorder_inserts_missing_row_with_full_identity(self, db):
        """The incident's core gap: routing points at a session id whose
        create_session write failed. The peer refresh must create the row
        with identity, not silently no-op."""
        sid = "20260806_161836_" + uuid.uuid4().hex[:8]
        assert db.get_session(sid) is None
        db.record_gateway_session_peer(
            sid,
            source="telegram",
            user_id=PEER["user_id"],
            session_key=PEER["session_key"],
            chat_id=PEER["chat_id"],
            chat_type="dm",
            display_name="Teknium",
            origin_json='{"platform": "telegram"}',
        )
        row = db.get_session(sid)
        assert row is not None, "peer refresh must self-heal the missing row"
        assert row["session_key"] == PEER["session_key"]
        assert row["chat_id"] == PEER["chat_id"]
        assert row["origin_json"] == '{"platform": "telegram"}'

    def test_lazy_writer_then_peer_refresh_repairs_identity(self, db):
        """Orphan-factory shape: update_token_counts materializes the row
        identity-less first; the next peer refresh must stamp it."""
        sid = "lazy_" + uuid.uuid4().hex[:8]
        db.update_token_counts(sid, input_tokens=10, output_tokens=5)
        row = db.get_session(sid)
        assert row is not None and row["session_key"] is None  # the orphan
        db.record_gateway_session_peer(
            sid,
            source="telegram",
            user_id=PEER["user_id"],
            session_key=PEER["session_key"],
            chat_id=PEER["chat_id"],
            chat_type="dm",
        )
        row = db.get_session(sid)
        assert row["session_key"] == PEER["session_key"]
        assert row["chat_id"] == PEER["chat_id"]


class TestPeerResolutionRecency:
    def test_prefers_recent_activity_over_started_at(self, db):
        """The Aug 9 misroute: zombie predecessor (older start, no recent
        activity, still open+keyed) must lose to the keyed row with newer
        activity."""
        now = time.time()
        _mk_session(db, "zombie", started_at=now - 6 * 86400,
                    last_activity_at=now - 3 * 86400, msgs=4)
        _mk_session(db, "live", started_at=now - 3 * 86400,
                    last_activity_at=now - 600, msgs=6)
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found is not None
        assert found["id"] == "live"

    def test_empty_keyed_row_returned_not_none(self, db):
        """Returning None mints a fresh id (worse than an empty resume);
        an empty keyed row must be returned."""
        _mk_session(db, "emptyrow", msgs=0)
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found is not None
        assert found["id"] == "emptyrow"

    def test_rows_with_messages_beat_empty_rows(self, db):
        now = time.time()
        _mk_session(db, "hascontent", last_activity_at=now - 86400, msgs=4)
        _mk_session(db, "emptynewer", last_activity_at=now - 60, msgs=0)
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found["id"] == "hascontent"

    def test_explicit_reset_rows_stay_unrecoverable(self, db):
        now = time.time()
        _mk_session(db, "resetold", last_activity_at=now - 60,
                    ended_at=now - 30, end_reason="session_reset", msgs=4)
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found is None, "/new boundaries must never be resurrected"

    def test_incident_shape_still_resolves_to_best_available(self, db):
        """Pre-fix production shape: zombie keyed+open, real session
        UNSTAMPED. The resolver can only see the zombie — proving stamping
        (fixes 1-2) is load-bearing; but with the real session stamped by
        the self-heal, the resolver must pick it."""
        now = time.time()
        _mk_session(db, "zombie2", started_at=now - 6 * 86400,
                    last_activity_at=now - 3 * 86400, msgs=4)
        _mk_session(db, "real", key=False, started_at=now - 3 * 86400,
                    last_activity_at=now - 600, msgs=8)
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found["id"] == "zombie2"  # unstamped rows are invisible
        # Self-heal stamps the real row (any later peer refresh):
        db.record_gateway_session_peer(
            "real", source="telegram", user_id=PEER["user_id"],
            session_key=PEER["session_key"], chat_id=PEER["chat_id"],
            chat_type="dm",
        )
        found = db.find_latest_gateway_session_for_peer(**PEER)
        assert found["id"] == "real"


class TestLoadTranscriptReroutes:
    def test_load_transcript_raises_when_message_read_fails(self, tmp_path, monkeypatch):
        from gateway.session import SessionStore
        from gateway.session_transcript import TranscriptReadError

        from gateway.config import GatewayConfig

        store = SessionStore(sessions_dir=tmp_path / "gw-failed-read", config=GatewayConfig())
        db = store._db
        assert db is not None
        monkeypatch.setattr(db, "get_compression_tip", lambda _session_id: None)

        def _malformed(_session_id, *, repair_alternation):
            assert repair_alternation is True
            raise RuntimeError("database disk image is malformed")

        monkeypatch.setattr(db, "get_messages_as_conversation", _malformed)

        with pytest.raises(TranscriptReadError) as exc_info:
            store.load_transcript("existing-session")

        assert exc_info.value.session_id == "existing-session"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_load_transcript_follows_reroute_chain(self, tmp_path):
        from gateway.session import SessionStore

        from gateway.config import GatewayConfig

        store = SessionStore(sessions_dir=tmp_path / "gw", config=GatewayConfig())
        db = store._db
        assert db is not None
        db.create_session("parent", "telegram", session_key=PEER["session_key"])
        db.create_session("child", "telegram", session_key=PEER["session_key"])
        db.append_message("child", "user", "hello from the child")
        # Write-side reroute installed after a compression rotation:
        store._transcript_reroutes["parent"] = "child"
        msgs = store.load_transcript("parent")
        assert any("hello from the child" in str(m.get("content", "")) for m in msgs), (
            "reads must follow the same reroute chain writes use"
        )

    def test_load_transcript_follows_durable_compression_tip(self, tmp_path):
        from gateway.session import SessionStore

        from gateway.config import GatewayConfig

        store = SessionStore(sessions_dir=tmp_path / "gw2", config=GatewayConfig())
        db = store._db
        assert db is not None
        db.create_session("p2", "telegram", session_key=PEER["session_key"])
        db.append_message("p2", "user", "old")
        db.publish_compression_child(
            parent_session_id="p2",
            child_session_id="c2",
            source="telegram",
            messages=[{"role": "user", "content": "compressed history"}],
            require_compression_lease=False,
        )
        # No in-memory reroute (fresh store after restart) — durable tip only.
        store._transcript_reroutes.clear()
        msgs = store.load_transcript("p2")
        assert any("compressed history" in str(m.get("content", "")) for m in msgs)
