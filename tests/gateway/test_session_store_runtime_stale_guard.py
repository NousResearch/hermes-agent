"""Runtime self-heal for stale sessions.json routing entries (#54878).

`_prune_stale_sessions_locked` only runs at gateway startup. A session ended
in state.db while the gateway stays alive (e.g. any path that finalizes the
row without clearing sessions.json) leaves a stale `session_key -> session_id`
mapping whose session has `end_reason` set. Before this fix,
`get_or_create_session` returned that stale entry as a live routing key (it
never consulted end_reason), so every subsequent message was silently routed
into a closed session and dropped — no log, no error, no response — until the
next restart pruned it.

This is the live-gateway variant of #52804/FM9 (#52808/#54138 startup prune),
which required an actual gateway *crash*. Here the guard inside
`get_or_create_session` detects the ended row at routing time and drops the
stale entry, falling through to `_recover_session_from_db` (which reopens
`agent_close`-ended rows and resumes the SAME session_id, preserving the
transcript) or, failing recovery, to a fresh session.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(key: str, session_id: str, **kw) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        **kw,
    )


def _db_returning(rows: dict) -> MagicMock:
    """SessionDB mock where get_session maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    # By default recovery finds nothing (forces a fresh session).
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    # No compression continuation → the tip is the session itself (identity),
    # mirroring the real SessionDB.get_compression_tip. Without this a bare Mock
    # would return a Mock the routing heal then assigns as session_id.
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _make_store_with_db(tmp_path, db_mock) -> SessionStore:
    """Build a SessionStore with a mock SessionDB, bypassing disk load."""
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _source() -> SessionSource:
    # session_key for this peer is deterministic; matches the entry key we seed.
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8494508720",
        chat_type="dm",
        user_id="8494508720",
    )


# ---------------------------------------------------------------------------
# _is_session_ended_in_db helper
# ---------------------------------------------------------------------------

class TestIsSessionEndedInDb:
    def test_ended_row_is_stale(self, tmp_path):
        db = _db_returning({"sid": {"end_reason": "agent_close", "id": "sid"}})
        store = _make_store_with_db(tmp_path, db)
        assert store._is_session_ended_in_db("sid") is True

    def test_alive_row_not_stale(self, tmp_path):
        db = _db_returning({"sid": {"end_reason": None, "id": "sid"}})
        store = _make_store_with_db(tmp_path, db)
        assert store._is_session_ended_in_db("sid") is False

    def test_absent_row_not_stale(self, tmp_path):
        # Not yet persisted / legacy — must NOT be treated as ended, else a
        # freshly-created in-memory session would be wrongly discarded.
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        assert store._is_session_ended_in_db("sid_absent") is False

    def test_no_db_not_stale(self, tmp_path):
        store = _make_store_with_db(tmp_path, _db_returning({}))
        store._db = None
        assert store._is_session_ended_in_db("sid") is False

    def test_empty_session_id_not_stale(self, tmp_path):
        store = _make_store_with_db(tmp_path, _db_returning({}))
        assert store._is_session_ended_in_db("") is False

    def test_db_error_not_stale(self, tmp_path):
        db = MagicMock()
        db.get_session.side_effect = Exception("DB locked")
        store = _make_store_with_db(tmp_path, db)
        # On error, never block routing — treat as not-stale (keep).
        assert store._is_session_ended_in_db("sid") is False


# ---------------------------------------------------------------------------
# get_or_create_session — runtime self-heal
# ---------------------------------------------------------------------------

class TestRuntimeStaleGuard:
    def test_stale_agent_close_entry_recovered_preserving_session_id(self, tmp_path):
        """Stale `agent_close` entry → recovery reopens the SAME session_id."""
        source = _source()
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        # Recovery finds the agent_close row and reopens it (transcript-preserving).
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_stale",
            "started_at": (datetime.now() - timedelta(hours=2)).timestamp(),
        }
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_stale")

        result = store.get_or_create_session(source)

        # SAME session_id (resumed), not a brand-new one, and not silently
        # routed into the closed entry.
        assert result.session_id == "sid_stale"
        db.reopen_session.assert_called_once_with("sid_stale")
        # A brand-new session row must NOT have been created.
        db.create_session.assert_not_called()

    def test_stale_entry_creates_fresh_when_recovery_returns_none(self, tmp_path):
        """Stale entry, no recoverable row → brand-new session (no silent drop)."""
        source = _source()
        # Ended with a non-recoverable reason (e.g. /new) → finder returns None.
        db = _db_returning({"sid_stale": {"end_reason": "new_command", "id": "sid_stale"}})
        db.find_latest_gateway_session_for_peer.return_value = None
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_stale")

        result = store.get_or_create_session(source)

        assert result.session_id != "sid_stale"
        # A fresh session row was created for the new session_id.
        db.create_session.assert_called_once()
        assert store._entries[key].session_id == result.session_id

    def test_live_entry_returned_unchanged(self, tmp_path):
        """A session still alive in the DB is returned as-is (no churn)."""
        source = _source()
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_live")

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_live"
        db.find_latest_gateway_session_for_peer.assert_not_called()
        db.create_session.assert_not_called()

    def test_stale_check_wins_over_suspended(self, tmp_path):
        """A stale entry that is ALSO suspended is still dropped via the stale
        path — we must not consult the dead entry's reset/suspend state."""
        source = _source()
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        db.find_latest_gateway_session_for_peer.return_value = None  # → fresh
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_stale", suspended=True)

        result = store.get_or_create_session(source)

        # Did not return the stale (suspended) entry; created a fresh session.
        assert result.session_id != "sid_stale"
        db.create_session.assert_called_once()

    def test_force_new_skips_stale_check(self, tmp_path):
        """force_new short-circuits the whole existing-entry branch; the stale
        DB lookup must not even run."""
        source = _source()
        db = _db_returning({"sid_old": {"end_reason": "agent_close", "id": "sid_old"}})
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_old")

        result = store.get_or_create_session(source, force_new=True)

        assert result.session_id != "sid_old"
        db.get_session.assert_not_called()


# ---------------------------------------------------------------------------
# _is_routing_entry_purged helper (#59512)
# ---------------------------------------------------------------------------

class TestIsRoutingEntryPurged:
    def test_missing_routing_row_is_purged(self, tmp_path):
        db = _db_returning({})
        db.gateway_routing_entry_exists.return_value = False
        store = _make_store_with_db(tmp_path, db)
        assert store._is_routing_entry_purged("some_key") is True

    def test_present_routing_row_not_purged(self, tmp_path):
        db = _db_returning({})
        db.gateway_routing_entry_exists.return_value = True
        store = _make_store_with_db(tmp_path, db)
        assert store._is_routing_entry_purged("some_key") is False

    def test_scoped_to_store_routing_scope(self, tmp_path):
        db = _db_returning({})
        db.gateway_routing_entry_exists.return_value = True
        store = _make_store_with_db(tmp_path, db)

        store._is_routing_entry_purged("some_key")

        db.gateway_routing_entry_exists.assert_called_once_with(
            "some_key", scope=store._routing_scope()
        )

    def test_no_db_not_purged(self, tmp_path):
        store = _make_store_with_db(tmp_path, MagicMock())
        store._db = None
        assert store._is_routing_entry_purged("some_key") is False

    def test_empty_key_not_purged(self, tmp_path):
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        assert store._is_routing_entry_purged("") is False

    def test_db_missing_checker_not_purged(self, tmp_path):
        """A SessionDB without gateway_routing_entry_exists (older schema /
        partial test double) must not be treated as "everything purged"."""
        db = MagicMock(spec=["get_session"])
        store = _make_store_with_db(tmp_path, db)
        assert store._is_routing_entry_purged("some_key") is False

    def test_db_error_not_purged(self, tmp_path):
        db = _db_returning({})
        db.gateway_routing_entry_exists.side_effect = RuntimeError("boom")
        store = _make_store_with_db(tmp_path, db)
        assert store._is_routing_entry_purged("some_key") is False


# ---------------------------------------------------------------------------
# Runtime self-heal for a routing entry purged by a concurrent
# `hermes sessions prune` (#59512) — the "existing [live] store" half of the
# reviewer-requested coverage; test_prune_sessions_drops_stale_sessions_json_entries
# in tests/test_hermes_state.py covers the "restarted store" half.
# ---------------------------------------------------------------------------

class TestRuntimePruneGuard:
    def test_purged_entry_creates_fresh_session(self, tmp_path):
        """session_id row is gone entirely (hard-deleted by a concurrent
        prune) — not merely ended. get_session returns None (not "ended"),
        so _is_session_ended_in_db alone would say "keep"; the routing-table
        presence check must catch it instead."""
        source = _source()
        db = _db_returning({})  # get_session(...) -> None for any id: hard-deleted
        db.gateway_routing_entry_exists.return_value = False  # purged from gateway_routing too
        db.find_latest_gateway_session_for_peer.return_value = None
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_pruned")

        result = store.get_or_create_session(source)

        assert result.session_id != "sid_pruned"
        db.create_session.assert_called_once()

    def test_live_entry_with_persisted_routing_row_unaffected(self, tmp_path):
        """A live, still-persisted entry must not be disturbed by the new
        check — this is the "not yet persisted" ambiguity the check exists
        to avoid misfiring on."""
        source = _source()
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        db.gateway_routing_entry_exists.return_value = True
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_live")

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_live"
        db.create_session.assert_not_called()

    def test_ended_in_db_takes_priority_and_skips_routing_lookup(self, tmp_path):
        """When end_reason already answers the question, the extra routing
        existence check is redundant — cheap short-circuit, not a
        correctness requirement, but pins the actual call order."""
        source = _source()
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        db.find_latest_gateway_session_for_peer.return_value = None
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_stale")

        result = store.get_or_create_session(source)

        assert result.session_id != "sid_stale"
        db.gateway_routing_entry_exists.assert_not_called()
