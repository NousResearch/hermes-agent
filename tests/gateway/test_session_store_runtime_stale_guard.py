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


# ---------------------------------------------------------------------------
# get_or_create_session — runtime self-heal
# ---------------------------------------------------------------------------

class TestRuntimeStaleGuard:

    def test_stale_ws_orphan_reap_entry_recovered_preserving_session_id(self, tmp_path):
        """Stale ``ws_orphan_reap`` entry → recovery reopens the SAME session_id (#63207)."""
        source = _source()
        db = _db_returning({"sid_stale": {"end_reason": "ws_orphan_reap", "id": "sid_stale"}})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_stale",
            "started_at": (datetime.now() - timedelta(hours=2)).timestamp(),
        }
        store = _make_store_with_db(tmp_path, db)
        key = store._generate_session_key(source)
        store._entries[key] = _make_entry(key, "sid_stale")

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_stale"
        db.reopen_session.assert_called_once_with("sid_stale")
        db.create_session.assert_not_called()


    def test_stale_agent_close_overdue_policy_creates_fresh_session(
        self, tmp_path,
    ):
        """Stale `agent_close` entry + overdue reset policy → fresh session.

        The #54878 self-healing path popped the stale sessions.json entry and
        recovered the same session_id from the DB without checking whether a
        daily/idle reset was actually due.  This test guards the fix at
        gateway/session.py:1765 — when the session is overdue under the
        configured reset policy, we must create a fresh session (new id,
        auto-reset metadata set, reopen_session NOT called).
        """
        source = _source()
        # Idle policy: reset after 60 minutes of inactivity.
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        # Recovery would normally reopen this row — but it shouldn't, because
        # the reset policy says this session is overdue.
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_stale",
            "started_at": (datetime.now() - timedelta(hours=3)).timestamp(),
        }

        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        key = store._generate_session_key(source)
        # Entry last updated 2 hours ago → well past the 60-minute idle window.
        store._entries[key] = _make_entry(key, "sid_stale")
        store._entries[key].updated_at = datetime.now() - timedelta(hours=2)

        result = store.get_or_create_session(source)

        # Fresh session — NOT the stale session_id.
        assert result.session_id != "sid_stale"
        # Auto-reset metadata is set.
        assert result.was_auto_reset is True
        assert result.auto_reset_reason == "idle"
        # reopen_session must NOT have been called (we skipped recovery).
        db.reopen_session.assert_not_called()
        # A brand-new session row was created.
        db.create_session.assert_called_once()


class TestRecoveredSessionResetPolicy:
    """Recovery must not resurrect sessions as freshly active.

    ``_create_entry_from_recovered_row`` used to stamp ``updated_at=now`` on
    the rebuilt entry, so an opt-in idle/daily ``session_reset`` policy could
    never fire across a gateway restart: the recovered session always looked
    freshly active, and every subsequent message bumped ``updated_at`` again
    — a recovered stale session could never age out. The entry now carries
    the durable ``last_activity_at`` the finder already returns on the row
    and the recovery paths evaluate ``_should_reset`` before resuming.
    """

    def test_recovered_entry_carries_durable_last_activity(self, tmp_path):
        """A recovered mapping reports the DB's last message time, not now()."""
        source = _source()
        started = (datetime.now() - timedelta(hours=3)).timestamp()
        last_activity = (datetime.now() - timedelta(hours=2)).timestamp()
        db = _db_returning({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recovered",
            "started_at": started,
            "last_activity_at": last_activity,
        }
        store = _make_store_with_db(tmp_path, db)  # default mode="none"

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_recovered"
        assert result.created_at == datetime.fromtimestamp(started)
        assert result.updated_at == datetime.fromtimestamp(last_activity)
        assert result.reset_had_activity is True

    def test_recovered_session_past_idle_policy_resets_instead_of_resuming(
        self, tmp_path,
    ):
        """Lost mapping + overdue recoverable row → reset, not silent resume."""
        source = _source()
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )
        db = _db_returning({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_idle",
            "started_at": (datetime.now() - timedelta(hours=3)).timestamp(),
            "last_activity_at": (
                datetime.now() - timedelta(hours=2)
            ).timestamp(),
        }
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True
        # No in-memory entry: the mapping was lost (e.g. crash before save).

        result = store.get_or_create_session(source)

        assert result.session_id != "sid_idle"
        assert result.was_auto_reset is True
        assert result.auto_reset_reason == "idle"
        assert result.reset_had_activity is True
        assert result.prev_session_id == "sid_idle"
        db.reopen_session.assert_not_called()
        db.promote_to_session_reset.assert_called_once_with("sid_idle", "idle")
        db.end_session.assert_not_called()
        db.create_session.assert_called_once()

    def test_default_none_policy_recovery_resumes_unchanged(self, tmp_path):
        """mode="none" (the default) still resumes recoverable rows as before."""
        source = _source()
        db = _db_returning({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recovered",
            "started_at": (datetime.now() - timedelta(days=30)).timestamp(),
            "last_activity_at": (
                datetime.now() - timedelta(days=30)
            ).timestamp(),
        }
        store = _make_store_with_db(tmp_path, db)  # default mode="none"

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_recovered"
        db.reopen_session.assert_called_once_with("sid_recovered")
        db.promote_to_session_reset.assert_not_called()
        db.end_session.assert_not_called()
        db.create_session.assert_not_called()

    def test_recovery_tolerates_row_without_last_activity(self, tmp_path):
        """A row lacking last_activity_at falls back to created_at."""
        source = _source()
        started = (datetime.now() - timedelta(hours=3)).timestamp()
        db = _db_returning({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recovered",
            "started_at": started,
        }
        store = _make_store_with_db(tmp_path, db)

        result = store.get_or_create_session(source)

        assert result.session_id == "sid_recovered"
        assert result.updated_at == datetime.fromtimestamp(started)
        assert result.reset_had_activity is False


class TestAdvanceCompressionSession:
    def test_cas_advances_route_without_reopening_rows(self, tmp_path):
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        source = _source()
        key = store._generate_session_key(source)
        original = _make_entry(key, "sid_parent")
        store._entries[key] = original

        result = store.advance_compression_session(
            key,
            "sid_parent",
            "sid_tip",
        )

        assert result is not None
        assert result is original
        assert result.session_id == "sid_tip"
        assert store.peek_session_id(key) == "sid_tip"
        db.end_session.assert_not_called()
        db.reopen_session.assert_not_called()


