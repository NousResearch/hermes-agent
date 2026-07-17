"""Tests for reset-policy enforcement on DB-recovered gateway sessions.

Regression coverage for #66255: when a session key is missing from the
in-memory ``_entries`` map (e.g. right after a gateway restart),
``get_or_create_session`` recovers it from state.db via
``_query_recoverable_session``. That path used to adopt the recovered row
unconditionally — never evaluating ``_should_reset()`` — and stamped
``updated_at`` with the recovery moment instead of the session's real last
activity, re-arming the idle clock from zero. A session that should have
expired hours or days ago would be silently resumed forever.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionSource, SessionStore


def _make_store_with_db(tmp_path, db_mock, *, reset_policy) -> SessionStore:
    config = GatewayConfig(default_reset_policy=reset_policy)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5140768830",
        chat_type="dm",
        user_id="5140768830",
        user_name="João",
    )


class TestRecoveredSessionHonorsResetPolicy:
    def test_recovered_row_past_idle_deadline_is_not_resumed(self, tmp_path):
        """A row recovered from state.db that's already idle-expired must be
        ended, not silently adopted with a zeroed idle clock."""
        db = MagicMock()
        old_started_at = (datetime.now() - timedelta(hours=5)).timestamp()
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_old",
            "started_at": old_started_at,
            "message_count": 12,
        }
        # No messages fixture wired up -> falls back to started_at, which is
        # also well past the idle deadline.
        db.get_last_message_timestamp.return_value = None

        store = _make_store_with_db(
            tmp_path, db,
            reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )

        entry, reset_reason, had_activity = store._query_recoverable_session(
            session_key="k", source=_make_source(), now=datetime.now(),
        )

        assert entry is None
        assert reset_reason == "idle"
        assert had_activity is True
        db.end_session.assert_called_once_with("sid_old", "session_reset")

    def test_recovered_row_within_idle_window_is_resumed(self, tmp_path):
        """A row recovered shortly after activity is still a valid resume."""
        db = MagicMock()
        recent_ts = (datetime.now() - timedelta(minutes=5)).timestamp()
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recent",
            "started_at": recent_ts,
            "message_count": 3,
        }
        db.get_last_message_timestamp.return_value = recent_ts

        store = _make_store_with_db(
            tmp_path, db,
            reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )

        entry, reset_reason, had_activity = store._query_recoverable_session(
            session_key="k", source=_make_source(), now=datetime.now(),
        )

        assert entry is not None
        assert entry.session_id == "sid_recent"
        assert reset_reason is None
        db.end_session.assert_not_called()

    def test_recovered_updated_at_uses_last_message_not_recovery_moment(self, tmp_path):
        """updated_at must reflect real last activity, not the recovery time.

        Without this, a session recovered long after its last message would
        report updated_at == now, making it look freshly active to any
        future _should_reset() evaluation.
        """
        db = MagicMock()
        old_started_at = (datetime.now() - timedelta(days=3)).timestamp()
        old_message_ts = (datetime.now() - timedelta(days=2)).timestamp()
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_old2",
            "started_at": old_started_at,
            "message_count": 5,
        }
        db.get_last_message_timestamp.return_value = old_message_ts

        store = _make_store_with_db(
            tmp_path, db,
            # mode=none so the row isn't ended -- we're only checking the
            # timestamp derivation here, not the reset decision.
            reset_policy=SessionResetPolicy(mode="none"),
        )

        entry, reset_reason, had_activity = store._query_recoverable_session(
            session_key="k", source=_make_source(), now=datetime.now(),
        )

        assert entry is not None
        assert reset_reason is None
        assert entry.updated_at == datetime.fromtimestamp(old_message_ts)
        assert entry.updated_at != entry.created_at

    def test_mode_none_never_ends_a_recovered_row(self, tmp_path):
        """The default policy (mode=none) never rejects a recovered row,
        regardless of age -- this fix must not change default behavior."""
        db = MagicMock()
        ancient = (datetime.now() - timedelta(days=30)).timestamp()
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_ancient",
            "started_at": ancient,
            "message_count": 1,
        }
        db.get_last_message_timestamp.return_value = ancient

        store = _make_store_with_db(
            tmp_path, db, reset_policy=SessionResetPolicy(mode="none"),
        )

        entry, reset_reason, had_activity = store._query_recoverable_session(
            session_key="k", source=_make_source(), now=datetime.now(),
        )

        assert entry is not None
        assert reset_reason is None
        db.end_session.assert_not_called()
