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
from hermes_state import SessionDB


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
        db.promote_to_session_reset.assert_called_once_with("sid_old", "idle")
        db.reopen_recoverable_session.assert_not_called()

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
        db.reopen_recoverable_session.assert_called_once_with("sid_recent")

    def test_reopen_race_loss_declines_recovery_instead_of_resurrecting(self, tmp_path):
        """reopen_recoverable_session() returning False must not publish the row.

        Between the recovery lookup and the reopen call (no lock is held),
        another thread can finalize the same row with an explicit boundary
        (session_reset/session_switch/compression) -- reopen_recoverable_session's
        conditional UPDATE then affects zero rows and returns False. Ignoring
        that return value would resurrect a session someone else already
        closed (#66255).
        """
        db = MagicMock()
        recent_ts = (datetime.now() - timedelta(minutes=5)).timestamp()
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_raced",
            "started_at": recent_ts,
            "message_count": 3,
        }
        db.get_last_message_timestamp.return_value = recent_ts
        # Simulate the race: another thread claimed the row first.
        db.reopen_recoverable_session.return_value = False

        store = _make_store_with_db(
            tmp_path, db,
            reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )

        entry, reset_reason, had_activity = store._query_recoverable_session(
            session_key="k", source=_make_source(), now=datetime.now(),
        )

        assert entry is None
        assert reset_reason is None
        assert had_activity is False
        db.reopen_recoverable_session.assert_called_once_with("sid_raced")
        # The row must not be durably touched here -- the thread that won
        # the race already recorded its own boundary.
        db.promote_to_session_reset.assert_not_called()
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


class TestRecoveredSessionRealDB:
    def test_expired_durable_row_is_promoted_and_no_longer_recoverable(self, tmp_path):
        """An expired agent_close row becomes a durable idle boundary (#66255)."""
        source = _make_source()
        session_key = "agent:main:telegram:dm:5140768830"
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session(
                "sid_expired",
                "telegram",
                user_id=source.user_id,
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
            )
            old_timestamp = (datetime.now() - timedelta(hours=5)).timestamp()
            db.append_message(
                "sid_expired", "user", "old message", timestamp=old_timestamp,
            )
            db.end_session("sid_expired", "agent_close")
            store = _make_store_with_db(
                tmp_path,
                db,
                reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
            )

            entry = store._recover_session_from_db(
                session_key=session_key, source=source, now=datetime.now(),
            )

            assert entry is None
            row = db.get_session("sid_expired")
            assert row["end_reason"] == "idle"
            assert db.find_latest_gateway_session_for_peer(
                source="telegram",
                user_id=source.user_id,
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
            ) is None
        finally:
            db.close()

    def test_real_last_message_timestamp_query_drives_recovered_updated_at(self, tmp_path):
        """Recovery reads the real MAX(messages.timestamp) from SessionDB."""
        source = _make_source()
        session_key = "agent:main:telegram:dm:5140768830"
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session(
                "sid_timestamp",
                "telegram",
                user_id=source.user_id,
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
            )
            earlier = (datetime.now() - timedelta(hours=3)).timestamp()
            latest = (datetime.now() - timedelta(minutes=7)).timestamp()
            db.append_message("sid_timestamp", "user", "first", timestamp=earlier)
            db.append_message("sid_timestamp", "assistant", "second", timestamp=latest)
            store = _make_store_with_db(
                tmp_path, db, reset_policy=SessionResetPolicy(mode="none"),
            )

            entry, reset_reason, had_activity = store._query_recoverable_session(
                session_key=session_key, source=source, now=datetime.now(),
            )

            assert db.get_last_message_timestamp("sid_timestamp") == latest
            assert entry is not None
            assert entry.updated_at == datetime.fromtimestamp(latest)
            assert reset_reason is None
            assert had_activity is False
        finally:
            db.close()
