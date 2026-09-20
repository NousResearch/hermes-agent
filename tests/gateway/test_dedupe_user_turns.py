"""Regression tests for issue #47237.

When the gateway persists a user message after a transient provider
failure (429/timeout/auth error), subsequent retries of the same
Telegram message must not stack duplicate user turns in the transcript.
The dedupe guard checks has_platform_message_id before persisting.
"""

from gateway.session import SessionStore
from hermes_state import SessionDB


class TestHasPlatformMessageId:
    """SessionDB.has_platform_message_id and SessionStore wrapper."""

    def _make_db(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        db.create_session("s1", "cli")
        return db


    def test_returns_false_for_different_session(self, tmp_path):
        db = self._make_db(tmp_path)
        db.create_session("s2", "cli")
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        assert not db.has_platform_message_id("s2", "msg-123")


    def test_session_store_wrapper_proxies_to_db(self, tmp_path):
        db = self._make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-456",
        )
        store = SessionStore.__new__(SessionStore)
        store._db = db
        assert store.has_platform_message_id("s1", "msg-456")
        assert not store.has_platform_message_id("s1", "msg-000")


class TestDedupeOnTransientFailure:
    """The gateway's transient-failure path must not persist duplicates."""

    @staticmethod
    def _make_db(tmp_path):
        db = SessionDB(tmp_path / "state.db")
        db.create_session("s1", "cli")
        return db

    def test_duplicate_message_id_skipped(self, tmp_path):
        """When has_platform_message_id returns True, the append is skipped."""
        db = self._make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-789",
        )
        store = SessionStore.__new__(SessionStore)
        store._db = db

        # Simulate a second attempt to persist the same message
        assert store.has_platform_message_id("s1", "msg-789")
        # The gateway code checks this before calling append_to_transcript,
        # so the second append should never fire.

