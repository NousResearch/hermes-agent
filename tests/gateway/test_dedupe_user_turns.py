"""Regression tests for issue #47237.

When the gateway persists a user message after a transient provider
failure (429/timeout/auth error), subsequent retries of the same
Telegram message must not stack duplicate user turns in the transcript.
The dedupe guard checks has_platform_message_id before persisting.
"""

from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


class TestHasPlatformMessageId:
    """``messages.has_platform_message_id`` on the store handle."""

    def _make_db(self, tmp_path):
        db = storage_handle(tmp_path)
        seed_session(db, "s1", "cli")
        return db


    def test_returns_false_for_different_session(self, tmp_path):
        db = self._make_db(tmp_path)
        seed_session(db, "s2", "cli")
        seed_messages(db, "s1", {"role": "user", "content": "hello", "platform_message_id": "msg-123"})
        assert not db.messages.has_platform_message_id("s2", "msg-123")




        # The gateway code checks this before calling append_to_transcript,
        # so the second append should never fire.

