"""Session expiry finalization closes sessions as session_reset.

Regression coverage for #61220: the expiry watcher marks a session expired,
then agent cleanup can close it as ``agent_close``. Stale routing recovery treats
``agent_close`` as recoverable, so expired sessions were reopened with full
history unless expiry finalization also persisted the real conversation boundary
as ``end_reason='session_reset'``.

These tests use a real store handle to verify the actual recovery contract in
``routing.latest_session_for_peer``, not just call counts on a MagicMock.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.fixture
def db(tmp_path: Path):
    return storage_handle(tmp_path)


_SESSION_KEY = "agent:main:telegram:dm:8494508720"
_SOURCE = "telegram"
_USER_ID = "8494508720"


# ------------------------------------------------------------------
# sessions.promote_to_reset: unit tests on a real store
# ------------------------------------------------------------------

class TestPromoteToSessionReset:
    """``sessions.promote_to_reset`` promotes only safe rows."""

    def test_promotes_live_row(self, db) -> None:
        """A live row (ended_at IS NULL) is promoted to session_reset."""
        seed_session(
            db, "sid-live", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        # Seed a message so recovery would match
        seed_messages(db, "sid-live", {"role": "user", "content": "hello"})

        assert db.sessions.promote_to_reset("sid-live") is True
        row = session_row(db, "sid-live")
        assert row["end_reason"] == "session_reset"
        assert row["ended_at"] is not None


    def test_does_not_overwrite_existing_session_reset(self, db) -> None:
        """Already-promoted rows are idempotently skipped."""
        seed_session(
            db, "sid-reset", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.sessions.end("sid-reset", "session_reset")

        # Should be a no-op (rowcount = 0)
        assert db.sessions.promote_to_reset("sid-reset") is False
        row = session_row(db, "sid-reset")
        assert row["end_reason"] == "session_reset"


# ------------------------------------------------------------------
# Integration: promotion blocks stale-route recovery
# ------------------------------------------------------------------

class TestPromotionBlocksRecovery:
    """After promotion, routing.latest_session_for_peer must NOT
    recover the session — it is now ended with session_reset, which
    the recovery query excludes (only live/agent_close are recoverable).
    """

    def test_live_session_recoverable_before_promotion(self, db) -> None:
        seed_session(
            db, "sid-pre", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        seed_messages(db, "sid-pre", {"role": "user", "content": "hello"})

        recovered = db.routing.latest_session_for_peer(
            source=_SOURCE, session_key=_SESSION_KEY,
            user_id=_USER_ID, chat_id="8494508720", chat_type="dm",
        )
        assert recovered is not None
        assert recovered.session.session_id == "sid-pre"


