"""Session expiry finalization closes sessions as session_reset.

Regression coverage for #61220: the expiry watcher marks a session expired,
then agent cleanup can close it as ``agent_close``. Stale routing recovery treats
``agent_close`` as recoverable, so expired sessions were reopened with full
history unless expiry finalization also persisted the real conversation boundary
as ``end_reason='session_reset'``.

These tests use a real ``SessionDB`` (in-memory) to verify the actual recovery
contract in ``find_latest_gateway_session_for_peer`` — not just call counts on
a MagicMock.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


_SESSION_KEY = "agent:main:telegram:dm:8494508720"
_SOURCE = "telegram"
_USER_ID = "8494508720"


# ------------------------------------------------------------------
# promote_to_session_reset — unit tests on real DB
# ------------------------------------------------------------------

class TestPromoteToSessionReset:
    """promote_to_session_reset promotes only safe rows."""

    def test_promotes_live_row(self, db: SessionDB) -> None:
        """A live row (ended_at IS NULL) is promoted to session_reset."""
        db.create_session(
            "sid-live", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        # Seed a message so recovery would match
        db.append_message("sid-live", "user", "hello")

        assert db.promote_to_session_reset("sid-live") is True
        row = db.get_session("sid-live")
        assert row["end_reason"] == "session_reset"
        assert row["ended_at"] is not None

    def test_promotes_agent_close_row(self, db: SessionDB) -> None:
        """A row ended with agent_close is promoted to session_reset."""
        db.create_session(
            "sid-ac", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.append_message("sid-ac", "user", "hello")
        db.end_session("sid-ac", "agent_close")

        assert db.promote_to_session_reset("sid-ac") is True
        row = db.get_session("sid-ac")
        assert row["end_reason"] == "session_reset"

    def test_does_not_overwrite_compression(self, db: SessionDB) -> None:
        """An existing compression boundary must not be overwritten."""
        db.create_session(
            "sid-comp", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.end_session("sid-comp", "compression")

        assert db.promote_to_session_reset("sid-comp") is False
        row = db.get_session("sid-comp")
        assert row["end_reason"] == "compression"

    def test_does_not_overwrite_existing_session_reset(self, db: SessionDB) -> None:
        """Already-promoted rows are idempotently skipped."""
        db.create_session(
            "sid-reset", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.end_session("sid-reset", "session_reset")

        # Should be a no-op (rowcount = 0)
        assert db.promote_to_session_reset("sid-reset") is False
        row = db.get_session("sid-reset")
        assert row["end_reason"] == "session_reset"

    def test_does_not_overwrite_new_command(self, db: SessionDB) -> None:
        """A /new-command boundary is preserved."""
        db.create_session(
            "sid-new", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.end_session("sid-new", "new_command")

        assert db.promote_to_session_reset("sid-new") is False
        row = db.get_session("sid-new")
        assert row["end_reason"] == "new_command"

    def test_noop_on_missing_session(self, db: SessionDB) -> None:
        """Non-existent session_id returns False without error."""
        assert db.promote_to_session_reset("nonexistent") is False

    def test_noop_on_empty_session_id(self, db: SessionDB) -> None:
        assert db.promote_to_session_reset("") is False


# ------------------------------------------------------------------
# Integration: promotion blocks stale-route recovery
# ------------------------------------------------------------------

class TestPromotionBlocksRecovery:
    """After promotion, find_latest_gateway_session_for_peer must NOT
    recover the session — it is now ended with session_reset, which
    the recovery query excludes (only live/agent_close are recoverable).
    """

    def test_live_session_recoverable_before_promotion(self, db: SessionDB) -> None:
        db.create_session(
            "sid-pre", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.append_message("sid-pre", "user", "hello")

        recovered = db.find_latest_gateway_session_for_peer(
            source=_SOURCE, session_key=_SESSION_KEY,
            user_id=_USER_ID, chat_id="8494508720", chat_type="dm",
        )
        assert recovered is not None
        assert recovered["id"] == "sid-pre"

    def test_promoted_session_not_recoverable(self, db: SessionDB) -> None:
        db.create_session(
            "sid-post", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.append_message("sid-post", "user", "hello")
        db.promote_to_session_reset("sid-post")

        recovered = db.find_latest_gateway_session_for_peer(
            source=_SOURCE, session_key=_SESSION_KEY,
            user_id=_USER_ID, chat_id="8494508720", chat_type="dm",
        )
        # session_reset rows are not in the recovery set
        assert recovered is None

    def test_agent_close_session_recoverable_but_not_after_promotion(
        self, db: SessionDB
    ) -> None:
        db.create_session(
            "sid-ac-rec", _SOURCE,
            user_id=_USER_ID, session_key=_SESSION_KEY,
            chat_id="8494508720", chat_type="dm",
        )
        db.append_message("sid-ac-rec", "user", "hello")
        db.end_session("sid-ac-rec", "agent_close")

        # agent_close is recoverable
        recovered = db.find_latest_gateway_session_for_peer(
            source=_SOURCE, session_key=_SESSION_KEY,
            user_id=_USER_ID, chat_id="8494508720", chat_type="dm",
        )
        assert recovered is not None
        assert recovered["id"] == "sid-ac-rec"

        # After promotion, it is no longer recoverable
        db.promote_to_session_reset("sid-ac-rec")
        recovered2 = db.find_latest_gateway_session_for_peer(
            source=_SOURCE, session_key=_SESSION_KEY,
            user_id=_USER_ID, chat_id="8494508720", chat_type="dm",
        )
        assert recovered2 is None
