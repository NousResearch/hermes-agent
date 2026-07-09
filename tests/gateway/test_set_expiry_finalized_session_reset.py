"""Test that set_expiry_finalized sets end_reason='session_reset' to prevent silent recovery."""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from gateway.session import SessionStore, SessionEntry


def test_set_expiry_finalized_sets_session_reset_end_reason(tmp_path):
    """set_expiry_finalized must call reopen_session + end_session(session_reset).

    This prevents the recovery query (find_latest_gateway_session_for_peer)
    from treating expired sessions as recoverable, which would cause token
    accumulation bugs where expired sessions are silently resumed with full
    history. Issue #61220.
    """
    # Create a minimal GatewayConfig for SessionStore.
    config = MagicMock()
    config.write_sessions_json = True

    # Create a SessionStore.
    store = SessionStore(tmp_path, config)

    # Mock the DB.
    db = MagicMock()
    store._db = db

    # Create a session entry.
    entry = SessionEntry(
        session_id="test-session",
        session_key="agent:main:telegram:dm:x",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Set expiry finalized.
    store.set_expiry_finalized(entry)

    # Verify expiry_finalized flag is set (existing behavior).
    assert entry.expiry_finalized

    # Verify DB.set_expiry_finalized was called (existing behavior).
    db.set_expiry_finalized.assert_called_once_with("test-session", True)

    # Verify reopen_session + end_session(session_reset) were called (new behavior).
    db.reopen_session.assert_called_once_with("test-session")
    db.end_session.assert_called_once_with("test-session", "session_reset")


def test_set_expiry_finalized_db_errors_are_caught(tmp_path):
    """DB errors in reopen_session/end_session should be caught and logged.

    This ensures the expiry_finalized flag is still set even if the DB write
    fails, matching the existing error handling pattern for set_expiry_finalized.
    """
    config = MagicMock()
    config.write_sessions_json = True
    store = SessionStore(tmp_path, config)

    db = MagicMock()
    db.reopen_session.side_effect = Exception("DB error")
    store._db = db

    entry = SessionEntry(
        session_id="test-session",
        session_key="agent:main:telegram:dm:x",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Should not raise.
    store.set_expiry_finalized(entry)

    # Verify expiry_finalized flag is still set.
    assert entry.expiry_finalized
    db.set_expiry_finalized.assert_called_once_with("test-session", True)


def test_set_expiry_finalized_without_db(tmp_path):
    """set_expiry_finalized should work when DB is None (e.g. during init)."""
    config = MagicMock()
    config.write_sessions_json = True
    store = SessionStore(tmp_path, config)
    store._db = None

    entry = SessionEntry(
        session_id="test-session",
        session_key="agent:main:telegram:dm:x",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Should not raise.
    store.set_expiry_finalized(entry)

    # Verify expiry_finalized flag is set.
    assert entry.expiry_finalized