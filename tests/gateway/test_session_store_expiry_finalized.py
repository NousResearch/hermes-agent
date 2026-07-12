"""Session expiry finalization closes sessions as session_reset.

Regression coverage for #61220: the expiry watcher marks a session expired,
then agent cleanup can close it as ``agent_close``. Stale routing recovery treats
``agent_close`` as recoverable, so expired sessions were reopened with full
history unless expiry finalization also persisted the real conversation boundary
as ``end_reason='session_reset'``.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore


def _make_store_with_db(tmp_path, db_mock) -> SessionStore:
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="daily"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _entry(session_id: str = "sid-expired") -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:8494508720",
        session_id=session_id,
        created_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        model_override={"provider": "openrouter", "model": "test/model"},
    )


def test_set_expiry_finalized_persists_session_reset_boundary(tmp_path):
    db = MagicMock()
    db.set_expiry_finalized.return_value = None
    db.reopen_session.return_value = None
    db.end_session.return_value = None
    store = _make_store_with_db(tmp_path, db)
    entry = _entry()

    store.set_expiry_finalized(entry)

    assert entry.expiry_finalized is True
    assert entry.model_override is None
    db.set_expiry_finalized.assert_called_once_with("sid-expired", True)
    db.reopen_session.assert_called_once_with("sid-expired")
    db.end_session.assert_called_once_with("sid-expired", "session_reset")


def test_set_expiry_finalized_still_sets_flag_if_end_session_fails(tmp_path):
    db = MagicMock()
    db.set_expiry_finalized.return_value = None
    db.reopen_session.side_effect = RuntimeError("database locked")
    store = _make_store_with_db(tmp_path, db)
    entry = _entry()

    store.set_expiry_finalized(entry)

    assert entry.expiry_finalized is True
    db.set_expiry_finalized.assert_called_once_with("sid-expired", True)
    db.end_session.assert_not_called()
