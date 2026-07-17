"""Durable session boundary tests for expiry finalization (#61220).

The expiry watcher must make an expired transcript non-recoverable without
breaking recovery for ordinary agent resource eviction. The database update
must also be atomic: the expiry flag and logical end reason cannot diverge.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore
from hermes_state import SessionDB


_SESSION_KEY = "agent:main:telegram:dm:8494508720"
_SOURCE = "telegram"
_USER_ID = "8494508720"


@pytest.fixture
def db(tmp_path: Path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _create_recoverable_session(db: SessionDB, session_id: str) -> None:
    db.create_session(
        session_id,
        _SOURCE,
        user_id=_USER_ID,
        session_key=_SESSION_KEY,
        chat_id=_USER_ID,
        chat_type="dm",
    )
    db.append_message(session_id, "user", "hello")


def _recover(db: SessionDB):
    return db.find_latest_gateway_session_for_peer(
        source=_SOURCE,
        session_key=_SESSION_KEY,
        user_id=_USER_ID,
        chat_id=_USER_ID,
        chat_type="dm",
    )


@pytest.mark.parametrize("prior_reason", [None, "agent_close", "ws_orphan_reap"])
def test_finalize_expired_session_atomically_closes_recoverable_rows(
    db: SessionDB, prior_reason: str | None
) -> None:
    _create_recoverable_session(db, "sid-expired")
    if prior_reason is not None:
        db.end_session("sid-expired", prior_reason)

    assert db.finalize_expired_session("sid-expired") is True

    row = db.get_session("sid-expired")
    assert row is not None
    assert row["expiry_finalized"] == 1
    assert row["end_reason"] == "session_reset"
    assert row["ended_at"] is not None
    assert _recover(db) is None


def test_finalize_expired_session_preserves_explicit_boundary(db: SessionDB) -> None:
    _create_recoverable_session(db, "sid-compressed")
    db.end_session("sid-compressed", "compression")
    before = db.get_session("sid-compressed")
    assert before is not None
    ended_at = before["ended_at"]

    assert db.finalize_expired_session("sid-compressed") is True

    row = db.get_session("sid-compressed")
    assert row is not None
    assert row["expiry_finalized"] == 1
    assert row["end_reason"] == "compression"
    assert row["ended_at"] == ended_at


def test_recovery_rejects_legacy_finalized_agent_close_row(db: SessionDB) -> None:
    """Backstop old rows created before end_reason promotion existed."""
    _create_recoverable_session(db, "sid-legacy")
    db.end_session("sid-legacy", "agent_close")
    db.set_expiry_finalized("sid-legacy", True)

    assert _recover(db) is None


def test_recovery_keeps_ordinary_agent_close_recoverable(db: SessionDB) -> None:
    """Resource eviction without expiry must retain #54878 recovery semantics."""
    _create_recoverable_session(db, "sid-evicted")
    db.end_session("sid-evicted", "agent_close")

    recovered = _recover(db)
    assert recovered is not None
    assert recovered["id"] == "sid-evicted"


def _store(tmp_path: Path, db_mock: MagicMock) -> SessionStore:
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _entry() -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=_SESSION_KEY,
        session_id="sid-store",
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        model_override={"model": "temporary"},
    )


def test_store_does_not_mark_finalized_when_durable_boundary_fails(
    tmp_path: Path,
) -> None:
    """A DB failure must remain retryable instead of becoming a false success."""
    db_mock = MagicMock()
    db_mock.finalize_expired_session.side_effect = RuntimeError("database locked")
    store = _store(tmp_path, db_mock)
    entry = _entry()

    with pytest.raises(RuntimeError, match="database locked"):
        store.set_expiry_finalized(entry)

    assert entry.expiry_finalized is False
    assert entry.model_override == {"model": "temporary"}


def test_store_persists_boundary_before_marking_entry_finalized(tmp_path: Path) -> None:
    db_mock = MagicMock()
    db_mock.finalize_expired_session.return_value = True
    store = _store(tmp_path, db_mock)
    entry = _entry()

    store.set_expiry_finalized(entry)

    db_mock.finalize_expired_session.assert_called_once_with("sid-store")
    assert entry.expiry_finalized is True
    assert entry.model_override is None
