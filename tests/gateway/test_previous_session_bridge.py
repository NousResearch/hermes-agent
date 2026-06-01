"""Tests for the previous-session bridge feature."""
from datetime import datetime, timezone

import pytest

from gateway.session import SessionEntry


def test_session_entry_carries_previous_session_id():
    entry = SessionEntry(
        session_key="signal:dm:user-abc",
        session_id="20260531_120000_aaaa",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="20260530_150000_bbbb",
    )
    assert entry.previous_session_id == "20260530_150000_bbbb"


def test_session_entry_round_trip_preserves_previous_session_id():
    entry = SessionEntry(
        session_key="k",
        session_id="new",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="old",
    )
    restored = SessionEntry.from_dict(entry.to_dict())
    assert restored.previous_session_id == "old"


def test_session_entry_default_previous_session_id_is_none():
    entry = SessionEntry(
        session_key="k",
        session_id="s",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert entry.previous_session_id is None


# ---------------------------------------------------------------------------
# Task 2: SessionStore.get_or_create_session populates previous_session_id
# ---------------------------------------------------------------------------

from pathlib import Path
from gateway.config import GatewayConfig, SessionResetPolicy
from gateway.session import SessionStore, SessionSource, Platform


def _src(uid="u1"):
    return SessionSource(
        platform=Platform.SIGNAL, chat_id="c1", user_id=uid, chat_type="dm"
    )


def test_auto_reset_populates_previous_session_id(tmp_path, monkeypatch):
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1)

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    src = _src()
    first = store.get_or_create_session(src)
    first.total_tokens = 500  # simulate activity
    # Force expiry by backdating updated_at (gateway uses naive local datetimes)
    from datetime import datetime, timedelta
    first.updated_at = datetime.now() - timedelta(hours=2)

    second = store.get_or_create_session(src)
    assert second.session_id != first.session_id
    assert second.was_auto_reset is True
    assert second.previous_session_id == first.session_id


def test_first_session_has_no_previous_session_id(tmp_path):
    cfg = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    entry = store.get_or_create_session(_src())
    assert entry.previous_session_id is None


def test_empty_session_rotation_does_not_set_previous_session_id(tmp_path):
    """If the prior session had zero activity, don't bridge — nothing useful to carry."""
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    src = _src()
    first = store.get_or_create_session(src)
    # No total_tokens bump — simulates an empty session
    from datetime import datetime, timedelta
    first.updated_at = datetime.now() - timedelta(hours=2)

    second = store.get_or_create_session(src)
    assert second.was_auto_reset is True
    assert second.previous_session_id is None
