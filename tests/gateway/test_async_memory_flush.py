"""Tests for proactive memory flush on session expiry.

Verifies that:
1. _is_session_expired() works from a SessionEntry alone (no source needed)
2. The sync callback is no longer called in get_or_create_session
3. _pre_flushed_sessions tracking works correctly
4. The background watcher can detect expired sessions
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from gateway.config import Platform, GatewayConfig, SessionResetPolicy
from gateway.session import SessionSource, SessionStore, SessionEntry


@pytest.fixture()
def idle_store(tmp_path):
    """SessionStore with a 60-minute idle reset policy."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path, config=config)
    s._db = None
    s._loaded = True
    return s


@pytest.fixture()
def no_reset_store(tmp_path):
    """SessionStore with no reset policy (mode=none)."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path, config=config)
    s._db = None
    s._loaded = True
    return s


class TestIsSessionExpired:
    """_is_session_expired should detect expiry from entry alone."""

    def test_idle_session_expired(self, idle_store):
        entry = SessionEntry(
            session_key="agent:main:telegram:dm",
            session_id="sid_1",
            created_at=datetime.now() - timedelta(hours=3),
            updated_at=datetime.now() - timedelta(minutes=120),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert idle_store._is_session_expired(entry) is True

    def test_active_session_not_expired(self, idle_store):
        entry = SessionEntry(
            session_key="agent:main:telegram:dm",
            session_id="sid_2",
            created_at=datetime.now() - timedelta(hours=1),
            updated_at=datetime.now() - timedelta(minutes=10),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert idle_store._is_session_expired(entry) is False

    def test_none_mode_never_expires(self, no_reset_store):
        entry = SessionEntry(
            session_key="agent:main:telegram:dm",
            session_id="sid_3",
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now() - timedelta(days=30),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert no_reset_store._is_session_expired(entry) is False

    def test_active_processes_prevent_expiry(self, idle_store):
        """Sessions with active background processes should never expire."""
        idle_store._has_active_processes_fn = lambda key: True
        entry = SessionEntry(
            session_key="agent:main:telegram:dm",
            session_id="sid_4",
            created_at=datetime.now() - timedelta(hours=5),
            updated_at=datetime.now() - timedelta(hours=5),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert idle_store._is_session_expired(entry) is False

    def test_daily_mode_expired(self, tmp_path):
        """Daily mode should expire sessions from before today's reset hour."""
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="daily", at_hour=4),
        )
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True

        entry = SessionEntry(
            session_key="agent:main:telegram:dm",
            session_id="sid_5",
            created_at=datetime.now() - timedelta(days=2),
            updated_at=datetime.now() - timedelta(days=2),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert store._is_session_expired(entry) is True


class TestGetOrCreateSessionNoCallback:
    """get_or_create_session should NOT call a sync flush callback."""

    def test_auto_reset_cleans_pre_flushed_marker(self, idle_store):
        """When a session auto-resets, the pre_flushed marker should be discarded."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        )
        # Create initial session
        entry1 = idle_store.get_or_create_session(source)
        old_sid = entry1.session_id

        # Simulate the watcher having flushed it
        idle_store._pre_flushed_sessions.add(old_sid)

        # Simulate the session going idle
        entry1.updated_at = datetime.now() - timedelta(minutes=120)
        idle_store._save()

        # Next call should auto-reset
        entry2 = idle_store.get_or_create_session(source)
        assert entry2.session_id != old_sid
        assert entry2.was_auto_reset is True

        # The old session_id should be removed from pre_flushed
        assert old_sid not in idle_store._pre_flushed_sessions

    def test_no_sync_callback_invoked(self, idle_store):
        """No synchronous callback should block during auto-reset."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        )
        entry1 = idle_store.get_or_create_session(source)
        entry1.updated_at = datetime.now() - timedelta(minutes=120)
        idle_store._save()

        # Verify no _on_auto_reset attribute
        assert not hasattr(idle_store, '_on_auto_reset')

        # This should NOT block (no sync LLM call)
        entry2 = idle_store.get_or_create_session(source)
        assert entry2.was_auto_reset is True


class TestPreFlushedSessionsTracking:
    """The _pre_flushed_sessions set should prevent double-flushing."""

    def test_starts_empty(self, idle_store):
        assert len(idle_store._pre_flushed_sessions) == 0

    def test_add_and_check(self, idle_store):
        idle_store._pre_flushed_sessions.add("sid_old")
        assert "sid_old" in idle_store._pre_flushed_sessions
        assert "sid_other" not in idle_store._pre_flushed_sessions

    def test_discard_on_reset(self, idle_store):
        """discard should remove without raising if not present."""
        idle_store._pre_flushed_sessions.add("sid_a")
        idle_store._pre_flushed_sessions.discard("sid_a")
        assert "sid_a" not in idle_store._pre_flushed_sessions
        # discard on non-existent should not raise
        idle_store._pre_flushed_sessions.discard("sid_nonexistent")
