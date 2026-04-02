"""Tests for proactive memory flush on session expiry.

Verifies that:
1. _is_session_expired() works from a SessionEntry alone (no source needed)
2. The sync callback is no longer called in get_or_create_session
3. memory_flushed flag persists across save/load cycles (prevents restart re-flush)
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

    def test_auto_reset_creates_new_session_after_flush(self, idle_store):
        """When a flushed session auto-resets, a new session_id is created."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        )
        # Create initial session
        entry1 = idle_store.get_or_create_session(source)
        old_sid = entry1.session_id

        # Simulate the watcher having flushed it
        entry1.memory_flushed = True

        # Simulate the session going idle
        entry1.updated_at = datetime.now() - timedelta(minutes=120)
        idle_store._save()

        # Next call should auto-reset
        entry2 = idle_store.get_or_create_session(source)
        assert entry2.session_id != old_sid
        assert entry2.was_auto_reset is True
        # New session starts with memory_flushed=False
        assert entry2.memory_flushed is False

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


class TestMemoryFlushedFlag:
    """The memory_flushed flag on SessionEntry prevents double-flushing."""

    def test_defaults_to_false(self):
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:123",
            session_id="sid_new",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        assert entry.memory_flushed is False

    def test_persists_through_save_load(self, idle_store):
        """memory_flushed=True must survive a save/load cycle (simulates restart)."""
        key = "agent:main:discord:thread:789"
        entry = SessionEntry(
            session_key=key,
            session_id="sid_flushed",
            created_at=datetime.now() - timedelta(hours=5),
            updated_at=datetime.now() - timedelta(hours=5),
            platform=Platform.DISCORD,
            chat_type="thread",
            memory_flushed=True,
        )
        idle_store._entries[key] = entry
        idle_store._save()

        # Simulate restart: clear in-memory state, reload from disk
        idle_store._entries.clear()
        idle_store._loaded = False
        idle_store._ensure_loaded()

        reloaded = idle_store._entries[key]
        assert reloaded.memory_flushed is True

    def test_unflushed_entry_survives_restart_as_unflushed(self, idle_store):
        """An entry without memory_flushed stays False after reload."""
        key = "agent:main:telegram:dm:456"
        entry = SessionEntry(
            session_key=key,
            session_id="sid_not_flushed",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(hours=2),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        idle_store._entries[key] = entry
        idle_store._save()

        idle_store._entries.clear()
        idle_store._loaded = False
        idle_store._ensure_loaded()

        reloaded = idle_store._entries[key]
        assert reloaded.memory_flushed is False

    def test_roundtrip_to_dict_from_dict(self):
        """to_dict/from_dict must preserve memory_flushed."""
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:999",
            session_id="sid_rt",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
            memory_flushed=True,
        )
        d = entry.to_dict()
        assert d["memory_flushed"] is True

        restored = SessionEntry.from_dict(d)
        assert restored.memory_flushed is True

    def test_legacy_entry_without_field_defaults_false(self):
        """Old sessions.json entries missing memory_flushed should default to False."""
        data = {
            "session_key": "agent:main:telegram:dm:legacy",
            "session_id": "sid_legacy",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "platform": "telegram",
            "chat_type": "dm",
            # no memory_flushed key
        }
        entry = SessionEntry.from_dict(data)
        assert entry.memory_flushed is False
