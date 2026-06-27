"""Tests for parent_session_id persistence on auto-reset (#12857)."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


class TestSessionEntryParentId:
    """Tests for SessionEntry.parent_session_id field."""

    def test_accepts_parent_session_id_field(self):
        """SessionEntry should accept parent_session_id in constructor."""
        entry = SessionEntry(
            session_key="test_key",
            session_id="new_session_123",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_session_id="old_session_456",
        )
        assert entry.parent_session_id == "old_session_456"

    def test_parent_session_id_defaults_to_none(self):
        """New sessions should have parent_session_id=None by default."""
        entry = SessionEntry(
            session_key="test_key",
            session_id="new_session",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert entry.parent_session_id is None

    def test_serialization_roundtrip_includes_parent_session_id(self):
        """parent_session_id should survive to_dict/from_dict roundtrip."""
        entry = SessionEntry(
            session_key="test_key",
            session_id="new_session_abc",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_session_id="parent_session_xyz",
        )

        d = entry.to_dict()
        assert d["parent_session_id"] == "parent_session_xyz"

        restored = SessionEntry.from_dict(d)
        assert restored.parent_session_id == "parent_session_xyz"

    def test_from_dict_handles_missing_parent_session_id(self):
        """from_dict should default parent_session_id to None when missing."""
        data = {
            "session_key": "test_key",
            "session_id": "s1",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            # No parent_session_id — old format
        }
        entry = SessionEntry.from_dict(data)
        assert entry.parent_session_id is None

    def test_from_dict_handles_null_parent_session_id(self):
        """from_dict should handle explicit null parent_session_id."""
        data = {
            "session_key": "test_key",
            "session_id": "s1",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "parent_session_id": None,
        }
        entry = SessionEntry.from_dict(data)
        assert entry.parent_session_id is None


class TestGetOrCreateSessionParentId:
    """Tests for parent_session_id in get_or_create_session auto-reset."""

    def _make_source(self):
        """Helper to create a test SessionSource."""
        return SessionSource(
            platform=Platform.LOCAL,
            chat_id="test_chat",
            chat_name="Test Chat",
            chat_type="dm",
            user_id="test_user",
        )

    def _make_store(self, tmp_path):
        """Helper to create a SessionStore with correct constructor."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            return SessionStore(sessions_dir=tmp_path, config=config)

    def test_parent_session_id_set_on_auto_reset(self, tmp_path):
        """When session auto-resets, parent_session_id should be the old session_id."""
        store = self._make_store(tmp_path)
        source = self._make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        old_session_id = entry1.session_id
        assert entry1.parent_session_id is None

        # Simulate auto-reset by marking the session as suspended
        entry1.suspended = True
        store._save()

        # Trigger auto-reset
        entry2 = store.get_or_create_session(source)

        assert entry2.was_auto_reset is True
        assert entry2.parent_session_id == old_session_id
        assert entry2.session_id != old_session_id

    def test_parent_session_id_none_on_fresh_session(self, tmp_path):
        """A fresh session (no reset) should have parent_session_id=None."""
        store = self._make_store(tmp_path)
        source = self._make_source()
        entry = store.get_or_create_session(source)

        assert entry.was_auto_reset is False
        assert entry.parent_session_id is None

    def test_parent_session_id_none_when_not_auto_reset(self, tmp_path):
        """Even with an existing session that doesn't auto-reset, parent_session_id stays None."""
        store = self._make_store(tmp_path)
        source = self._make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        original_id = entry1.session_id

        # Get same session again (no reset triggered)
        entry2 = store.get_or_create_session(source)

        assert entry2.session_id == original_id
        assert entry2.parent_session_id is None

    def test_parent_session_id_forwarded_to_sqlite_on_auto_reset(self, tmp_path):
        """Auto-reset should persist the parent link in the SQLite session graph."""
        source = self._make_source()
        fake_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=fake_db):
            store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

        entry1 = store.get_or_create_session(source)
        old_session_id = entry1.session_id
        entry1.suspended = True
        store._save()

        entry2 = store.get_or_create_session(source)

        fake_db.end_session.assert_called_once_with(old_session_id, "session_reset")
        fake_db.create_session.assert_called_with(
            session_id=entry2.session_id,
            source=source.platform.value,
            user_id=source.user_id,
            parent_session_id=old_session_id,
        )
