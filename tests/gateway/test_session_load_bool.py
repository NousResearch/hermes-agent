"""Regression tests for issue #46994.

Corrupted sessions.json entries (e.g. a bare bool where a dict is expected)
must not crash the entire session loading loop. The TypeError from
`"origin" in True` escaped the (ValueError, KeyError) except and aborted
loading ALL remaining sessions, not just the corrupted one.
"""

import json
import threading
from pathlib import Path

from gateway.session import SessionStore


class TestSessionLoadBoolCorruption:
    """Verify that non-dict entries in sessions.json are skipped, not fatal."""

    def _make_store(self, tmp_path: Path, sessions_data: dict) -> SessionStore:
        """Create a SessionStore with a pre-populated sessions.json."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "sessions.json").write_text(
            json.dumps(sessions_data), encoding="utf-8"
        )
        # SessionStore requires a config object with session reset policy
        class FakeConfig:
            session_idle_ttl = 0
            session_daily_ttl = 0
            group_sessions_per_user = True
            thread_sessions_per_user = False
            multiplex_profiles = False
            def get_reset_policy(self, *a, **kw):
                return None

        store = SessionStore.__new__(SessionStore)
        store.sessions_dir = sessions_dir
        store._entries = {}
        store._loaded = False
        store._lock = threading.RLock()
        store.config = FakeConfig()
        store._has_active_processes_fn = None
        return store

    def _valid_entry(self, session_id: str = "20260101_120000_abc12345") -> dict:
        return {
            "session_key": "agent:main:telegram:dm:123456",
            "session_id": session_id,
            "created_at": "2026-01-01T12:00:00",
            "updated_at": "2026-01-01T12:30:00",
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_type": "dm",
            },
        }

    def test_bool_entry_skipped_not_fatal(self, tmp_path):
        """A bool entry must not crash the loop or block other sessions."""
        data = {
            "_README": "test sentinel",
            "corrupted_key": True,
            "valid_key": self._valid_entry(),
        }
        store = self._make_store(tmp_path, data)
        store._ensure_loaded()

        # The valid entry must still be loaded
        assert "valid_key" in store._entries
        assert store._entries["valid_key"].session_id == "20260101_120000_abc12345"
        # The corrupted entry must NOT be loaded
        assert "corrupted_key" not in store._entries

    def test_string_entry_skipped(self, tmp_path):
        """A string entry must also be skipped without crashing."""
        data = {
            "bad_string": "not a dict",
            "valid_key": self._valid_entry("20260101_130000_def67890"),
        }
        store = self._make_store(tmp_path, data)
        store._ensure_loaded()

        assert "valid_key" in store._entries
        assert "bad_string" not in store._entries


