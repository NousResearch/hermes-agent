"""
Regression test for NousResearch/hermes-agent#46994

Gateway: "Failed to load sessions: argument of type 'bool' is not iterable"
on every startup when sessions.json contains a corrupted entry where a dict
field (e.g. origin) was replaced with a boolean.
"""

import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime

from gateway.session import SessionStore, SessionEntry, SessionSource, Platform
from gateway.config import GatewayConfig


def _valid_entry(key: str) -> dict:
    """Return a minimal valid SessionEntry dict."""
    return {
        "session_key": key,
        "session_id": f"sess-{key}",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "platform": "telegram",
        "chat_type": "dm",
        "origin": {
            "platform": "telegram",
            "chat_id": "123",
        },
    }


def _corrupted_entry_bool_origin(key: str) -> dict:
    """Return a corrupted entry where 'origin' is a bool instead of a dict."""
    d = _valid_entry(key)
    d["origin"] = True  # bool instead of dict → triggers TypeError during from_dict
    return d


def _corrupted_entry_bool_value(key: str) -> bool:
    """Return a corrupted entry where the whole value is a bool instead of a dict.

    This happens when sessions.json gets corrupted and an entry value is replaced
    with a bare boolean (e.g. `{"some-key": true}`).  When `from_dict` checks
    `"origin" in data` it hits `argument of type 'bool' is not iterable`.
    """
    return True


class TestSessionStoreLoadCorruption:
    def test_loads_valid_entries_skips_bool_value(self, tmp_path: Path):
        """Regression for #46994: corrupted entry value is a bare boolean."""
        sessions_file = tmp_path / "sessions.json"
        data = {
            "valid-1": _valid_entry("valid-1"),
            "corrupt-1": _corrupted_entry_bool_value("corrupt-1"),
            "valid-2": _valid_entry("valid-2"),
        }
        sessions_file.write_text(json.dumps(data), encoding="utf-8")

        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._ensure_loaded()

        assert "valid-1" in store._entries
        assert "corrupt-1" not in store._entries
        assert "valid-2" in store._entries
        assert len(store._entries) == 2

    def test_loads_valid_entries_skips_bool_origin(self, tmp_path: Path):
        """Also guard the case where origin field itself is a bool."""
        sessions_file = tmp_path / "sessions.json"
        data = {
            "valid-1": _valid_entry("valid-1"),
            "corrupt-1": _corrupted_entry_bool_value("corrupt-1"),
            "valid-2": _valid_entry("valid-2"),
        }
        sessions_file.write_text(json.dumps(data), encoding="utf-8")

        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._ensure_loaded()

        assert "valid-1" in store._entries
        assert "corrupt-1" not in store._entries
        assert "valid-2" in store._entries
        assert len(store._entries) == 2

    def test_all_corrupted_loads_empty(self, tmp_path: Path):
        sessions_file = tmp_path / "sessions.json"
        data = {
            "corrupt-1": _corrupted_entry_bool_value("corrupt-1"),
            "corrupt-2": _corrupted_entry_bool_value("corrupt-2"),
        }
        sessions_file.write_text(json.dumps(data), encoding="utf-8")

        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._ensure_loaded()

        assert len(store._entries) == 0
        assert store._loaded is True

    def test_no_sessions_file_loads_empty(self, tmp_path: Path):
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._ensure_loaded()
        assert len(store._entries) == 0
        assert store._loaded is True
