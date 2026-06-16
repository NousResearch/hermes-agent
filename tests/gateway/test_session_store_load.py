"""Tests for SessionStore._ensure_loaded_locked resilience.

Verifies that sessions.json entries with unexpected types (bool, null, int)
are skipped gracefully instead of aborting the entire load with an unhandled
TypeError, which caused '[gateway] Warning: Failed to load sessions: argument
of type bool is not iterable' on every gateway startup (issue #46994).
"""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = None
    store._loaded = False
    return store


def _valid_entry_dict(key: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "session_key": key,
        "session_id": f"sid_{key}",
        "created_at": now,
        "updated_at": now,
        "platform": "telegram",
        "chat_type": "dm",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "last_prompt_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_status": "unknown",
        "expiry_finalized": False,
        "suspended": False,
        "resume_pending": False,
        "resume_reason": None,
        "last_resume_marked_at": None,
        "is_fresh_reset": False,
        "was_auto_reset": False,
        "auto_reset_reason": None,
        "reset_had_activity": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSessionStoreLoadResilience:
    def test_bool_entry_does_not_abort_load(self, tmp_path):
        """A JSON bool entry must be skipped, not raise TypeError and abort.

        Regression for 'Failed to load sessions: argument of type bool is not
        iterable' (#46994): `except (ValueError, KeyError)` did not catch the
        TypeError raised by `"origin" in True` inside SessionEntry.from_dict.
        """
        good_key = "agent:main:telegram:dm:123"
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({
                good_key: _valid_entry_dict(good_key),
                "stale_migration_key": True,   # the corrupted / migrated entry
            }),
            encoding="utf-8",
        )
        store = _make_store(tmp_path)
        # Must not raise; must not print the "Failed to load sessions" warning
        store._ensure_loaded_locked()

        assert store._loaded is True
        # Good entry loaded
        assert good_key in store._entries
        # Bad entry silently skipped
        assert "stale_migration_key" not in store._entries

    def test_null_entry_does_not_abort_load(self, tmp_path):
        """A JSON null (None) entry is also skipped without crashing."""
        good_key = "agent:main:slack:dm:U9999"
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({
                good_key: _valid_entry_dict(good_key),
                "null_entry": None,
            }),
            encoding="utf-8",
        )
        store = _make_store(tmp_path)
        store._ensure_loaded_locked()

        assert good_key in store._entries
        assert "null_entry" not in store._entries

    def test_integer_entry_does_not_abort_load(self, tmp_path):
        """Any non-dict entry type is skipped without crashing."""
        good_key = "agent:main:discord:dm:D1"
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({
                good_key: _valid_entry_dict(good_key),
                "int_entry": 42,
            }),
            encoding="utf-8",
        )
        store = _make_store(tmp_path)
        store._ensure_loaded_locked()

        assert good_key in store._entries
        assert "int_entry" not in store._entries

    def test_unknown_platform_loaded_with_none(self, tmp_path):
        """Unknown platform names are loaded gracefully with platform=None.

        SessionEntry.from_dict() catches the ValueError from Platform() internally
        and continues — the entry is NOT discarded, just has platform=None.
        This is intentional: don't throw away session state for a plugin platform
        that isn't registered at startup.
        """
        good_key = "agent:main:telegram:dm:456"
        plugin_key = "agent:main:unknown_future_platform:dm:789"
        sessions_file = tmp_path / "sessions.json"
        plugin_entry = _valid_entry_dict(plugin_key)
        plugin_entry["platform"] = "unknown_future_platform"
        sessions_file.write_text(
            json.dumps({
                good_key: _valid_entry_dict(good_key),
                plugin_key: plugin_entry,
            }),
            encoding="utf-8",
        )
        store = _make_store(tmp_path)
        store._ensure_loaded_locked()

        assert good_key in store._entries
        # Unknown-platform entry is retained (platform=None), not discarded
        assert plugin_key in store._entries
        assert store._entries[plugin_key].platform is None

    def test_warning_logged_for_bool_entry(self, tmp_path, caplog):
        """A warning is emitted to the logger when a malformed entry is skipped."""
        import logging
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({"bad_key": True}),
            encoding="utf-8",
        )
        store = _make_store(tmp_path)
        with caplog.at_level(logging.WARNING, logger="gateway.session"):
            store._ensure_loaded_locked()

        assert any("bad_key" in r.message for r in caplog.records)
