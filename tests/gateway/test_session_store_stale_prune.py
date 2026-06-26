"""Tests for SessionStore._prune_stale_sessions_locked — FM9 self-healing.

When a gateway crashes (exit code 1) the normal shutdown path is skipped and
sessions.json is left pointing at sessions that are already ended in state.db.
On the next startup _ensure_loaded_locked calls _prune_stale_sessions_locked to
detect and remove those stale routing entries before they can silently drop
incoming messages.

Failure mode being tested (FM9):
  - Gateway crashes mid-run → sessions.json not cleared
  - state.db marks the session ended (end_reason IS NOT NULL)
  - Gateway restarts → trusts sessions.json → every message silently dropped
  - Fix: prune stale entries on startup
"""

import json
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(key: str, session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )


def _make_store_with_db(tmp_path, db_mock) -> SessionStore:
    """Build a SessionStore with a mock SessionDB, bypassing disk load."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _db_returning(rows: dict) -> MagicMock:
    """Return a SessionDB mock where get_session maps session_id → row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    return db


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestPruneStaleSessionsLocked:
    def test_prunes_ended_session(self, tmp_path):
        """Entry pointing at a session with end_reason is removed."""
        db = _db_returning({
            "sid_dm": {"end_reason": "agent_close", "id": "sid_dm"},
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["dm_key"] = _make_entry("dm_key", "sid_dm")

        store._prune_stale_sessions_locked()

        assert "dm_key" not in store._entries

    def test_keeps_live_session(self, tmp_path):
        """Entry pointing at a session with end_reason=None is kept."""
        db = _db_returning({
            "sid_live": {"end_reason": None, "id": "sid_live"},
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        store._prune_stale_sessions_locked()

        assert "live_key" in store._entries

    def test_keeps_session_absent_from_db(self, tmp_path):
        """Entry for a session_id not in state.db (legacy) is left alone."""
        db = _db_returning({})  # empty — session never in DB
        store = _make_store_with_db(tmp_path, db)
        store._entries["legacy_key"] = _make_entry("legacy_key", "sid_legacy")

        store._prune_stale_sessions_locked()

        assert "legacy_key" in store._entries

    def test_prunes_multiple_stale_entries(self, tmp_path):
        """All stale entries are removed in one pass."""
        db = _db_returning({
            "sid_a": {"end_reason": "agent_close", "id": "sid_a"},
            "sid_b": {"end_reason": "session_reset", "id": "sid_b"},
            "sid_c": {"end_reason": None, "id": "sid_c"},  # alive — keep
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["key_a"] = _make_entry("key_a", "sid_a")
        store._entries["key_b"] = _make_entry("key_b", "sid_b")
        store._entries["key_c"] = _make_entry("key_c", "sid_c")

        store._prune_stale_sessions_locked()

        assert "key_a" not in store._entries
        assert "key_b" not in store._entries
        assert "key_c" in store._entries

    def test_noop_when_db_is_none(self, tmp_path):
        """If SQLite is unavailable (_db=None) pruning is silently skipped."""
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
        )
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries  # unchanged

    def test_noop_when_no_entries(self, tmp_path):
        """Empty _entries dict → no DB calls, no error."""
        db = MagicMock()
        store = _make_store_with_db(tmp_path, db)

        store._prune_stale_sessions_locked()

        db.get_session.assert_not_called()

    def test_db_error_is_non_fatal(self, tmp_path):
        """A DB exception during pruning must not crash the gateway startup."""
        db = MagicMock()
        db.get_session.side_effect = Exception("DB locked")
        store = _make_store_with_db(tmp_path, db)
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        # Entry is left intact (safe fallback)
        assert "key" in store._entries

    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        """sessions.json must be updated after stale entries are removed."""
        db = _db_returning({
            "sid_stale": {"end_reason": "agent_close", "id": "sid_stale"},
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()

    def test_sessions_json_not_rewritten_when_nothing_pruned(self, tmp_path):
        """_save() must NOT be called when no entries are removed."""
        db = _db_returning({
            "sid_live": {"end_reason": None, "id": "sid_live"},
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _ensure_loaded_locked calls _prune_stale_sessions_locked
# ---------------------------------------------------------------------------

class TestEnsureLoadedCallsPrune:
    def test_prune_called_during_load(self, tmp_path):
        """_prune_stale_sessions_locked must be invoked at the end of startup."""
        # Write a sessions.json with one stale entry
        entry = _make_entry("dm_key", "sid_stale")
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({"dm_key": entry.to_dict()}, indent=2),
            encoding="utf-8",
        )

        db = _db_returning({
            "sid_stale": {"end_reason": "agent_close", "id": "sid_stale"},
        })
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
        )
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        # Trigger the real load path (not mocked)
        store._ensure_loaded()

        assert "dm_key" not in store._entries, (
            "Stale entry must be pruned during _ensure_loaded"
        )

    def test_live_entry_survives_load(self, tmp_path):
        """A live sessions.json entry must not be removed during load."""
        entry = _make_entry("active_key", "sid_live")
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(
            json.dumps({"active_key": entry.to_dict()}, indent=2),
            encoding="utf-8",
        )

        db = _db_returning({
            "sid_live": {"end_reason": None, "id": "sid_live"},
        })
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
        )
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "active_key" in store._entries
