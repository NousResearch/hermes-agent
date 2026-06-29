"""Tests for SessionStore._prune_stale_sessions_locked — crash self-healing.

When a gateway crashes (exit code 1) the graceful shutdown path is skipped and
sessions.json is left pointing at sessions already ended in state.db. On the
next startup _ensure_loaded_locked calls _prune_stale_sessions_locked to detect
and remove those stale routing entries before get_or_create_session() can reuse
them and silently route incoming messages into a closed session (#52804).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

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
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _db_returning(rows: dict) -> MagicMock:
    """SessionDB mock where get_session maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    return db


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestPruneStaleSessionsLocked:
    def test_prunes_ended_session(self, tmp_path):
        db = _db_returning({"sid_dm": {"end_reason": "agent_close", "id": "sid_dm"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["dm_key"] = _make_entry("dm_key", "sid_dm")

        store._prune_stale_sessions_locked()

        assert "dm_key" not in store._entries

    def test_keeps_live_session(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        store._prune_stale_sessions_locked()

        assert "live_key" in store._entries

    def test_keeps_session_absent_from_db(self, tmp_path):
        """Entry for a session_id not in state.db (legacy) is left alone."""
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        store._entries["legacy_key"] = _make_entry("legacy_key", "sid_legacy")

        store._prune_stale_sessions_locked()

        assert "legacy_key" in store._entries

    def test_prunes_multiple_stale_entries(self, tmp_path):
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
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries

    def test_noop_when_no_entries(self, tmp_path):
        db = MagicMock()
        store = _make_store_with_db(tmp_path, db)

        store._prune_stale_sessions_locked()

        db.get_session.assert_not_called()

    def test_db_error_is_non_fatal(self, tmp_path):
        db = MagicMock()
        db.get_session.side_effect = Exception("DB locked")
        store = _make_store_with_db(tmp_path, db)
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries  # safe fallback — keep on error

    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()

    def test_sessions_json_not_rewritten_when_nothing_pruned(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _ensure_loaded_locked calls _prune_stale_sessions_locked
# ---------------------------------------------------------------------------

class TestEnsureLoadedCallsPrune:
    def test_stale_entry_pruned_during_load(self, tmp_path):
        entry = _make_entry("dm_key", "sid_stale")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"dm_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "dm_key" not in store._entries

    def test_live_entry_survives_load(self, tmp_path):
        entry = _make_entry("active_key", "sid_live")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"active_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "active_key" in store._entries


def test_session_store_remove():
    """SessionStore.remove() should remove a key from the routing index.

    Regression test for #54878: after idle-TTL eviction, sessions.json
    held stale entries that routed messages into ended sessions.
    """
    import tempfile
    from pathlib import Path
    from gateway.session import SessionStore, SessionSource
    from gateway.config import GatewayConfig, Platform

    with tempfile.TemporaryDirectory() as tmp:
        sessions_dir = Path(tmp) / "sessions"
        config = GatewayConfig()
        store = SessionStore(sessions_dir, config)

        # Create a session entry via get_or_create
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            user_id="user1",
        )
        entry = store.get_or_create_session(source)
        assert entry is not None

        # Remove the key
        removed = store.remove(entry.session_key)
        assert removed is True

        # Removing again should return False
        removed2 = store.remove(entry.session_key)
        assert removed2 is False
