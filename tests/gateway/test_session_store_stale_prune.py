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
from gateway.session import SessionEntry, SessionSource, SessionStore


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


def _make_entry_with_origin(key: str, session_id: str) -> SessionEntry:
    entry = _make_entry(key, session_id)
    entry.origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5140768830",
        chat_type="dm",
        user_id="5140768830",
        user_name="João",
    )
    return entry


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


    def test_keeps_stale_entry_when_recovery_lookup_raises(self, tmp_path):
        """Indeterminate recovery must not delete the only routing handle.

        Startup pruning sees an ended parent and tries to repoint it to the
        latest live gateway child.  If that recovery query raises, deleting the
        sessions.json entry loses the routing key entirely; keeping it lets the
        runtime stale guard retry recovery on the next message.
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({"sid_parent": {"end_reason": "compression", "id": "sid_parent"}})
        db.find_latest_gateway_session_for_peer.side_effect = RuntimeError("db busy")
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_parent"

    def test_noop_when_db_is_none(self, tmp_path):
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries


    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Startup recovery honours the reset policy
# ---------------------------------------------------------------------------

class TestStartupRecoveryResetPolicy:
    """Startup repoint must not resurrect an overdue session as fresh.

    The startup pruner repoints a stale entry to the recovered row via
    ``_recover_session_from_db``. The rebuilt entry used to be stamped
    ``updated_at=now``, so an opt-in idle/daily ``session_reset`` policy was
    silently skipped across every gateway restart. Recovery now evaluates
    ``_should_reset`` against the durable last message timestamp and promotes
    an overdue session to a durable reset boundary instead of reopening it.
    """

    def test_overdue_recovered_session_promoted_to_reset_and_pruned(self, tmp_path):
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning(
            {"sid_parent": {"end_reason": "agent_close", "id": "sid_parent"}}
        )
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_child",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
        }
        db.get_last_activity.return_value = (
            datetime.now() - timedelta(hours=4)
        ).timestamp()
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        with patch.object(store, "_save"):
            store._prune_stale_sessions_locked()

        assert key not in store._entries
        db.promote_to_session_reset.assert_called_once_with("sid_child", "idle")
        db.reopen_session.assert_not_called()

    def test_none_policy_startup_repoint_unchanged(self, tmp_path):
        """mode="none" (the default) still repoints to the recovered row."""
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning(
            {"sid_parent": {"end_reason": "compression", "id": "sid_parent"}}
        )
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_child",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
        }
        last_activity = (datetime.now() - timedelta(hours=4)).timestamp()
        db.get_last_activity.return_value = last_activity
        store = _make_store_with_db(tmp_path, db)  # default mode="none"
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        with patch.object(store, "_save"):
            store._prune_stale_sessions_locked()

        assert store._entries[key].session_id == "sid_child"
        assert store._entries[key].updated_at == datetime.fromtimestamp(
            last_activity
        )
        db.reopen_session.assert_called_once_with("sid_child")
        db.promote_to_session_reset.assert_not_called()


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

