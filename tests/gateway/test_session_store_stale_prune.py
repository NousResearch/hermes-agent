"""Tests for SessionStore._prune_stale_sessions_locked — crash self-healing.

When a gateway crashes (exit code 1) the graceful shutdown path is skipped and
sessions.json is left pointing at sessions already ended in state.db. On the
next startup _ensure_loaded_locked calls _prune_stale_sessions_locked to detect
and remove those stale routing entries before get_or_create_session() can reuse
them and silently route incoming messages into a closed session (#52804).
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


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
    """Build a SessionStore pinned to a mock store, bypassing disk load."""
    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    if isinstance(db_mock, MagicMock):
        store._persistence_beside = lambda db: db  # the mock is the storage interface handle too
    store._loaded = True
    return store


def _db_returning(rows: dict) -> MagicMock:
    """Storage handle mock where sessions.get maps session_id -> a record of the row's columns (no stored
    origin or chat, as on these rows)."""
    db = MagicMock()
    db.sessions.get.side_effect = (
        lambda sid: SimpleNamespace(origin=None, chat_id=None, **rows[sid]) if sid in rows else None)
    db.routing.route_retired.return_value = False
    return db


def _peer(row: dict):
    """The ``routing.latest_session_for_peer`` answer for a recovered row."""
    session = SimpleNamespace(session_id=row.pop("id"), session_key=None, message_count=0, origin=None, **row)
    return SimpleNamespace(session=session, has_messages=True)


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
        db.routing.latest_session_for_peer.side_effect = RuntimeError("db busy")
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_parent"

    def test_keeps_stale_entry_when_recovery_returns_same_session_id(self, tmp_path):
        """A successful same-id recovery must NOT prune the routing entry.

        When the startup sweep finds a stale entry whose session has ended in
        state.db but ``_recover_session_from_db`` succeeds and returns the SAME
        session id (proving the route is still resumable — the ``!=`` repoint
        guard only exists for the compression-rotation child case), the entry
        must be kept in place. The old code fell through to the prune branch
        whenever the recovered id did not differ, deleting a perfectly valid
        resumable mapping (#95957).
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning(
            {"sid_parent": {"end_reason": "agent_close", "id": "sid_parent"}}
        )
        # Recovery returns the row for sid_parent itself — same id as the entry.
        db.routing.latest_session_for_peer.return_value = _peer({
            "id": "sid_parent",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
            "last_activity_at": (datetime.now() - timedelta(hours=4)).timestamp(),
        })
        store = _make_store_with_db(tmp_path, db)  # default mode="none"
        original_entry = _make_entry_with_origin(key, "sid_parent")
        original_entry.model_override = {"model": "custom/model", "provider": "openrouter"}
        original_entry.resume_pending = True
        store._entries[key] = original_entry

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()

        # The successfully-recovered route must survive the sweep.
        assert key in store._entries
        assert store._entries[key].session_id == "sid_parent"
        # The ORIGINAL entry object is kept — a rebuilt entry would silently
        # drop live state (model_override, resume_pending, token counters).
        assert store._entries[key] is original_entry
        assert store._entries[key].model_override == {
            "model": "custom/model", "provider": "openrouter"
        }
        assert store._entries[key].resume_pending is True
        # The row is reopened in state.db by recovery.
        db.sessions.reopen.assert_called_once_with("sid_parent")
        # Nothing in sessions.json changed, so no rewrite is needed.
        mock_save.assert_not_called()

    def test_noop_when_db_is_none(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries

    def test_reset_boundary_does_not_recover_older_session_for_peer(self, tmp_path, monkeypatch):
        """Startup pruning must not search past an intentional reset boundary.

        The durable recovery query deliberately excludes ``session_reset``
        rows — and a newer reset row must also fence any *older* still-open
        row for the same peer. If startup pruning invokes recovery for a
        routing entry that points at such a row, the query must not return
        an older live session for the same peer and silently restore the
        context that the user reset. Exercise the real store query here
        rather than mocking its result.
        """
        key = "agent:main:telegram:dm:5140768830"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        db = storage_handle(tmp_path)
        peer = {
            "user_id": "5140768830",
            "session_key": key,
            "chat_id": "5140768830",
            "chat_type": "dm",
        }
        seed_session(db, "sid_before_reset", "telegram", **peer)
        seed_messages(db, "sid_before_reset", {"role": "user", "content": "private old context"})
        seed_session(db, "sid_reset", "telegram", **peer)
        seed_messages(db, "sid_reset", {"role": "user", "content": "/new"})
        db.sessions.end("sid_reset", "session_reset")

        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        store._loaded = True
        stale_entry = _make_entry_with_origin(key, "sid_reset")
        store._entries[key] = stale_entry

        # Model restart startup followed by the peer's first incoming message.
        store._prune_stale_sessions_locked()
        assert stale_entry.origin is not None
        current = store.get_or_create_session(stale_entry.origin)

        assert current.session_id not in {"sid_before_reset", "sid_reset"}
        assert store._entries[key].session_id == current.session_id
        reset_row = session_row(db, "sid_reset")
        assert reset_row is not None
        assert reset_row["end_reason"] == "session_reset"


# ---------------------------------------------------------------------------
# Startup recovery honours the reset policy
# ---------------------------------------------------------------------------

class TestStartupRecovery:
    """Recovery keeps the durable session and activity timestamp."""

    def test_startup_repoint_unchanged(self, tmp_path):
        """Startup repoints to the recovered row."""
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning(
            {"sid_parent": {"end_reason": "compression", "id": "sid_parent"}}
        )
        last_activity = (datetime.now() - timedelta(hours=4)).timestamp()
        db.routing.latest_session_for_peer.return_value = _peer({
            "id": "sid_child",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
            "last_activity_at": last_activity,
        })
        store = _make_store_with_db(tmp_path, db)  # default mode="none"
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        with patch.object(store, "_save"):
            store._prune_stale_sessions_locked()

        assert store._entries[key].session_id == "sid_child"
        assert store._entries[key].updated_at == datetime.fromtimestamp(
            last_activity
        )
        db.sessions.reopen.assert_called_once_with("sid_child")
        db.sessions.promote_to_reset.assert_not_called()


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
        config = GatewayConfig()
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db, store._persistence_beside = db, lambda pinned: pinned
        store._ensure_loaded()

        assert "dm_key" not in store._entries
