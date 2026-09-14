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
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
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
    config = GatewayConfig()
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
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_parent",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
            "last_activity_at": (
                datetime.now() - timedelta(hours=4)
            ).timestamp(),
        }
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
        db.reopen_session.assert_called_once_with("sid_parent")
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


    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()

    def test_reset_boundary_does_not_recover_older_session_for_peer(self, tmp_path):
        """Startup pruning must not search past an intentional reset boundary.

        The durable recovery query deliberately excludes ``session_reset``
        rows — and a newer reset row must also fence any *older* still-open
        row for the same peer. If startup pruning invokes recovery for a
        routing entry that points at such a row, the query must not return
        an older live session for the same peer and silently restore the
        context that the user reset. Exercise the real SessionDB query here
        rather than mocking its result.
        """
        from hermes_state import SessionDB

        key = "agent:main:telegram:dm:5140768830"
        db = SessionDB(tmp_path / "state.db")
        peer = {
            "user_id": "5140768830",
            "session_key": key,
            "chat_id": "5140768830",
            "chat_type": "dm",
        }
        db.create_session("sid_before_reset", "telegram", **peer)
        db.append_message("sid_before_reset", "user", "private old context")
        db.create_session("sid_reset", "telegram", **peer)
        db.append_message("sid_reset", "user", "/new")
        db.end_session("sid_reset", "session_reset")

        store = _make_store_with_db(tmp_path / "sessions", db)
        stale_entry = _make_entry_with_origin(key, "sid_reset")
        store._entries[key] = stale_entry

        # Model restart startup followed by the peer's first incoming message.
        store._prune_stale_sessions_locked()
        assert stale_entry.origin is not None
        current = store.get_or_create_session(stale_entry.origin)

        assert current.session_id not in {"sid_before_reset", "sid_reset"}
        assert store._entries[key].session_id == current.session_id
        reset_row = db.get_session("sid_reset")
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
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_child",
            "started_at": (datetime.now() - timedelta(hours=5)).timestamp(),
            "last_activity_at": last_activity,
        }
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
        config = GatewayConfig()
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "dm_key" not in store._entries


@pytest.fixture
def compression_crash(tmp_path):
    """Commit real compression rows but leave the primary routing index on the parent."""
    from hermes_state import SessionDB

    stores = []
    root, profile = tmp_path / "root", tmp_path / "closure-profile"
    root.mkdir()
    profile.mkdir()

    def make(*, marker=0, hops=1, named=False, source=None, selected=True):
        config = GatewayConfig(multiplex_profiles=True)
        source = source or SessionSource(platform=Platform.TELEGRAM, chat_id="closure-chat",
                                         user_id="closure-user", chat_type="dm", thread_id="thread")
        source = replace(source, profile="closure" if named else "default")

        def fresh():
            store = SessionStore(tmp_path / "sessions", config)
            store._routing_home = root
            store._profile_home_cache["closure"] = profile
            stores.append(store)
            return store

        store = fresh()
        entry = store.get_or_create_session(source)
        db = store._db_for_key(entry.session_key)
        assert isinstance(db, SessionDB)
        db.append_message(entry.session_id, "user", "before compression")
        store.set_session_metadata(entry.session_key, "unrelated-policy", {"do-not-copy": True})
        if selected:
            store.set_session_metadata(entry.session_key, "manual_fallback_index", marker,
                                       require_primary=True, expected_session_id=entry.session_id)
        parent = entry.session_id
        for hop in range(hops):
            child = f"closure-child-{hop}"
            db.publish_compression_child(
                parent_session_id=parent, child_session_id=child, source=source.platform.value,
                messages=[{"role": "user", "content": f"handoff-{hop}"}],
                require_compression_lease=False,
            )
            assert db.get_session(parent)["end_reason"] == "compression"
            assert db.get_messages(child)[0]["content"] == f"handoff-{hop}"
            parent = child
        routing_db = store._routing_db
        assert isinstance(routing_db, SessionDB)
        raw = routing_db.load_gateway_routing_entries(scope=store._routing_scope())
        assert json.loads(raw[entry.session_key])["session_id"] == entry.session_id
        assert store.lookup_by_session_key(entry.session_key) is entry
        return SimpleNamespace(store=store, db=db, source=source, entry=entry,
                               key=entry.session_key, child=parent, fresh=fresh)

    yield make
    for store in stores:
        store.close_all_db_handles()


def _assert_recovered_twice(e, marker):
    durable_child = e.db.get_session(e.child)
    expected = None
    for _ in range(2):
        store = e.fresh()
        recovered = store.lookup_by_session_key(e.key)
        assert recovered is not None and recovered.session_id == e.child
        assert recovered.metadata == {"manual_fallback_index": marker}
        assert type(recovered.metadata["manual_fallback_index"]) is type(marker)
        assert recovered.created_at == datetime.fromtimestamp(durable_child["started_at"])
        assert recovered.updated_at == datetime.fromtimestamp(
            durable_child["last_activity_at"] or durable_child["started_at"])
        if expected is not None:
            assert recovered.to_dict() == expected
        expected = recovered.to_dict()
        raw = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
        assert json.loads(raw[e.key]) == expected


@pytest.mark.parametrize("marker", [0, None, "0", True, False, {"invalid": 1}])
@pytest.mark.parametrize("hops", [1, 3])
def test_manual_marker_survives_compression_commit_crash_and_two_reloads(compression_crash, marker, hops):
    e = compression_crash(marker=marker, hops=hops)
    assert e.db.get_compression_tip(e.entry.session_id) == e.child
    _assert_recovered_twice(e, marker)


def test_manual_recovery_uses_owning_profile_db(compression_crash):
    e = compression_crash(named=True, hops=2)
    assert e.db is not e.store._routing_db
    assert e.store._routing_db.get_session(e.entry.session_id) is None
    _assert_recovered_twice(e, 0)


@pytest.mark.parametrize("operation", ["tip", "child-row"])
def test_manual_lineage_lookup_error_keeps_original_route(compression_crash, monkeypatch, operation):
    from hermes_state import SessionDB

    e = compression_crash()
    original = e.entry.to_dict()
    with monkeypatch.context() as m:
        if operation == "tip":
            m.setattr(SessionDB, "get_compression_tip", lambda *a: (_ for _ in ()).throw(OSError("busy")))
        else:
            get_session = SessionDB.get_session
            def fail_child(db, session_id):
                if session_id == e.child:
                    raise OSError("busy")
                return get_session(db, session_id)
            m.setattr(SessionDB, "get_session", fail_child)
        for _ in range(2):
            store = e.fresh()
            assert store.lookup_by_session_key(e.key).to_dict() == original
            raw = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
            assert json.loads(raw[e.key]) == original
    _assert_recovered_twice(e, 0)


@pytest.mark.parametrize("column,value", [
    ("session_key", "agent:main:telegram:dm:other"),
    ("user_id", "other-user"), ("chat_id", "other-chat"),
    ("chat_type", "group"), ("thread_id", "other-thread"), ("source", "discord"),
])
def test_manual_marker_never_crosses_durable_parent_peer_mismatch(compression_crash, column, value):
    e = compression_crash()
    # The live child still matches the requested route; only its durable parent disagrees.
    e.db._write_sql(f"UPDATE sessions SET {column} = ? WHERE id = ?", (value, e.entry.session_id))
    assert e.db.get_compression_tip(e.entry.session_id) == e.child
    for _ in range(2):
        recovered = e.fresh().lookup_by_session_key(e.key)
        assert recovered.session_id == e.child
        assert recovered.metadata == {}


@pytest.mark.parametrize("kind", ["unrelated", "branch", "delegate", "tool", "foreign-user", "foreign-profile", "non-compression", "ordinary"])
def test_manual_marker_not_copied_to_unrelated_native_recovery(compression_crash, kind):
    e = compression_crash(selected=kind != "ordinary")
    if kind == "unrelated":
        e.db.create_session("unrelated-child", e.source.platform.value, session_key=e.key,
                            user_id=e.source.user_id, chat_id=e.source.chat_id,
                            chat_type=e.source.chat_type, thread_id=e.source.thread_id)
        e.db.append_message("unrelated-child", "user", "newer unrelated conversation")
        expected_id = "unrelated-child"
    else:
        expected_id = e.child
        if kind in {"branch", "delegate"}:
            flag = "_branched_from" if kind == "branch" else "_delegate_from"
            e.db.update_session_meta(e.child, json.dumps({flag: e.entry.session_id}))
        elif kind == "tool":
            e.db._write_sql("UPDATE sessions SET source = ? WHERE id = ?", ("tool", e.child))
            expected_id = None
        elif kind == "foreign-user":
            e.db._write_sql("UPDATE sessions SET user_id = ? WHERE id = ?", ("foreign", e.child))
        elif kind == "foreign-profile":
            e.db._write_sql("UPDATE sessions SET session_key = ? WHERE id = ?",
                            ("agent:foreign:telegram:dm:closure-chat", e.child))
            expected_id = None
        elif kind == "non-compression":
            e.db._write_sql("UPDATE sessions SET end_reason = ? WHERE id = ?", ("agent_close", e.entry.session_id))
    for _ in range(2):
        recovered = e.fresh().lookup_by_session_key(e.key)
        if expected_id is None:
            assert recovered is None
        else:
            assert recovered is not None and recovered.session_id == expected_id
            assert recovered.metadata == {}


@pytest.mark.parametrize("scope", ["other-workspace", None])
def test_manual_recovery_preserves_native_workspace_gate(compression_crash, scope):
    source = SessionSource(platform=Platform.SLACK, chat_id="channel", chat_type="group",
                           user_id="user", scope_id="original-workspace")
    e = compression_crash(source=source)
    origin = dict(source.to_dict(), scope_id=scope)
    e.db._write_sql("UPDATE sessions SET origin_json = ? WHERE id = ?", (json.dumps(origin), e.child))
    assert e.db.get_compression_tip(e.entry.session_id) == e.child
    for _ in range(2):
        assert e.fresh().lookup_by_session_key(e.key) is None
