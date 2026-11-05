"""``SQLiteProfileIdentityRepository`` against the legacy ``SessionDB.rekey_profile_state`` /
``purge_profile_state``: each case seeds one store with ``SessionDB``, copies it, runs both and compares the counts
and the tables."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from persistence.housekeeping.profile_identity import ProfileIdentityCollision
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

NOW = time.time()
_TABLES = {
    "sessions": "SELECT id, profile_name, session_key, origin_json FROM sessions",
    "gateway_heartbeats": "SELECT backend_id, profile FROM gateway_heartbeats",
    "gateway_routing": "SELECT scope, session_key, entry_json FROM gateway_routing",
    "delivery_obligations": "SELECT obligation_id, session_key, state, adapter_profile FROM delivery_obligations",
    "telegram_dm_topic_mode": "SELECT profile_name, chat_id, user_id FROM telegram_dm_topic_mode",
    "telegram_dm_topic_bindings": "SELECT profile_name, chat_id, thread_id, session_key, session_id"
                                  " FROM telegram_dm_topic_bindings",
}
_LEDGER_DDL = (
    "CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL,"
    " platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT, content TEXT NOT NULL, state TEXT NOT NULL,"
    " attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL, owner_pid INTEGER,"
    " owner_started_at INTEGER, last_error TEXT{adapter})"
)


def _sql(db, sql, params=()):
    db._execute_write(lambda conn: conn.execute(sql, params))


def _seed_identity(db, *, ledger="adapter", topics=True):
    """Rows of profile ``old`` (and a bystander) in every identity shape; ``ledger`` None/"legacy"/"adapter"."""
    for sid, profile, key, origin in (
        ("s-old", "old", "agent:old:telegram:dm:1", {"profile": "old", "chat": "1"}),
        ("s-other", "other", "agent:other:telegram:dm:2", {"profile": "other"}),
        ("s-prefix", "older", "agent:older:x", None),
    ):
        db.create_session(sid, "telegram", profile_name=profile, session_key=key,
                          origin_json=json.dumps(origin) if origin else None)
    _sql(db, "UPDATE sessions SET origin_json = 'not json' WHERE id = 's-prefix'")
    db.register_backend_heartbeat(backend_id="gw-old", pid=1, started_at=NOW, last_heartbeat=NOW, profile="old")
    db.register_backend_heartbeat(backend_id="gw-other", pid=2, started_at=NOW, last_heartbeat=NOW, profile="other")
    for scope, key, entry in (
        ("", "agent:old:telegram:dm:1", {"session_key": "agent:old:telegram:dm:1", "origin": {"profile": "old"}}),
        ("hosted", "agent:old:discord:3", "raw text"),
        ("", "agent:other:telegram:dm:2", {"session_key": "agent:other:telegram:dm:2"}),
    ):
        _sql(db, "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES (?, ?, ?, ?)",
             (scope, key, entry if isinstance(entry, str) else json.dumps(entry), NOW))
    if ledger:
        _sql(db, _LEDGER_DDL.format(adapter=", adapter_profile TEXT" if ledger == "adapter" else ""))
        for oid, key, state in (("o1", "agent:old:telegram:dm:1", "pending"), ("o2", "agent:old:x", "delivered"),
                                ("o3", "agent:other:y", "pending")):
            _sql(db, "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id, content, state,"
                     " created_at, updated_at) VALUES (?, ?, 'telegram', '1', 'c', ?, ?, ?)", (oid, key, state, NOW, NOW))
        if ledger == "adapter":
            _sql(db, "UPDATE delivery_obligations SET adapter_profile = 'old' WHERE obligation_id = 'o3'")
    if topics:
        db.apply_telegram_topic_migration()
        _sql(db, "INSERT INTO telegram_dm_topic_mode (profile_name, chat_id, user_id, activated_at, updated_at)"
                 " VALUES ('old', '1', 'u', ?, ?)", (NOW, NOW))
        _sql(db, "INSERT INTO telegram_dm_topic_bindings (profile_name, chat_id, thread_id, user_id, session_key,"
                 " session_id, linked_at, updated_at) VALUES ('old', '1', '7', 'u', 'agent:old:telegram:dm:1',"
                 " 's-old', ?, ?)", (NOW, NOW))


def _rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return sorted(conn.execute(sql).fetchall(), key=repr)
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        conn.close()


@pytest.fixture
def pair(tmp_path, backend, scope_a, legacy_store):
    """``pair(seed)``: a legacy ``SessionDB`` and the backend handle, each over its own copy of the seeded store."""
    from hermes_state import SessionDB

    opened = []

    def build(seed):
        seed_path, legacy_path = tmp_path / "seed.db", tmp_path / "legacy.db"
        db = SessionDB(db_path=seed_path)
        seed(db)
        db.close()
        for target in (legacy_path, store_for(scope_a)):
            source, copy = sqlite3.connect(seed_path), sqlite3.connect(target)
            source.backup(copy)
            source.close()
            copy.close()
        opened.append(SessionDB(db_path=legacy_path))
        return opened[-1], backend.open(scope_a)

    yield build
    for db in opened:
        db.close()


def _assert_same_tables(legacy, store):
    for table, sql in _TABLES.items():
        assert _rows(legacy.db_path, sql) == _rows(store.store_path, sql), table


_SHAPES = [("adapter", True), ("legacy", True), (None, False)]


@pytest.mark.parametrize(("ledger", "topics"), _SHAPES)
def test_rekey_matches_legacy(pair, ledger, topics):
    legacy, store = pair(lambda db: _seed_identity(db, ledger=ledger, topics=topics))
    counts = store.profile_identity.rekey_profile(" old ", "new")
    assert counts.as_dict() == legacy.rekey_profile_state(" old ", "new")
    assert counts.sessions_session_key == 1 and counts.gateway_routing == 2
    assert (counts.delivery_obligations_adapter_profile is None) is (ledger != "adapter")
    assert (counts.telegram_dm_topic_mode_profile_name is None) is (not topics)
    _assert_same_tables(legacy, store)
    again = store.profile_identity.rekey_profile("old", "new")
    assert again.as_dict() == legacy.rekey_profile_state("old", "new")
    assert not any(again.as_dict().values())
    for old, new in (("", "new"), ("old", " "), ("same", " same ")):
        assert store.profile_identity.rekey_profile(old, new).as_dict() == legacy.rekey_profile_state(old, new) == {}


@pytest.mark.parametrize("collision", ["routing", "topic_mode", "topic_binding"])
def test_rekey_collision_refuses_before_any_write(pair, collision):
    def seed(db):
        _seed_identity(db)
        if collision == "routing":
            _sql(db, "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)"
                     " VALUES ('hosted', 'agent:new:discord:3', '{}', ?)", (NOW,))
        elif collision == "topic_mode":
            _sql(db, "INSERT INTO telegram_dm_topic_mode (profile_name, chat_id, user_id, activated_at, updated_at)"
                     " VALUES ('new', '1', 'u2', ?, ?)", (NOW, NOW))
        else:
            _sql(db, "INSERT INTO telegram_dm_topic_bindings (profile_name, chat_id, thread_id, user_id, session_key,"
                     " session_id, linked_at, updated_at) VALUES ('new', '1', '7', 'u', 'agent:new:z', 's-other', ?, ?)",
                 (NOW, NOW))
    legacy, store = pair(seed)
    with pytest.raises(ValueError) as legacy_refusal:
        legacy.rekey_profile_state("old", "new")
    with pytest.raises(ProfileIdentityCollision) as refusal:
        store.profile_identity.rekey_profile("old", "new")
    assert str(refusal.value) == str(legacy_refusal.value)
    assert refusal.value.entity_id == "new"
    _assert_same_tables(legacy, store)
    assert _rows(store.store_path, "SELECT profile_name FROM sessions WHERE id = 's-old'") == [("old",)]


@pytest.mark.parametrize(("ledger", "topics"), _SHAPES)
def test_purge_matches_legacy(pair, ledger, topics):
    legacy, store = pair(lambda db: _seed_identity(db, ledger=ledger, topics=topics))
    counts = store.profile_identity.purge_profile(" old ")
    assert counts.as_dict() == legacy.purge_profile_state(" old ")
    assert counts.gateway_routing == 2 and counts.gateway_heartbeats == 1
    assert counts.delivery_obligations == (None if ledger is None else (2 if ledger == "adapter" else 1))
    assert (counts.telegram_dm_topic_bindings is None) is (not topics)
    _assert_same_tables(legacy, store)
    # Sessions are never touched; delivered history stays delivered.
    assert _rows(store.store_path, "SELECT COUNT(*) FROM sessions WHERE profile_name = 'old'") == [(1,)]
    again = store.profile_identity.purge_profile("old")
    assert again.as_dict() == legacy.purge_profile_state("old")
    assert not any(again.as_dict().values())
    assert store.profile_identity.purge_profile("  ").as_dict() == legacy.purge_profile_state("  ") == {}
