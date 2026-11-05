"""The bulk delete the never-active prune runs (``SessionRepository.delete_many``) against the legacy
``SessionDB.delete_sessions``: each case seeds one store with ``SessionDB``, copies it, deletes on both copies and
compares the results and the tables."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from persistence import AuthorityRefusal
from persistence.conversation.sessions import DeleteManyOutcome
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

NOW = time.time()
MEDIA_ROOT = "/nonexistent/native-inputs"
_TABLES = (
    "SELECT id, parent_session_id FROM sessions",
    "SELECT session_id, content FROM messages",
    "SELECT hash FROM system_prompts",
    "SELECT key FROM state_meta",
    "SELECT session_id, holder FROM compression_locks",
    "SELECT conversation_id, holder FROM session_turn_leases",
)


def _sql(db, sql, params=()):
    db._execute_write(lambda conn: conn.execute(sql, params))


def _session(db, sid, parent=None, *, delegate=False, prompt=None):
    db.create_session(sid, "cli", parent_session_id=parent,
                      model_config={"_delegate_from": parent} if delegate else None, system_prompt=prompt)
    _sql(db, "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', ?, ?)", (sid, sid, NOW))


def _guard(db, table, key, expires):
    column = "session_id" if table == "compression_locks" else "conversation_id"
    _sql(db, f"INSERT INTO {table} ({column}, holder, acquired_at, expires_at) VALUES (?, 'other', ?, ?)",
         (key, NOW - 60, expires))


def _seed(db):
    _session(db, "parent", prompt="shared prompt")
    _session(db, "child", "parent", delegate=True)
    _session(db, "grand", "child", delegate=True)
    _session(db, "untagged", "parent")
    _session(db, "guarded")
    _guard(db, "session_turn_leases", "guarded", NOW + 3600)
    _session(db, "guarded-parent")
    _session(db, "guarded-child", "guarded-parent", delegate=True)
    _guard(db, "compression_locks", "guarded-child", NOW + 3600)
    _session(db, "expired")
    _guard(db, "session_turn_leases", "expired", NOW - 10)
    _session(db, "kept", prompt="shared prompt")


def _rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return sorted(conn.execute(sql).fetchall(), key=repr)
    finally:
        conn.close()


@pytest.fixture
def pair(tmp_path, backend, scope_a, legacy_store):
    """A legacy ``SessionDB`` and the backend handle, each over its own copy of the seeded store."""
    from hermes_state import SessionDB

    seed_path, legacy_path = tmp_path / "seed.db", tmp_path / "legacy.db"
    db = SessionDB(db_path=seed_path)
    _seed(db)
    db.close()
    for target in (legacy_path, store_for(scope_a)):
        source, copy = sqlite3.connect(seed_path), sqlite3.connect(target)
        source.backup(copy)
        source.close()
        copy.close()
    legacy = SessionDB(db_path=legacy_path)
    yield legacy, backend.open(scope_a)
    legacy.close()


def _assert_same_tables(legacy, store):
    for sql in _TABLES:
        assert _rows(legacy.db_path, sql) == _rows(store.store_path, sql), sql


@pytest.mark.parametrize("exclude_guards", [True, False])
def test_delete_sessions_matches_legacy(pair, exclude_guards):
    legacy, store = pair
    ids = ["parent", "guarded", "guarded-parent", "expired", "missing", "parent", "", None]
    legacy_skipped: list[str] = []
    expected = legacy.delete_sessions(ids, exclude_active_write_guards=exclude_guards, skipped_ids=legacy_skipped)
    outcome = store.sessions.delete_many(ids, skip_guarded=exclude_guards, media_root=MEDIA_ROOT)
    count, skipped, removed = outcome.deleted, list(outcome.skipped_guarded), outcome.removed_ids
    assert count == expected
    assert skipped == legacy_skipped == (["guarded", "guarded-parent"] if exclude_guards else [])
    deleted = {"parent", "child", "grand", "expired"} | (set() if exclude_guards else {
        "guarded", "guarded-parent", "guarded-child"})
    assert set(removed) == deleted and len(removed) == len(deleted)
    _assert_same_tables(legacy, store)
    # Untagged children are detached, not cascaded; the shared prompt survives with its other user.
    assert _rows(store.store_path, "SELECT parent_session_id FROM sessions WHERE id = 'untagged'") == [(None,)]
    assert _rows(store.store_path, "SELECT COUNT(*) FROM system_prompts") == [(1,)]


def test_delete_sessions_without_known_ids_writes_nothing(pair):
    legacy, store = pair
    assert store.sessions.delete_many([], media_root=MEDIA_ROOT) == DeleteManyOutcome(deleted=0, skipped_guarded=(), removed_ids=())
    assert store.sessions.delete_many(["missing"], media_root=MEDIA_ROOT) == DeleteManyOutcome(deleted=0, skipped_guarded=(), removed_ids=())
    assert legacy.delete_sessions(["missing"]) == 0
    _assert_same_tables(legacy, store)


def test_delete_sessions_refuses_live_ledger_work_as_legacy(pair):
    from hermes_state_runtime import RuntimeStoreError

    legacy, store = pair
    for db_holder in (legacy, store.holder()):
        db_holder._execute_write(lambda conn: conn.execute(
            "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id,"
            " lineage_json, payload_json, payload_digest, intent, status)"
            " VALUES ('a', 'r', 'p', 'kept', '[]', ?, 'd', 'queue', 'queued')", (json.dumps({}),)))
    with pytest.raises(RuntimeStoreError) as legacy_refusal:
        legacy.delete_sessions(["kept"])
    with pytest.raises(AuthorityRefusal) as refusal:
        store.sessions.delete_many(["kept"], media_root=MEDIA_ROOT)
    assert refusal.value.reason == legacy_refusal.value.reason == "session_busy"
    assert (refusal.value.field, refusal.value.detail) == (legacy_refusal.value.field, legacy_refusal.value.detail)
    _assert_same_tables(legacy, store)
