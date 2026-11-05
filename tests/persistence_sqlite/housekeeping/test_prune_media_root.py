"""Every SQLite prune verb records as retired media exactly the hosted documents under the media root its caller
passes, as the legacy store does under the same root."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

import pytest

import gateway.session_ingress_media
from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

OLD = time.time() - 200 * 86400.0
ADMISSION_SQL = (
    "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
    " payload_json, payload_digest, intent, status, outcome)"
    " VALUES (?, ?, 'principal', ?, '[]', ?, 'digest', 'queue', 'terminal', 'completed')"
)
RETIRED_SQL = "SELECT value FROM state_meta WHERE key LIKE 'gateway.media.retired.v1.%'"
# The source and columns that put a fresh row in the shape each verb prunes.
SHAPES = {
    "prune_sessions": ("cli", {"ended_at": OLD + 60, "end_reason": "user_exit"}),
    "prune_empty_ghost_sessions": ("tui", {"ended_at": OLD + 60, "end_reason": "user_exit"}),
    "prune_never_active": ("telegram", {"ended_at": None, "session_key": "agent:main:telegram:dm:1"}),
}
BACKEND_VERBS = {
    "prune_sessions": lambda store, root: store.maintenance.prune_sessions(90, media_root=root).deleted_ids,
    "prune_empty_ghost_sessions": lambda store, root: store.maintenance.prune_empty_ghost_sessions(media_root=root),
    "prune_never_active": lambda store, root: store.routing.prune_never_active(older_than_days=5, media_root=root).deleted_ids,
}
LEGACY_VERBS = {
    "prune_sessions": lambda db: db.prune_sessions(90),
    "prune_empty_ghost_sessions": lambda db: db.prune_empty_ghost_sessions(),
    "prune_never_active": lambda db: db.prune_never_active_keyed_sessions(older_than_days=5),
}


def _document(root, name):
    return str(root / hashlib.sha256(name.encode()).hexdigest() / f"{name}.txt")


def _seed(execute_write, session_id, verb, documents):
    columns = {"started_at": OLD, "last_activity_at": None, "title": None, **SHAPES[verb][1]}
    text = "shared" + "".join(f"\n[Shared attachment] file: {path}\n" for path in documents)
    def _do(conn):
        conn.execute(f"UPDATE sessions SET {', '.join(f'{k} = ?' for k in columns)} WHERE id = ?",
                     (*columns.values(), session_id))
        conn.execute(ADMISSION_SQL, (f"adm-{session_id}", f"hosted:room:{session_id}", session_id,
                                     json.dumps({"text": text})))
    execute_write(_do)


def _retired(path):
    with sqlite3.connect(path) as connection:
        return {json.loads(value)["path"] for (value,) in connection.execute(RETIRED_SQL)}


@pytest.mark.parametrize("verb", sorted(BACKEND_VERBS))
def test_each_media_root_retires_only_the_documents_under_it(backend, scope_a, tmp_path, monkeypatch, legacy_store,
                                                             verb):
    import hermes_state_media
    from hermes_state import SessionDB

    roots = {name: (tmp_path / name / "native-inputs").resolve() for name in ("a", "b", "other")}
    documents = {name: _document(root, f"doc-{name}") for name, root in roots.items()}
    store = backend.open(scope_a)
    db = SessionDB(db_path=tmp_path / "legacy.db")
    # The legacy prune collects retired candidates after its commit; the candidates are what is compared here.
    monkeypatch.setattr(hermes_state_media, "collect_retired_media", lambda db: None)
    try:
        retired = []
        for sid, name in (("for-a", "a"), ("for-b", "b")):
            source = SHAPES[verb][0]
            store.sessions.create(NewSession(sid, source))
            _seed(store.holder()._execute_write, sid, verb, documents.values())
            db.create_session(sid, source)
            _seed(db._execute_write, sid, verb, documents.values())
            assert BACKEND_VERBS[verb](store, str(roots[name])) == (sid,)
            monkeypatch.setattr(gateway.session_ingress_media, "_media_root", lambda root=roots[name]: root)
            LEGACY_VERBS[verb](db)
            assert store.sessions.get(sid) is None and db.get_session(sid) is None
            retired.append(documents[name])
            assert _retired(store_for(scope_a)) == _retired(tmp_path / "legacy.db") == set(retired)
    finally:
        db.close()
