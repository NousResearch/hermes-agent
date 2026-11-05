"""Every SQLite verb that retires sessions records as retired media exactly the hosted documents under the media root
its caller passes, as the legacy store does under the same root."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import gateway.session_ingress_media
from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

ADMISSION_SQL = (
    "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
    " payload_json, payload_digest, intent, status, outcome)"
    " VALUES (?, ?, 'principal', ?, '[]', ?, 'digest', 'queue', 'terminal', 'completed')"
)
RETIRED_SQL = "SELECT value FROM state_meta WHERE key LIKE 'gateway.media.retired.v1.%'"


def _document(root, name):
    return str(root / hashlib.sha256(name.encode()).hexdigest() / f"{name}.txt")


def _admit(execute_write, session_id, documents):
    text = "shared" + "".join(f"\n[Shared attachment] file: {path}\n" for path in documents)
    execute_write(lambda conn: conn.execute(
        ADMISSION_SQL, (f"adm-{session_id}", f"hosted:room:{session_id}", session_id, json.dumps({"text": text}))))


def _retired(path):
    with sqlite3.connect(path) as connection:
        return {json.loads(value)["path"] for (value,) in connection.execute(RETIRED_SQL)}


BACKEND_VERBS = {
    "delete": lambda sessions, sid, root: sessions.delete(sid, media_root=root),
    "delete_many": lambda sessions, sid, root: sessions.delete_many([sid], media_root=root),
    "delete_if_empty": lambda sessions, sid, root: sessions.delete_if_empty(sid, media_root=root),
    "delete_empty": lambda sessions, sid, root: (sessions.end(sid, "cli_close"), sessions.delete_empty(media_root=root)),
    "discard_unadmitted": lambda sessions, sid, root: sessions.discard_unadmitted(sid, media_root=root),
}
LEGACY_VERBS = {
    "delete": lambda db, sid: db.delete_session(sid),
    "delete_many": lambda db, sid: db.delete_sessions([sid]),
    "delete_if_empty": lambda db, sid: db.delete_session_if_empty(sid),
    "delete_empty": lambda db, sid: (db.end_session(sid, "cli_close"), db.delete_empty_sessions()),
    "discard_unadmitted": lambda db, sid: db.discard_unadmitted_session(sid),
}


@pytest.mark.parametrize("verb", sorted(BACKEND_VERBS))
def test_each_media_root_retires_only_the_documents_under_it(backend, scope_a, tmp_path, monkeypatch, legacy_store,
                                                             verb):
    import hermes_state_media
    from hermes_state import SessionDB

    roots = {name: (tmp_path / name / "native-inputs").resolve() for name in ("a", "b", "other")}
    documents = {name: _document(root, f"doc-{name}") for name, root in roots.items()}
    sessions = backend.open(scope_a).sessions
    db = SessionDB(db_path=tmp_path / "legacy.db")
    # The legacy delete collects retired candidates after its commit; the candidates are what is compared here.
    monkeypatch.setattr(hermes_state_media, "collect_retired_media", lambda db: None)
    try:
        for sid in ("for-a", "for-b"):
            sessions.create(NewSession(sid, "cli"))
            _admit(backend.open(scope_a).holder()._execute_write, sid, documents.values())
            db.create_session(sid, "cli")
            _admit(db._execute_write, sid, documents.values())
        retired = []
        for sid, name in (("for-a", "a"), ("for-b", "b")):
            BACKEND_VERBS[verb](sessions, sid, str(roots[name]))
            monkeypatch.setattr(gateway.session_ingress_media, "_media_root", lambda root=roots[name]: root)
            LEGACY_VERBS[verb](db, sid)
            assert sessions.get(sid) is None and db.get_session(sid) is None
            retired.append(documents[name])
            assert _retired(store_for(scope_a)) == _retired(tmp_path / "legacy.db") == set(retired)
    finally:
        db.close()
