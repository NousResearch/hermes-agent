"""``discard_unadmitted`` on SQLite returns the ids it removed: the row itself when nothing was admitted against it, what
the fenced delete removed when the row has an admission."""

from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession, SessionLineageKind

MEDIA_ROOT = "/nonexistent/native-inputs"


def test_unadmitted_row_returns_its_own_id_then_nothing(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("seed", "cli"))
    store.messages.append("seed", NewMessage("user", "seed copy"))
    assert store.sessions.discard_unadmitted("seed", media_root=MEDIA_ROOT) == ("seed",)
    assert store.sessions.get("seed") is None
    assert store.messages.read_transcript("seed", TranscriptRead(view=TranscriptView.ALL)) == ()
    assert store.sessions.discard_unadmitted("seed", media_root=MEDIA_ROOT) == ()
    assert store.sessions.create(NewSession("seed", "cli")) == "seed"
    assert store.sessions.discard_unadmitted("ghost", media_root=MEDIA_ROOT) == ()


def test_admitted_row_returns_what_the_fenced_delete_removed(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("real", "cli"))
    store.sessions.create(NewSession("real-child", "cli", parent_session_id="real",
                                     lineage_kind=SessionLineageKind.DELEGATE))
    store.holder()._execute_write(lambda conn: conn.execute(
        "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
        " payload_json, payload_digest, intent, status, outcome)"
        " VALUES ('adm-1', 'req-1', 'principal', 'real', '[]', '{}', 'digest', 'queue', 'terminal', 'completed')"))
    assert store.sessions.discard_unadmitted("real", media_root=MEDIA_ROOT) == ("real", "real-child")
    assert store.sessions.get("real") is None and store.sessions.get("real-child") is None
