"""A minted foreign-import id that collides with an existing row writes nothing and leaves that row as it was."""

import pytest

from persistence import PersistenceError, SessionId
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._reference.housekeeping import maintenance
from tests.contracts._scopes import profile_scope


@pytest.mark.parametrize("cwd", ["/kept", ""])
def test_foreign_identifier_collision_preserves_existing_fields(monkeypatch, cwd):
    backend = InMemoryBackend()
    scope = profile_scope("foreign-collision")
    sid = SessionId("20260101_000000_000000000000")
    store = backend.open(scope)
    store.sessions.create(NewSession(sid, "cli", model="kept-model", cwd=cwd, user_id="user", chat_id="chat", session_key="kept-key", model_config={"temperature": 0.4}))
    store.sessions.set_pinned(sid, True)
    before = store.sessions.get(sid)
    monkeypatch.setattr(maintenance, "_mint_session_id", lambda: str(sid))
    origin = ForeignOrigin("claude", "/history", "foreign")
    # The minted id is inserted, so the primary key refuses it and the write rolls back.
    with pytest.raises(PersistenceError):
        store.maintenance.import_foreign_history(origin, (ForeignTurn("user", "imported"),), title="Imported", cwd="/new", profile="p")
    assert store.sessions.get(sid) == before
    assert store.maintenance.find_foreign_import(origin) is None
    backend.close()
