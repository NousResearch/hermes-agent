"""Import timestamp boundaries and the foreign import's literal origin and clock."""

import pytest

from persistence import SessionId
from persistence.conversation.sessions import SessionQuery
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._scopes import profile_scope


@pytest.mark.parametrize("epoch", [-2, -1, -0.000001, 0, 0.000001, 1, 981173106.123456, 4200000000, 4200000000.000001, 4200000001])
def test_import_timestamp_trust_window(backend, scope_a, monkeypatch, epoch):
    store_a = backend.open(scope_a)
    monkeypatch.setattr("time.time", lambda: 1780000000.0)
    sid = SessionId("import-clock")
    payload = {"id": sid, "source": "cli", "started_at": epoch, "messages": []}
    result = store_a.maintenance.import_sessions([payload])
    # A start inside [0, 4200000000] is accepted; one outside it refuses the whole payload.
    if 0 <= epoch <= 4200000000:
        assert result.ok and result.imported == 1
        assert store_a.sessions.get(sid).started_at == float(epoch)
        assert store_a.maintenance.import_sessions([payload]).skipped == 1
    else:
        assert not result.ok
        assert result.errors == ({"index": 0, "error": "session.started_at must be a supported Unix timestamp", "session_id": sid},)
        assert store_a.sessions.get(sid) is None


@pytest.mark.parametrize("foreign_id", [None, "", "foreign", 42, "  ", [], 0, False])
def test_reference_foreign_import_keeps_literal_origin_and_clock(monkeypatch, foreign_id):
    backend = InMemoryBackend()
    store = backend.open(profile_scope("foreign-import"))
    now = 1800000000.0
    monkeypatch.setattr("time.time", lambda: now)
    origin = ForeignOrigin(tool="claude", path="/history", foreign_session_id=foreign_id)
    turns = (ForeignTurn("user", "first"), ForeignTurn("assistant", "second"), ForeignTurn("user", "third"))
    result = store.maintenance.import_foreign_history(origin, turns, title="Imported", cwd=None, profile="p")
    exported = store.maintenance.export_session(result.session_id)
    # Turns carry no timestamps, so the import stamps the clock, strictly increasing by a microsecond.
    assert [m.timestamp for m in exported.messages] == pytest.approx([now + index * 0.000001 for index in range(3)], abs=1e-7)
    assert [m.content for m in exported.messages] == ["first", "second", "third"]
    assert exported.session.started_at == now
    assert (exported.session.last_activity_at, exported.session.last_activity_description, exported.session.last_activity_provenance) == (None, None, None)
    [entry] = store.sessions.list_sessions(SessionQuery())
    assert entry.session.origin == {"imported_from": {"tool": "claude", "path": "/history", "foreign_session_id": foreign_id}}
    assert entry.session.created_source is None
    replay = store.maintenance.import_foreign_history(origin, turns, title="Imported", cwd=None, profile="p")
    assert replay.session_id == result.session_id and replay.already_imported
    backend.close()


@pytest.mark.parametrize("tool,title,cwd,field", [("claude", 42, None, "title"), ("claude", "title", 42, "cwd"), (42, "title", None, "source")])
def test_foreign_import_rejects_nontext_session_fields(backend, scope_a, tool, title, cwd, field):
    store_a = backend.open(scope_a)
    origin = ForeignOrigin(tool=tool, path="/history")
    with pytest.raises(ValueError, match=f"^{field} must be a string$"):
        store_a.maintenance.import_foreign_history(origin, (ForeignTurn("user", "first"),), title=title, cwd=cwd, profile="p")
    assert store_a.maintenance.find_foreign_import(origin) is None
