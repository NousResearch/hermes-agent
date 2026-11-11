"""A failed portable or foreign import publishes neither sessions nor transcripts."""

import pytest

from persistence import SessionId
from persistence.conversation.sessions import SessionQuery
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._reference.conversation import sessions
from tests.contracts._reference.housekeeping import maintenance
from tests.contracts._scopes import profile_scope

NOW = 1767225600.0


def _payload(sid: str, text: str) -> dict:
    return {"id": sid, "source": "cli", "started_at": NOW, "messages": [{"role": "user", "content": text, "timestamp": NOW}]}


def _contents(store, sid) -> list:
    exported = store.maintenance.export_session(SessionId(sid))
    return [] if exported is None else [message.content for message in exported.messages]


def test_portable_import_failure_leaves_batch_retryable(monkeypatch):
    backend = InMemoryBackend()
    store = backend.open(profile_scope("import-test"))
    payloads = [_payload(sid, sid) for sid in ("first", "second")]
    original = sessions.blank_session_row

    def fail_second(sid, **kwargs):
        if sid == "second":
            raise RuntimeError("second row insertion failed")
        return original(sid, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(sessions, "blank_session_row", fail_second)
        with pytest.raises(RuntimeError, match="^second row insertion failed$"):
            store.maintenance.import_sessions(payloads)
    for sid in ("first", "second"):
        assert store.sessions.get(SessionId(sid)) is None
        assert store.maintenance.assert_export_safe(SessionId(sid), 10) == 0
    result = store.maintenance.import_sessions(payloads)
    assert (result.imported, result.skipped, result.detached) == (2, 0, 0)
    assert store.maintenance.import_sessions(payloads).skipped == 2
    assert [_contents(store, sid) for sid in ("first", "second")] == [["first"], ["second"]]
    backend.close()


@pytest.mark.parametrize("fail_after", [0, 1])
def test_foreign_import_failure_publishes_no_rows_or_counters(monkeypatch, fail_after):
    backend = InMemoryBackend()
    store = backend.open(profile_scope("foreign-atomic"))
    store.maintenance.import_sessions([_payload("existing", "control")])
    before = store.maintenance.export_session(SessionId("existing"))
    origin = ForeignOrigin(tool="claude", path="/history", foreign_session_id="")
    turns = (ForeignTurn("user", "first"), ForeignTurn("assistant", "second"))
    original = maintenance._imported_message
    calls = 0

    def fail_message(message):
        nonlocal calls
        if calls == fail_after:
            raise RuntimeError("transcript construction failed")
        calls += 1
        return original(message)

    with monkeypatch.context() as patcher:
        patcher.setattr(maintenance, "_imported_message", fail_message)
        with pytest.raises(RuntimeError, match="^transcript construction failed$"):
            store.maintenance.import_foreign_history(origin, turns, title="Imported", cwd=None, profile="p")
    assert [entry.session.session_id for entry in store.sessions.list_sessions(SessionQuery())] == ["existing"]
    assert store.maintenance.export_session(SessionId("existing")) == before
    assert store.maintenance.find_foreign_import(origin) is None
    imported = store.maintenance.import_foreign_history(origin, turns, title="Imported", cwd=None, profile="p")
    assert not imported.already_imported
    record = store.sessions.get(imported.session_id)
    assert (record.message_count, record.tool_call_count) == (2, 0)
    replay = store.maintenance.import_foreign_history(origin, turns, title="Imported", cwd=None, profile="p")
    assert replay.already_imported and replay.session_id == imported.session_id
    backend.close()
