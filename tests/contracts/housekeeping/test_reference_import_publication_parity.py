"""Import publication preserves concurrent message rows and their allocated ids."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.housekeeping import maintenance
from tests.contracts._scopes import profile_scope

NOW = 1767225600.0


@pytest.mark.parametrize("kind", ["portable", "foreign"])
@pytest.mark.parametrize("fail_at", [None, "payload", "publish"])
def test_import_allocates_ids_without_replacing_concurrent_rows(monkeypatch, kind, fail_at):
    backend = InMemoryBackend()
    scope = profile_scope("import-publication")
    backend.open(scope).sessions.create(NewSession("existing", "cli"))
    original_id = backend.open(scope).messages.append("existing", NewMessage("user", "original", timestamp=NOW))
    origin = ForeignOrigin("claude", "/history", "foreign")
    entered, release, attempted, completed = (threading.Event() for _ in range(4))
    original = maintenance._imported_message

    def paused(message):
        if message.get("content") == "second":
            entered.set()
            assert release.wait(10), "import gate timed out"
            if fail_at == "payload":
                raise RuntimeError("import payload failed")
        return original(message)

    def import_history():
        if kind == "portable":
            messages = [{"role": "user", "content": text, "timestamp": NOW} for text in ("first", "second")]
            result = backend.open(scope).maintenance.import_sessions([{"id": "imported", "source": "cli", "started_at": NOW, "messages": messages}])
            assert (result.imported, result.skipped, result.detached) == (1, 0, 0)
            return SessionId("imported")
        result = backend.open(scope).maintenance.import_foreign_history(origin, (ForeignTurn("user", "first"), ForeignTurn("user", "second")), title="Imported", cwd=None, profile="p")
        assert not result.already_imported
        return result.session_id

    def write():
        attempted.set()
        try:
            return backend.open(scope).messages.append("existing", NewMessage("user", "concurrent", timestamp=NOW))
        finally:
            completed.set()

    monkeypatch.setattr(maintenance, "_imported_message", paused)
    if fail_at == "publish":
        state = authority_state(backend, scope)
        state.failpoints.arm(state.PUBLISH)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            imported = executor.submit(import_history)
            try:
                assert entered.wait(10), "import did not reach payload construction"
                written = executor.submit(write)
                assert attempted.wait(10), "writer did not start"
                completed.wait(0.2)
            finally:
                release.set()
            if fail_at:
                error = InjectedFailure if fail_at == "publish" else RuntimeError
                message = "publish" if fail_at == "publish" else "import payload failed"
                with pytest.raises(error, match=f"^{message}$"):
                    imported.result(timeout=10)
                sid = None
            else:
                sid = imported.result(timeout=10)
            concurrent_id = written.result(timeout=10)
        store = backend.open(scope)
        rows = store.messages.read_transcript("existing")
        assert [(row.id, row.content) for row in rows] == [(original_id, "original"), (concurrent_id, "concurrent")]
        assert store.sessions.get("existing").message_count == 2
        if fail_at:
            assert store.sessions.get("imported") is None
            assert store.messages.read_transcript("imported") == ()
            assert store.maintenance.find_foreign_import(origin) is None
            imported_rows = ()
        else:
            imported_rows = store.messages.read_transcript(sid)
            assert [row.content for row in imported_rows] == ["first", "second"]
            exported = store.maintenance.export_session(sid)
            assert (exported.session.message_count, [row.content for row in exported.messages]) == (2, ["first", "second"])
        after_id = store.messages.append("existing", NewMessage("user", "after", timestamp=NOW))
        ids = [original_id, concurrent_id, after_id, *(row.id for row in imported_rows)]
        assert sorted(ids) == list(range(1, len(ids) + 1))
        assert after_id == max(ids)
        assert store.sessions.get("existing").message_count == 3
    finally:
        backend.close()
