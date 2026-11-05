"""Import publication preserves successful writes by other repositories."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._reference.conversation import sessions
from tests.contracts._scopes import profile_scope

NOW = 1767225600.0
ORIGIN = ForeignOrigin("claude", "/history", "foreign")


@pytest.mark.parametrize("kind", ["portable", "foreign"])
@pytest.mark.parametrize("writer", ["create", "append"])
@pytest.mark.parametrize("fail_import", [False, True])
def test_import_preserves_concurrent_writer(monkeypatch, kind, writer, fail_import):
    backend = InMemoryBackend()
    scope = profile_scope("import-concurrency")
    backend.open(scope).sessions.create(NewSession(SessionId("existing"), "cli"))
    backend.open(scope).messages.append(SessionId("existing"), NewMessage("user", "original", timestamp=NOW))
    entered, release, attempted, completed = (threading.Event() for _ in range(4))
    original = sessions.blank_session_row

    def paused(sid, **kwargs):
        if sid != "concurrent":
            entered.set()
            assert release.wait(10), "import gate timed out"
            if fail_import:
                raise RuntimeError("import construction failed")
        return original(sid, **kwargs)

    def import_history():
        if kind == "portable":
            return backend.open(scope).maintenance.import_sessions([{"id": "imported", "source": "cli", "started_at": NOW, "messages": [{"role": "user", "content": "imported", "timestamp": NOW}]}])
        return backend.open(scope).maintenance.import_foreign_history(ORIGIN, (ForeignTurn("user", "imported"),), title="Imported", cwd=None, profile="p")

    def write():
        attempted.set()
        try:
            if writer == "create":
                return backend.open(scope).sessions.create(NewSession(SessionId("concurrent"), "cli"))
            return backend.open(scope).messages.append(SessionId("existing"), NewMessage("user", "concurrent", timestamp=NOW))
        finally:
            completed.set()

    monkeypatch.setattr(sessions, "blank_session_row", paused)
    with ThreadPoolExecutor(max_workers=2) as executor:
        imported = executor.submit(import_history)
        try:
            assert entered.wait(10), "import did not reach construction"
            written = executor.submit(write)
            assert attempted.wait(10), "writer did not start"
            completed.wait(0.2)
        finally:
            release.set()
        if fail_import:
            with pytest.raises(RuntimeError, match="^import construction failed$"):
                imported.result(timeout=10)
        else:
            result = imported.result(timeout=10)
            sid = SessionId("imported") if kind == "portable" else result.session_id
            assert [message.content for message in backend.open(scope).maintenance.export_session(sid).messages] == ["imported"]
        written.result(timeout=10)
    if writer == "create":
        assert backend.open(scope).sessions.get(SessionId("concurrent")) is not None
    exported = backend.open(scope).maintenance.export_session(SessionId("existing"))
    expected = ["original", "concurrent"] if writer == "append" else ["original"]
    assert [message.content for message in exported.messages] == expected
    assert exported.session.message_count == len(expected)
    if fail_import:
        assert backend.open(scope).sessions.get(SessionId("imported")) is None
        assert backend.open(scope).maintenance.find_foreign_import(ORIGIN) is None
    backend.close()
