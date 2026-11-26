"""A handoff cannot outlive or create its session."""


from persistence.conversation.sessions import NewSession
from tests.contracts._reference import InMemoryBackend
from tests.contracts._scopes import profile_scope


def test_missing_session_handoff_is_refused_without_publication():
    backend = InMemoryBackend()
    scope = profile_scope("handoff")
    try:
        assert backend.open(scope).coordination.request_handoff(session_id="missing", platform="telegram") is False
        assert backend.open(scope).coordination.get_handoff(session_id="missing") is None
        assert backend.open(scope).coordination.list_pending_handoffs() == ()
        assert backend.open(scope).sessions.get("missing") is None
        backend.open(scope).sessions.create(NewSession(session_id="missing", source="cli"))
        assert backend.open(scope).coordination.request_handoff(session_id="missing", platform="telegram") is True
        assert [item.session_id for item in backend.open(scope).coordination.list_pending_handoffs()] == ["missing"]
    finally:
        backend.close()
