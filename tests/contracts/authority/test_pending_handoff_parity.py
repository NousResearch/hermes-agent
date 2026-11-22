"""Pending-handoff existence is a scope-local, read-only lifecycle query."""


from persistence import SessionId
from persistence.authority.coordination import HandoffStatus
from persistence.conversation.sessions import NewSession


def test_pending_handoff_existence_tracks_lifecycle_in_its_scope(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    coord = backend.open(scope_a).coordination
    session_id = SessionId("handoff")
    assert coord.has_pending_handoffs() is False
    store_a.sessions.create(NewSession(session_id=session_id, source="cli"))
    assert coord.has_pending_handoffs() is False
    assert coord.request_handoff(session_id=session_id, platform="local")
    assert coord.has_pending_handoffs() is True
    assert store_b.coordination.has_pending_handoffs() is False
    assert coord.has_pending_handoffs() is True
    assert (coord.get_handoff(session_id=session_id)).state is HandoffStatus.PENDING
    assert coord.claim_handoff(session_id=session_id)
    assert coord.has_pending_handoffs() is False
    coord.complete_handoff(session_id=session_id)
    assert coord.has_pending_handoffs() is False
    assert coord.request_handoff(session_id=session_id, platform="local")
    assert coord.fail_handoff(session_id=session_id, error="failed")
    assert coord.has_pending_handoffs() is False
    assert (coord.get_handoff(session_id=session_id)).state is HandoffStatus.FAILED
