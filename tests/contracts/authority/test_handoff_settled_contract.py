"""Contract cases for the settled-handoff refusal: ``request_handoff(require_settled=True)`` and ``check_handoff_settled``.

Durable work owning the session's runtime lineage refuses with :class:`persistence.core.errors.HandoffUnsettled`: a queued, started or unknown admission, a live worker, a live turn lease or a live compression lock. ``unknown_execution`` is the reason for an unknown admission or worker, ``session_busy`` for every other owner. A refusal writes nothing.
"""

from __future__ import annotations

import pytest

from persistence import SessionId
from persistence.authority.coordination import HandoffState, HandoffStatus
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal, HandoffUnsettled

SID = SessionId("handoff-settled")


def _store(backend, scope, *session_ids: str):
    store = backend.open(scope)
    for sid in session_ids or (SID,):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    return store


def _begin(store, instance: str = "instance-a") -> int:
    return store.runtime.begin_epoch(instance_id=instance)


def _admit(store, epoch: int, session_id: str, request_id: str = "req-1") -> None:
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=session_id, request_id=request_id, payload={"text": request_id})


def _queued(store, session_id: str = SID) -> None:
    _admit(store, _begin(store), session_id)


def _started(store, session_id: str = SID) -> None:
    epoch = _begin(store)
    _admit(store, epoch, session_id)
    store.admissions.claim(epoch=epoch, session_id=session_id)


def _unknown(store, session_id: str = SID) -> None:
    _started(store, session_id)
    store.admissions.recover(epoch=_begin(store, "instance-b"))


def _worker(store, session_id: str = SID) -> None:
    store.worker_executions.register(epoch=_begin(store), execution_id="w-1", session_id=SessionId(session_id), generation=0, kind="child", adoption_secret="sec")


def _unknown_worker(store, session_id: str = SID) -> None:
    store.worker_executions.register(epoch=_begin(store), execution_id="w-1", session_id=SessionId(session_id), generation=0, kind="compute", adoption_secret="sec")
    store.admissions.recover(epoch=_begin(store, "instance-b"))


def _turn_lease(store, session_id: str = SID) -> None:
    assert store.coordination.try_acquire_turn_lease(session_id=SessionId(session_id), holder="turn-holder") is True


def _compression_lock(store, session_id: str = SID) -> None:
    assert store.coordination.try_acquire_compression_lock(session_id=SessionId(session_id), holder="compress-holder") is True


OWNERS = [
    pytest.param(_queued, "session_busy", id="queued-admission"),
    pytest.param(_started, "session_busy", id="started-admission"),
    pytest.param(_unknown, "unknown_execution", id="unknown-admission"),
    pytest.param(_worker, "session_busy", id="live-worker"),
    pytest.param(_unknown_worker, "unknown_execution", id="unknown-worker"),
    pytest.param(_turn_lease, "session_busy", id="turn-lease"),
    pytest.param(_compression_lock, "session_busy", id="compression-lock"),
]


def test_handoff_unsettled_is_an_authority_refusal_with_its_reason():
    refusal = HandoffUnsettled("unknown_execution")
    # CONTRACT: the refusal is an AuthorityRefusal whose reason is the wire code.
    assert isinstance(refusal, AuthorityRefusal)
    assert refusal.reason == "unknown_execution"


@pytest.mark.parametrize(("seed", "reason"), OWNERS)
def test_request_handoff_require_settled_refuses_per_owner_and_writes_nothing(backend, scope_a, seed, reason):
    store = _store(backend, scope_a)
    seed(store)
    coord = store.coordination
    with pytest.raises(HandoffUnsettled) as caught:
        coord.request_handoff(session_id=SID, platform="telegram", require_settled=True)
    # CONTRACT: the reason names what owns the lineage; nothing is armed.
    assert caught.value.reason == reason
    assert coord.get_handoff(session_id=SID) is None
    assert coord.has_pending_handoffs() is False


def test_request_handoff_refusal_keeps_a_previous_failed_handoff(backend, scope_a):
    store = _store(backend, scope_a)
    coord = store.coordination
    assert coord.request_handoff(session_id=SID, platform="telegram") is True
    assert coord.fail_handoff(session_id=SID, error="gave up") is True
    _queued(store)
    with pytest.raises(HandoffUnsettled):
        coord.request_handoff(session_id=SID, platform="discord", require_settled=True)
    # CONTRACT: a refused request leaves the stored handoff columns as they were.
    assert coord.get_handoff(session_id=SID) == HandoffState(state=HandoffStatus.FAILED, platform="telegram", error="gave up")


@pytest.mark.parametrize(("seed", "reason"), OWNERS)
def test_request_handoff_default_does_not_check_settlement(backend, scope_a, seed, reason):
    store = _store(backend, scope_a)
    seed(store)
    # CONTRACT: without require_settled the request arms as before, whatever owns the session.
    assert store.coordination.request_handoff(session_id=SID, platform="telegram") is True
    assert store.coordination.get_handoff(session_id=SID).state is HandoffStatus.PENDING


def test_request_handoff_require_settled_on_an_idle_session_arms(backend, scope_a):
    store = _store(backend, scope_a)
    coord = store.coordination
    # CONTRACT: a settled session arms; a second request sees the in-flight handoff and is False, not a refusal.
    assert coord.request_handoff(session_id=SID, platform="telegram", require_settled=True) is True
    assert coord.get_handoff(session_id=SID) == HandoffState(state=HandoffStatus.PENDING, platform="telegram", error=None)
    assert coord.request_handoff(session_id=SID, platform="discord", require_settled=True) is False
    assert coord.get_handoff(session_id=SID).platform == "telegram"


def test_request_handoff_require_settled_refuses_before_the_in_flight_check(backend, scope_a):
    store = _store(backend, scope_a)
    coord = store.coordination
    assert coord.request_handoff(session_id=SID, platform="telegram") is True
    _queued(store)
    # CONTRACT: the settlement check runs first in the write, so busy work refuses even while a handoff is in flight.
    with pytest.raises(HandoffUnsettled) as caught:
        coord.request_handoff(session_id=SID, platform="discord", require_settled=True)
    assert caught.value.reason == "session_busy"
    assert coord.get_handoff(session_id=SID).platform == "telegram"


def test_request_handoff_require_settled_on_a_missing_session_is_false(backend, scope_a):
    store = backend.open(scope_a)
    # CONTRACT: no such session is False, as without the check.
    assert store.coordination.request_handoff(session_id=SessionId("never"), platform="telegram", require_settled=True) is False


def test_request_handoff_refuses_for_work_on_a_compression_relative(backend, scope_a):
    store = _store(backend, scope_a, "ho-root")
    store.sessions.end(SessionId("ho-root"), "compression")
    store.sessions.create(NewSession(session_id=SessionId("ho-tip"), source="cli", parent_session_id=SessionId("ho-root")))
    _started(store, "ho-tip")
    # CONTRACT: the check covers the session's whole runtime lineage, so work on the continuation refuses a request on its root.
    with pytest.raises(HandoffUnsettled) as caught:
        store.coordination.request_handoff(session_id=SessionId("ho-root"), platform="telegram", require_settled=True)
    assert caught.value.reason == "session_busy"
    assert store.coordination.get_handoff(session_id=SessionId("ho-root")) is None


def test_request_handoff_ignores_terminal_admissions(backend, scope_a):
    store = _store(backend, scope_a)
    epoch = _begin(store)
    record, _ = store.admissions.admit(epoch=epoch, principal_id="p", session_id=SID, request_id="done", payload={"text": "done"})
    store.admissions.cancel(epoch=epoch, admission_id=record.admission_id)
    # CONTRACT: terminal work does not own the session.
    assert store.coordination.request_handoff(session_id=SID, platform="telegram", require_settled=True) is True


@pytest.mark.parametrize(("seed", "reason"), OWNERS)
def test_check_handoff_settled_refuses_per_owner(backend, scope_a, seed, reason):
    store = _store(backend, scope_a)
    coord = store.coordination
    assert coord.request_handoff(session_id=SID, platform="telegram", require_settled=True) is True
    assert coord.claim_handoff(session_id=SID) is True
    seed(store)
    # CONTRACT: the watcher's re-check refuses work admitted after the request, with the same reasons, and writes nothing.
    with pytest.raises(HandoffUnsettled) as caught:
        coord.check_handoff_settled(session_id=SID)
    assert caught.value.reason == reason
    assert coord.get_handoff(session_id=SID) == HandoffState(state=HandoffStatus.RUNNING, platform="telegram", error=None)


def test_check_handoff_settled_passes_on_a_settled_session(backend, scope_a):
    store = _store(backend, scope_a)
    coord = store.coordination
    assert coord.request_handoff(session_id=SID, platform="telegram") is True
    assert coord.claim_handoff(session_id=SID) is True
    # CONTRACT: a settled session passes the re-check with nothing written.
    assert coord.check_handoff_settled(session_id=SID) is None
    assert coord.check_handoff_settled(session_id=SessionId("never")) is None
    assert coord.get_handoff(session_id=SID).state is HandoffStatus.RUNNING
    _turn_lease(store)
    with pytest.raises(HandoffUnsettled):
        coord.check_handoff_settled(session_id=SID)
    store.coordination.release_turn_lease(session_id=SID, holder="turn-holder")
    assert coord.check_handoff_settled(session_id=SID) is None
