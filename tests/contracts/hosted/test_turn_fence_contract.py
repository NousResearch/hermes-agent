"""The claimed-turn fence on every write: each write a store makes while a claimed turn of that store is bound checks the turn's fence (authority epoch + execution generation) inside the write's own transaction and lands nothing once the lease moved. Requires authority_fences.

Every case claims a real admission (``AdmissionRepository.claim``) and binds the turn the way the gateway does (``agent.runtime_scope.bind_turn`` under a bound scope, or a ``hermes_scope.TurnScope`` pushed directly), with the claim's own fence (``claim_fence``). Supersession (the session's next claim, a new authority epoch) runs in a fresh context, as another writer's would.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from agent.runtime_scope import bind_turn, claim_fence, outlive_turn
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, Scope, TurnScope, push_state
from persistence import DelegationId, SessionId, TurnFenceSuperseded, TurnLeaseLost
from persistence.authority.coordination import OwnerStamp
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.contracts.conftest import require_capability

SESSION = SessionId("turn-fence-s1")
NOW = 1_800_000_000.0


def _msg(content: str) -> NewMessage:
    return NewMessage(role="user", content=content)


def _elsewhere(fn, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` as another writer would: in a fresh context, outside any turn."""
    return contextvars.Context().run(lambda: fn(*args, **kwargs))


def _claimed(backend, scope, request_id: str = "r1"):
    """A store whose ``SESSION`` holds one started claim; returns the store, the epoch and the claim."""
    require_capability(backend, "authority_fences")
    require_capability(backend, "durable_admissions")
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=SESSION, source="contract"))
    epoch = store.runtime.begin_epoch(instance_id="owner-1")
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id=request_id, payload={"text": request_id})
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed and claimed.status == "started"
    return store, epoch, claimed


def _next_claim(store, epoch: int, claimed):
    """The owner settles ``claimed`` and claims the session's next input: the session's generation moves."""
    def run():
        store.admissions.settle(epoch=epoch, admission_id=claimed.admission_id, generation=claimed.generation, outcome="completed")
        store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r-next", payload={"text": "next"})
        following = store.admissions.claim(epoch=epoch, session_id=SESSION)
        assert following and following.generation > claimed.generation
        return following
    return _elsewhere(run)


def _take_over(store) -> int:
    """Another owner takes the store over: a new authority epoch."""
    return _elsewhere(store.runtime.begin_epoch, instance_id="owner-2")


@contextmanager
def _turn(scope, claimed) -> Iterator[None]:
    """The claimed turn bound as the gateway binds it: the profile's scope, then ``bind_turn`` naming the issuing store."""
    with push_state(scope=Scope(LOCAL_TENANT, str(scope.profile_dir), LOCAL_OWNER)), \
            bind_turn(SESSION, claimed.admission_id, claim_fence(claimed), store=scope):
        yield


def _contents(store, session_id: str = SESSION) -> list:
    return [row.content for row in store.messages.read_transcript(SessionId(session_id))]


def test_append_under_current_fence_lands(backend_name: str, backend, scope_a) -> None:
    """An append under the bound turn's current fence lands exactly as without one, also through a TurnScope bound alone."""
    store, _epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        store.messages.append(SESSION, _msg("one"))
    turn = TurnScope(LOCAL_TENANT, str(scope_a.profile_dir), LOCAL_OWNER, SESSION, claimed.admission_id, claim_fence(claimed))
    with push_state(scope=turn):
        store.messages.append(SESSION, _msg("two"))
    assert _contents(store) == ["one", "two"]
    assert store.sessions.get(SESSION).message_count == 2


def test_append_after_newer_claim_is_refused_and_lands_nothing(backend_name: str, backend, scope_a) -> None:
    """After the session's next claim, the superseded turn's append raises TurnFenceSuperseded and the transcript is unchanged."""
    store, epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        store.messages.append(SESSION, _msg("before"))
        _next_claim(store, epoch, claimed)
        with pytest.raises(TurnFenceSuperseded) as refused:
            store.messages.append(SESSION, _msg("stale reply"))
        # A session write of the same turn is refused too, and every lost-lease handler catches it.
        with pytest.raises(TurnLeaseLost):
            store.sessions.update_cwd(SESSION, "/elsewhere")
    assert refused.value.session_id == SESSION and refused.value.cause == "turn_lease"
    assert _contents(store) == ["before"]
    assert store.sessions.get(SESSION).message_count == 1
    assert store.sessions.get(SESSION).cwd != "/elsewhere"
    # The successor's own writes are unaffected.
    store.messages.append(SESSION, _msg("successor"))
    assert _contents(store) == ["before", "successor"]


def test_append_after_epoch_takeover_is_refused(backend_name: str, backend, scope_a) -> None:
    """After a new authority epoch, the old turn's append is refused, also from a TurnScope bound alone."""
    store, _epoch, claimed = _claimed(backend, scope_a)
    _take_over(store)
    with _turn(scope_a, claimed), pytest.raises(TurnFenceSuperseded):
        store.messages.append(SESSION, _msg("old owner"))
    turn = TurnScope(LOCAL_TENANT, str(scope_a.profile_dir), LOCAL_OWNER, SESSION, claimed.admission_id, claim_fence(claimed))
    with push_state(scope=turn), pytest.raises(TurnFenceSuperseded):
        store.messages.append(SESSION, _msg("old owner again"))
    assert _contents(store) == []


def test_compression_continuation_write_is_fenced(backend_name: str, backend, scope_a) -> None:
    """A compression continuation write made by a superseded turn is refused and publishes nothing."""
    store, epoch, claimed = _claimed(backend, scope_a)
    for i in range(3):
        store.messages.append(SESSION, _msg(f"turn {i}"))
    with _turn(scope_a, claimed):
        assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="compressor", ttl_seconds=60.0)
        _next_claim(store, epoch, claimed)
        with pytest.raises(TurnFenceSuperseded):
            store.lineage.publish_compression_child(
                SESSION, CompressionChild(session_id=SessionId("turn-fence-c1"), source="contract"), [_msg("summary")],
                compression_lock_holder="compressor", require_compression_lease=True, require_lease_refresh=True)
        # The superseded turn still releases its own lock: the release drops only the holder's row.
        store.coordination.release_compression_lock(session_id=SESSION, holder="compressor")
    assert store.sessions.get(SessionId("turn-fence-c1")) is None
    assert store.sessions.get(SESSION).ended_at is None
    assert store.lineage.tip(SESSION) == SESSION
    assert store.coordination.get_compression_lock_holder(session_id=SESSION) is None
    # The same publication by the session's current owner, outside the old turn, goes through.
    assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="compressor-2", ttl_seconds=60.0)
    store.lineage.publish_compression_child(
        SESSION, CompressionChild(session_id=SessionId("turn-fence-c2"), source="contract"), [_msg("summary")],
        compression_lock_holder="compressor-2", require_compression_lease=True, require_lease_refresh=True)
    assert store.lineage.tip(SESSION) == SessionId("turn-fence-c2")


def _dispatch(store, delegation_id: str) -> DelegationId:
    did = DelegationId(delegation_id)
    store.coordination.record_delegation_dispatch(
        delegation_id=did, origin_session="route-1", origin_ui_session_id="ui-1", parent_session_id=SESSION,
        dispatched_at=NOW, owner=OwnerStamp(pid=4242, started_at=777), task={"goal": "g"})
    return did


def _complete(store, did: DelegationId) -> None:
    store.coordination.record_delegation_completion(delegation_id=did, state="completed", completed_at=NOW + 1,
                                                    event={"delegation_id": did, "status": "completed"}, result={"status": "completed"})


def test_delegation_completion_keeps_owner_epoch_half(backend_name: str, backend, scope_a) -> None:
    """An async delegation completion (owner_only) lands after the session's next claim but is refused after an epoch takeover."""
    store, epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        first, second, plain = _dispatch(store, "d-next-claim"), _dispatch(store, "d-takeover"), _dispatch(store, "d-plain")
        copied = contextvars.copy_context()

    def report(did: DelegationId, *, outliving: bool = True) -> None:
        if outliving:
            with outlive_turn():
                _complete(store, did)
        else:
            _complete(store, did)
    _next_claim(store, epoch, claimed)
    # Without outlive_turn the turn's late write is a stale reply: the next claim refuses it.
    with pytest.raises(TurnFenceSuperseded):
        copied.run(report, plain, outliving=False)
    copied.run(report, first)
    _take_over(store)
    with pytest.raises(TurnFenceSuperseded):
        copied.run(report, second)
    assert store.coordination.get_delegation(delegation_id=first).state == "completed"
    assert store.coordination.get_delegation(delegation_id=second).state == "running"
    assert store.coordination.get_delegation(delegation_id=plain).state == "running"


def test_write_to_another_store_is_not_fenced(backend_name: str, backend, scope_a, scope_b) -> None:
    """A turn fenced on store A writes freely to store B (the fence says nothing about B)."""
    store_a, _epoch, claimed = _claimed(backend, scope_a)
    store_b = backend.open(scope_b)
    store_b.sessions.create(NewSession(session_id=SESSION, source="contract"))
    _take_over(store_a)
    with _turn(scope_a, claimed):
        store_b.messages.append(SESSION, _msg("another store"))
        with pytest.raises(TurnFenceSuperseded):
            store_a.messages.append(SESSION, _msg("issuing store"))
    assert _contents(store_b) == ["another store"]
    assert _contents(store_a) == []


def test_lease_holder_append_keeps_its_semantics(backend_name: str, backend, scope_a) -> None:
    """turn_lease_holder appends keep main's semantics under a bound fence: a lost holder raises TurnLeaseLost, a held one renews."""
    store, epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="turn-1", ttl_seconds=300.0)
        before = store.coordination.turn_lease_expires_at(session_id=SESSION, holder="turn-1")
        store.messages.append(SESSION, _msg("held"), turn_lease_holder="turn-1", turn_lease_ttl_seconds=600.0)
        assert store.coordination.turn_lease_expires_at(session_id=SESSION, holder="turn-1") >= before
        with pytest.raises(TurnLeaseLost) as lost:
            store.messages.append(SESSION, _msg("not the holder"), turn_lease_holder="turn-2")
        # A lease refusal stays a lease refusal while the fence holds.
        assert not isinstance(lost.value, TurnFenceSuperseded)
        _next_claim(store, epoch, claimed)
        with pytest.raises(TurnFenceSuperseded):
            store.messages.append(SESSION, _msg("held but superseded"), turn_lease_holder="turn-1")
        # The superseded turn still releases its own lease, so the session's next turn need not wait out the TTL.
        store.coordination.release_turn_lease(session_id=SESSION, holder="turn-1")
    assert _contents(store) == ["held"]
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="next-turn", ttl_seconds=300.0)
