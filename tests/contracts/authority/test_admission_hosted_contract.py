"""Contract cases for ``admit(hosted=)``: the hosted member check runs inside the admission write and refuses ``permission_denied``."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal
from persistence.authority.admissions import GateRefusal
from persistence.conversation.sessions import NewSession
from persistence.gateway.hosted_rooms import HostedMemberGuard, HostedOwnerClaim

SESSION = "room-member-session"
MEMBERS = [{"member_id": "m1", "profile": "p1"}]


def _guard(**changes) -> HostedMemberGuard:
    values = {"room_id": "room", "owner": "alice", "gateway_id": "gw", "authority_epoch": 1, "member_id": "m1", "profile": "p1", "home_current": True}
    return HostedMemberGuard(**{**values, **changes})


def _seed(store) -> int:
    store.sessions.create(NewSession(session_id=SESSION, source="hosted"))
    epoch = store.runtime.begin_epoch(instance_id="instance-hosted")
    store.hosted_rooms.create_room(placement="profile", room_id="room", name="Room", members=MEMBERS, authority_gateway_id="gw",
                                   owner=HostedOwnerClaim(epoch, "alice", rule=None), now=100.0)
    return epoch


def _admit(store, epoch, request_id, guard, **extras):
    return store.admissions.admit(epoch=epoch, principal_id="hosted:room:m1", session_id=SESSION, request_id=request_id, payload={"text": request_id}, hosted=guard, **extras)


def test_hosted_admission_runs_the_member_check(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    record, created = _admit(store, epoch, "hosted:room:1", _guard())
    assert created and record.status == "queued"
    for changes in ({"owner": "bob"}, {"member_id": "m3"}, {"authority_epoch": 2}, {"home_current": False}):
        with pytest.raises(AuthorityRefusal, match="^permission_denied$"):
            _admit(store, epoch, "hosted:room:2", _guard(**changes))
    assert [r.request_id for r in store.admissions.list(session_id=SESSION)] == ["hosted:room:1"]
    # CONTRACT: the exact live replay bypasses the check; the gate refuses before it.
    assert _admit(store, epoch, "hosted:room:1", _guard(home_current=False)) == (record, False)
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        _admit(store, epoch, "hosted:room:3", _guard(owner="bob"), gate=GateRefusal("runtime_draining"))


def test_unused_attempt_fences_delayed_submit_and_keeps_the_next_attempt_independent(backend, scope_a):
    from dataclasses import asdict
    import json

    from persistence.authority.vocabulary import OwnerRule
    from persistence.gateway.hosted_rooms import HostedTaskIdentity

    store = backend.open(scope_a)
    epoch = _seed(store)
    task = HostedTaskIdentity("room", "task", "thread", "turn")
    lookup = dict(epoch=epoch, session_id=SESSION, room_id="room", task_id="task",
                  principal="hosted:room:m1", owner=OwnerRule("alice", frozenset(), frozenset()))
    assert store.hosted_rooms.find_or_close_attempt(**lookup, generation=1) is None
    from persistence.gateway.hosted_rooms import HostedAttemptGuard
    def submit(generation):
        request_id = "hosted:" + json.dumps([asdict(task), generation])
        return _admit(store, epoch, request_id, HostedAttemptGuard(task, generation))
    with pytest.raises(AuthorityRefusal, match="stale_generation"):
        submit(1)
    admitted, created = submit(2)
    assert created
    assert store.hosted_rooms.find_or_close_attempt(**lookup, generation=2) == {
        "admission_id": admitted.admission_id, "status": "queued", "outcome": None}


@pytest.mark.parametrize("spelling", [r"\u0064task:23", " dtask:23 "])
def test_exact_lookup_finds_equivalent_stored_task_ids(backend, scope_a, spelling):
    from persistence.authority.vocabulary import OwnerRule

    store = backend.open(scope_a)
    epoch = _seed(store)
    # Both stored spellings decode to the same identifier in history.
    request_id = f'hosted:[{{"room_id":"room","task_id":"{spelling}","thread_id":"thread","turn_id":"turn" }},1]'
    row, _ = _admit(store, epoch, request_id, _guard())
    store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=1, outcome="completed",
                            result={"result": {"final_response": "saved"}, "usage": {}})
    found = store.hosted_rooms.find_or_close_attempt(
        epoch=epoch, session_id=SESSION, room_id="room", task_id="dtask:23", generation=1,
        principal="hosted:room:m1", owner=OwnerRule("alice", frozenset(), frozenset()))
    assert found == {"admission_id": row.admission_id, "status": "terminal", "outcome": "completed"}
