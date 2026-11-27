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
