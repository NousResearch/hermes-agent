"""``admit(hosted=)`` on SQLite: the hosted member check (the gateway family's body) runs inside the admission
write where the legacy ``authorize_write`` hook ran, after the gate, refusing ``permission_denied`` with nothing
written; an exact replay bypasses it."""

from __future__ import annotations

import dataclasses

import pytest

from persistence.authority.admissions import GateRefusal
from persistence.core.errors import AuthorityRefusal
from persistence.gateway.hosted_rooms import HostedMemberGuard
from tests.persistence_sqlite.authority._support import counted_transactions, dump
from tests.persistence_sqlite.authority._admission_support import EPOCH, admit

# The owner key prefix of main's gateway/session_hosted_service.py.
_OWNER = "gateway.hosted.owner.v1:"
GUARD = HostedMemberGuard(room_id="room1", owner="owner-1", gateway_id="gw-1", authority_epoch=1, member_id="m1",
                          profile="prof", home_current=True)


def _room(db, epoch):
    db.create_session("hs", source="bot_room")
    db.store.hosted_rooms.create_room(placement="profile", room_id="room1", name="Room",
                                      members=[{"member_id": "m1", "profile": "prof"}], authority_gateway_id="gw-1")
    db._execute_write(lambda conn: conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)",
                                                (_OWNER + "room1", "owner-1")))
    return admit(db, "hs", "hosted:room1:m1:1", principal_id="hosted", payload={"text": "first"})


def _admit(store, request_id="hosted:room1:m1:2", **kwargs):
    return store.admissions.admit(epoch=EPOCH, principal_id="hosted", session_id="hs", request_id=request_id,
                                  payload={"text": "second"}, **kwargs)


def test_admit_runs_the_member_check_and_admits(seeded):
    _path, store, _ = seeded(_room)
    with counted_transactions(store.holder()) as counts:
        row, created = _admit(store, hosted=GUARD)
    assert created and row.status == "queued"
    assert counts == {"write": 1, "read": 1}


@pytest.mark.parametrize("change", [{"owner": "other"}, {"room_id": "room2"}, {"gateway_id": "gw-2"},
                                    {"authority_epoch": 2}, {"member_id": "m2"}, {"profile": "other"},
                                    {"home_current": False}])
def test_admit_refuses_a_member_check_that_fails(seeded, change):
    path, store, _ = seeded(_room)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        _admit(store, hosted=dataclasses.replace(GUARD, **change))
    assert refused.value.reason == "permission_denied"
    assert dump(path) == before


def test_admit_raises_the_gate_before_the_member_check(seeded):
    path, store, _ = seeded(_room)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        _admit(store, hosted=dataclasses.replace(GUARD, home_current=False), gate=GateRefusal("runtime_draining"))
    assert refused.value.reason == "runtime_draining"
    assert dump(path) == before


def test_exact_replay_bypasses_the_member_check(seeded):
    _path, store, first = seeded(_room)
    row, created = store.admissions.admit(epoch=EPOCH, principal_id="hosted", session_id="hs",
                                          request_id="hosted:room1:m1:1", payload={"text": "first"},
                                          hosted=dataclasses.replace(GUARD, home_current=False))
    assert not created and row.admission_id == first["admission_id"]
