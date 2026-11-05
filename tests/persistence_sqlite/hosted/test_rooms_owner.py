"""The ``HostedRoomRepository`` verbs on the bound profile's ``state.db`` with no placement: owner claims and reads, the owner claim inside ``create_room``, transport bindings, the two member-session verbs and the hosted member check, under a real ``runtime_epoch``."""

from __future__ import annotations

import pytest

from gateway import hosted_rooms as m1
from persistence import AuthorityRefusal, FenceStale, SessionId
from persistence.conversation.sessions import TitleConflict
from persistence.gateway.hosted_rooms import (
    HostedMemberGuard, HostedOwnerClaim, HostedRoomError, HostedRoomRefusal, HostedTaskIdentity,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from tests.persistence_sqlite.hosted.conftest import NOW


pytestmark = pytest.mark.usefixtures("clock")

MEMBERS = [{"member_id": "m1", "profile": "p1"},
           {"member_id": "m2", "profile": "p2", "target": {"kind": "peer", "installation_id": "i2"}}]


def _epoch(store) -> int:
    return begin_runtime_epoch(store.holder(), instance_id="hosted-test")


def _create(store, room_id: str, *, owner: HostedOwnerClaim | None = None, placement: str = "profile", members=None):
    return store.hosted_rooms.create_room(placement=placement, room_id=room_id, name="Room",
                                          members=MEMBERS if members is None else members,
                                          authority_gateway_id="gw", owner=owner, now=NOW)


@pytest.fixture
def seeded(backend, scopes):
    store = backend.open(scopes["a"])
    epoch = _epoch(store)
    _create(store, "owned", owner=HostedOwnerClaim(epoch, "alice", rule=None))
    _create(store, "unowned")
    _create(store, "gone")
    store.hosted_rooms.disband_room(placement="profile", room_id="gone", expected_gateway_id="gw", expected_epoch=1, now=NOW)
    return store, epoch


def test_stale_epochs_refuse_as_fence_stale_before_anything_is_written(seeded):
    store, epoch = seeded
    with pytest.raises(FenceStale):
        store.hosted_rooms.claim_owner(epoch=epoch + 1, room_id="owned", actor="alice", owner=None)
    with pytest.raises(FenceStale):
        _create(store, "fenced", owner=HostedOwnerClaim(epoch - 1, "alice", rule=None))
    with pytest.raises(FenceStale):
        store.hosted_rooms.bind_transport(epoch=epoch - 1, session_id="s", binding="{}")
    assert store.hosted_rooms.owner("fenced") is None
    assert all(room["room_id"] != "fenced" for room in store.hosted_rooms.list_rooms(placement="profile"))
    assert store.hosted_rooms.transport_binding("s") is None


def test_the_claim_rolls_back_with_a_refused_room(backend, scopes):
    store = backend.open(scopes["a"])
    epoch = _epoch(store)
    for index in range(m1.MAX_ACTIVE_ROOMS):
        _create(store, f"room-{index}", members=[])
    with pytest.raises(HostedRoomError, match="too many active Group Chats"):
        _create(store, "one-more", owner=HostedOwnerClaim(epoch, "alice", rule=None), members=[])
    assert store.hosted_rooms.owner("one-more") is None
    with pytest.raises(ValueError, match="placement='profile'"):
        _create(store, "shared-owned", owner=HostedOwnerClaim(epoch, "alice", rule=None), placement="shared")


def test_owner_reads(seeded):
    store, epoch = seeded
    assert store.hosted_rooms.owner("owned") == "alice"
    assert store.hosted_rooms.owner("unowned") is None
    _create(store, "second", owner=HostedOwnerClaim(epoch, "bob", rule=None))
    assert store.hosted_rooms.owned_rooms() == frozenset({"owned", "second"})


def test_bind_transport_inserts_once_and_refuses_another_text(seeded):
    store, epoch = seeded
    store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_s", binding='{"a": 1}')
    store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_s", binding='{"a": 1}')
    with pytest.raises(AuthorityRefusal) as refused:
        store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_s", binding='{"a":1}')
    assert refused.value.reason == "permission_denied"
    assert store.hosted_rooms.transport_binding("room_s") == '{"a": 1}'


def test_label_room_session_titles_hides_and_reports_the_title_holder(seeded):
    store, epoch = seeded
    holder = store.holder()
    for sid in ("s-room", "s-other"):
        holder._execute_write(lambda conn, sid=sid: conn.execute(
            "INSERT INTO sessions(id, source, started_at) VALUES(?, 'api_server', ?)", (sid, NOW)))
    store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-room", title="Group: r1")
    store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-none", title="Group: r9")
    with pytest.raises(TitleConflict) as taken:
        store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-other", title="Group: r1")
    assert taken.value.holder == SessionId("s-room")
    rows = dict(holder._read_all("SELECT id, title || ':' || hidden FROM sessions WHERE id LIKE 's-%'"))
    assert rows == {"s-room": "Group: r1:1", "s-other": None}


def test_ensure_room_member_session_creates_one_hidden_row_or_refuses(seeded):
    store, _ = seeded
    assert store.hosted_rooms.ensure_room_member_session(session_id="room_1", title=" Group:  alpha ") == "room_1"
    row = store.holder()._read_one("SELECT source, title, hidden, started_at FROM sessions WHERE id='room_1'")
    assert tuple(row) == ("bot_room", "Group: alpha", 1, NOW)
    with pytest.raises(HostedRoomRefusal, match="^room session identity conflicts with existing data$"):
        store.hosted_rooms.ensure_room_member_session(session_id="room_1", title=" Group:  alpha ")
    assert store.hosted_rooms.ensure_room_member_session(session_id="room_1", title="Group: alpha") == "room_1"
    with pytest.raises(HostedRoomRefusal, match="^Another group already uses this room title on the target gateway"):
        store.hosted_rooms.ensure_room_member_session(session_id="room_2", title="Group: alpha")


def _guard(**changes) -> HostedMemberGuard:
    values = {"room_id": "owned", "owner": "alice", "gateway_id": "gw", "authority_epoch": 1, "member_id": "m1",
              "profile": "p1", "home_current": True}
    return HostedMemberGuard(**{**values, **changes})


def test_member_authorized_checks_owner_room_authority_seat_home_and_task(seeded):
    store, epoch = seeded
    hosted = store.hosted_rooms
    with pytest.raises(FenceStale):
        hosted.member_authorized(_guard(), epoch=epoch - 1, operation="submit")
    assert hosted.member_authorized(_guard(), epoch=epoch, operation="submit") is True
    for changes in ({"owner": "bob"}, {"room_id": "unowned"}, {"gateway_id": "other"}, {"authority_epoch": 2},
                    {"member_id": "m2", "profile": "p2"}, {"home_current": False}):
        assert hosted.member_authorized(_guard(**changes), epoch=epoch, operation="submit") is False, changes
    hosted.append_event(placement="profile", room_id="owned", event_id="e1", kind="message.user",
                        actor={"kind": "user", "id": "u1"}, payload={"text": "go"}, authority_gateway_id="gw",
                        authority_epoch=1, now=NOW)
    driver = store.hosted_room_driver
    lease = driver.acquire_lease(placement="profile", room_id="owned", gateway_id="gw", authority_epoch=1,
                                 process_generation="p-1", ttl_seconds=60, now=NOW)
    task = HostedTaskIdentity("owned", "task-1", "thread-1", "turn-1")
    driver.admit_task(task, placement="profile", payload={"target_profile": "p1", "target_member_id": "m1",
                                                          "prompt": "go", "source_event_seq": 1}, now=NOW)
    assert hosted.member_authorized(_guard(task=task, generation=1), epoch=epoch, operation="submit") is False
    driver.start_task(task, lease, placement="profile", expected_cancel_generation=0, now=NOW)
    assert hosted.member_authorized(_guard(task=task, generation=1), epoch=epoch, operation="submit") is True
    assert hosted.member_authorized(_guard(task=task, generation=2), epoch=epoch, operation="submit") is False
    assert hosted.member_authorized(_guard(task=task, generation=1, member_id="m1", profile="p2"), epoch=epoch, operation="control") is False
