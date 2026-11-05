"""Contract cases for the profile-bound verbs of ``persistence.gateway.hosted_rooms.HostedRoomRepository``: owner claims, transport bindings, the two member-session verbs, the hosted member check and the member homes Disband forgets.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs; epochs are begun through the backend's ``begin_epoch`` seam. The cases cover the owner claim, the member check, the binding, the label, the member session, room creation, member homes and the held-grant refusal of Disband.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, FenceStale, ProfileScope, SessionId
from persistence.conversation.sessions import NewSession, TitleConflict
from persistence.gateway.hosted_rooms import (
    HostedMemberGuard,
    HostedOwnerClaim,
    HostedRoomError,
    HostedRoomRefusal,
    HostedTaskIdentity,
    RoomHeldGrantsPendingError,
)
from tests.contracts._reference._authority import AuthorityState, InjectedFailure, authority_state
from tests.contracts.authority.test_owner_rule_contract import OWNER, RULE, STRANGER
from tests.contracts.conftest import require_failpoints

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2", "target": {"kind": "peer", "installation_id": "i2", "profile": "p2"}}]


def _epoch(backend, scope: ProfileScope) -> int:
    return backend.begin_epoch(scope, "instance-contract")


def _room(store, room_id: str = "room", *, owner: HostedOwnerClaim | None = None, now: float = 100.0, members=None):
    return store.hosted_rooms.create_room(
        placement="profile", room_id=room_id, name="Room", members=MEMBERS if members is None else members,
        authority_gateway_id="gw", owner=owner, now=now)


def _guard(**changes) -> HostedMemberGuard:
    values = {"room_id": "room", "owner": "alice", "gateway_id": "gw", "authority_epoch": 1, "member_id": "m1", "profile": "p1", "home_current": True}
    return HostedMemberGuard(**{**values, **changes})


# ── owner claims ──────────────────────────────────────────────────────────────


def test_create_room_with_an_owner_claims_it_and_claim_owner_authorizes_only_that_owner(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    room = _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None))
    # CONTRACT: the fresh room dict has no latest_seq and carries the normalized members.
    assert room == {
        "room_id": "room", "name": "Room", "members": MEMBERS, "authority_gateway_id": "gw", "authority_epoch": 1,
        "revision": 1, "created_at": 100.0, "updated_at": 100.0, "idempotent": False}
    assert store.hosted_rooms.owner("room") == "alice"
    assert store.hosted_rooms.owned_rooms() == frozenset({"room"})
    store.hosted_rooms.claim_owner(epoch=epoch, room_id="room", actor="alice", owner=None)
    with pytest.raises(AuthorityRefusal) as other:
        store.hosted_rooms.claim_owner(epoch=epoch, room_id="room", actor="bob", owner=None)
    assert other.value.reason == "permission_denied"
    # CONTRACT: without create, an unowned room is refused and stays unowned.
    with pytest.raises(AuthorityRefusal) as unowned:
        store.hosted_rooms.claim_owner(epoch=epoch, room_id="elsewhere", actor="alice", owner=None)
    assert unowned.value.reason == "permission_denied"
    assert store.hosted_rooms.owner("elsewhere") is None


def test_claims_read_the_stored_owner_through_the_owner_rule(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(epoch, "basic:admin", RULE))
    # CONTRACT: a stored owner belongs to the actor under the owner rule (a sign-in stamp is the host owner's), for claim_owner and for create_room's claim; the stored stamp is kept; without a rule only the equal stamp does.
    store.hosted_rooms.claim_owner(epoch=epoch, room_id="room", actor=OWNER, owner=RULE)
    assert _room(store, owner=HostedOwnerClaim(epoch, OWNER, RULE))["idempotent"] is True
    assert store.hosted_rooms.owner("room") == "basic:admin"
    for actor, rule in ((STRANGER, RULE), (OWNER, None)):
        with pytest.raises(AuthorityRefusal) as claim:
            store.hosted_rooms.claim_owner(epoch=epoch, room_id="room", actor=actor, owner=rule)
        assert claim.value.reason == "permission_denied"
        with pytest.raises(AuthorityRefusal) as create:
            _room(store, owner=HostedOwnerClaim(epoch, actor, rule))
        assert create.value.reason == "permission_denied"


def test_the_owner_replays_an_identical_create_and_anyone_else_is_refused(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None))
    replay = _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None), now=200.0)
    assert replay["idempotent"] is True and replay["latest_seq"] == 0 and replay["updated_at"] == 100.0
    with pytest.raises(AuthorityRefusal) as refused:
        _room(store, owner=HostedOwnerClaim(epoch, "bob", rule=None))
    assert refused.value.reason == "permission_denied"
    assert store.hosted_rooms.owner("room") == "alice"


def test_an_owner_claim_never_adopts_a_room_id_with_history(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store, "unowned")
    _room(store, "gone")
    store.hosted_rooms.disband_room(placement="profile", room_id="gone", expected_gateway_id="gw", expected_epoch=1, now=150.0)
    # CONTRACT: the claim runs first in the create transaction, so a live or retired room id refuses as permission_denied before the room's own checks.
    for room_id in ("unowned", "gone"):
        with pytest.raises(AuthorityRefusal) as refused:
            _room(store, room_id, owner=HostedOwnerClaim(epoch, "alice", rule=None))
        assert refused.value.reason == "permission_denied"
    assert store.hosted_rooms.owned_rooms() == frozenset()


def test_the_owner_claim_and_the_room_commit_or_roll_back_together(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    for index in range(256):
        _room(store, f"room-{index}", members=[])
    # CONTRACT: the active-room cap refuses after the claim passed; the claim rolls back with the room.
    with pytest.raises(HostedRoomError, match="This host has too many active Group Chats. Delete one and try again."):
        _room(store, "one-more", owner=HostedOwnerClaim(epoch, "alice", rule=None))
    assert store.hosted_rooms.owner("one-more") is None
    stale = epoch
    _epoch(backend, scope_a)
    with pytest.raises(FenceStale):
        _room(store, "fenced", owner=HostedOwnerClaim(stale, "alice", rule=None))
    assert store.hosted_rooms.owner("fenced") is None
    assert all(room["room_id"] != "fenced" for room in store.hosted_rooms.list_rooms(placement="profile", limit=500))


def test_probe_room_owner_answers_from_the_bound_profile_without_creating_its_store(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    # CONTRACT: a profile whose store was never opened answers False.
    assert store_a.hosted_rooms.probe_room_owner("room") is False
    _room(store_a, owner=HostedOwnerClaim(_epoch(backend, scope_a), "alice", rule=None))
    _room(store_b)
    assert store_a.hosted_rooms.probe_room_owner("room") is True
    assert store_a.hosted_rooms.probe_room_owner("other") is False
    # CONTRACT: a room without an owner entry is not served by the profile's gateway.
    assert store_b.hosted_rooms.probe_room_owner("room") is False


def test_owner_claims_stay_in_their_profile(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    _room(store_a, owner=HostedOwnerClaim(_epoch(backend, scope_a), "alice", rule=None))
    assert store_b.hosted_rooms.owner("room") is None
    assert store_b.hosted_rooms.owned_rooms() == frozenset()
    with pytest.raises(AuthorityRefusal):
        store_b.hosted_rooms.claim_owner(epoch=_epoch(backend, scope_b), room_id="room", actor="alice", owner=None)


def test_claim_owner_checks_the_epoch_before_the_owner(backend, scope_a):
    store = backend.open(scope_a)
    stale = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(stale, "alice", rule=None))
    _epoch(backend, scope_a)
    with pytest.raises(FenceStale):
        store.hosted_rooms.claim_owner(epoch=stale, room_id="room", actor="alice", owner=None)
    with pytest.raises(FenceStale):
        store.hosted_rooms.claim_owner(epoch=0, room_id="room", actor="alice", owner=None)


# ── transport bindings ────────────────────────────────────────────────────────


def test_bind_transport_is_insert_if_absent_and_refuses_a_different_binding(backend, scope_a, scope_b):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    first = '{"owner": "alice", "selector": {"room_id": "room"}}'
    store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_session", binding=first)
    # CONTRACT: the identical binding is a no-op; the comparison is byte-exact.
    store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_session", binding=first)
    for other in ('{"owner": "bob", "selector": {"room_id": "room"}}', first.replace(" ", "")):
        with pytest.raises(AuthorityRefusal) as refused:
            store.hosted_rooms.bind_transport(epoch=epoch, session_id="room_session", binding=other)
        assert refused.value.reason == "permission_denied"
    assert store.hosted_rooms.transport_binding("room_session") == first
    assert store.hosted_rooms.transport_binding("other") is None
    assert backend.open(scope_b).hosted_rooms.transport_binding("room_session") is None


def test_bind_transport_under_a_stale_epoch_writes_nothing(backend, scope_a):
    store = backend.open(scope_a)
    stale = _epoch(backend, scope_a)
    _epoch(backend, scope_a)
    with pytest.raises(FenceStale):
        store.hosted_rooms.bind_transport(epoch=stale, session_id="s-1", binding="{}")
    assert store.hosted_rooms.transport_binding("s-1") is None


def test_a_binding_that_dies_at_publication_leaves_nothing(backend, scope_a):
    # BROKEN WRITER (reference failpoints): a transition that dies at publication commits nothing.
    require_failpoints(backend)
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    authority_state(backend, scope_a).failpoints.arm(AuthorityState.PUBLISH)
    with pytest.raises(InjectedFailure):
        store.hosted_rooms.bind_transport(epoch=epoch, session_id="s", binding="{}")
    assert store.hosted_rooms.transport_binding("s") is None
    store.hosted_rooms.bind_transport(epoch=epoch, session_id="s", binding="{}")
    assert store.hosted_rooms.transport_binding("s") == "{}"


# ── member sessions ───────────────────────────────────────────────────────────


def test_label_room_session_titles_and_hides_one_row_under_the_epoch(backend, scope_a):
    store = backend.open(scope_a)
    for sid in ("s-room", "s-other"):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="api_server"))
    epoch = _epoch(backend, scope_a)
    store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-room", title="Group: r1")
    row = store.sessions.get(SessionId("s-room"))
    assert row is not None and row.title == "Group: r1" and row.hidden is True
    # CONTRACT: a missing row writes nothing and raises nothing.
    store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-none", title="Group: r9")
    assert store.sessions.get(SessionId("s-none")) is None
    # CONTRACT: a title another session holds is refused; nothing changes.
    with pytest.raises(TitleConflict) as taken:
        store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-other", title="Group: r1")
    assert taken.value.holder == "s-room"
    other = store.sessions.get(SessionId("s-other"))
    assert other is not None and other.title is None and other.hidden is False
    _epoch(backend, scope_a)
    with pytest.raises(FenceStale):
        store.hosted_rooms.label_room_session(epoch=epoch, session_id="s-other", title="Group: r2")


def test_ensure_room_member_session_creates_one_hidden_room_row(backend, scope_a):
    store = backend.open(scope_a)
    assert store.hosted_rooms.ensure_room_member_session(session_id="room-s1", title="  Group:   alpha ") == "room-s1"
    row = store.sessions.get(SessionId("room-s1"))
    assert row is not None and row.source == "bot_room" and row.hidden is True and row.title == "Group: alpha"
    # CONTRACT: an existing row must carry exactly this title and source; the raw title is compared.
    with pytest.raises(HostedRoomRefusal, match="room session identity conflicts with existing data"):
        store.hosted_rooms.ensure_room_member_session(session_id="room-s1", title="  Group:   alpha ")
    assert store.hosted_rooms.ensure_room_member_session(session_id="room-s1", title="Group: alpha") == "room-s1"
    store.sessions.create(NewSession(session_id=SessionId("plain"), source="cli"))
    with pytest.raises(HostedRoomRefusal, match="room session identity conflicts with existing data"):
        store.hosted_rooms.ensure_room_member_session(session_id="plain", title="anything")
    with pytest.raises(HostedRoomRefusal, match="Another group already uses this room title on the target gateway. Rename or migrate that group before retrying."):
        store.hosted_rooms.ensure_room_member_session(session_id="room-s2", title="Group: alpha")
    assert store.sessions.get(SessionId("room-s2")) is None


# ── hosted member check ───────────────────────────────────────────────────────


def test_member_authorized_requires_owner_room_authority_seat_and_home(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None))
    assert store.hosted_rooms.member_authorized(_guard(), epoch=epoch, operation="submit") is True
    # CONTRACT: each check refuses with False, in this order: owner, live room, authority, a seated local member, the home.
    for changes in (
        {"owner": "bob"}, {"room_id": "missing"}, {"gateway_id": "other"}, {"authority_epoch": 2},
        {"member_id": "m3"}, {"profile": "p2"}, {"member_id": "m2", "profile": "p2"}, {"home_current": False},
    ):
        assert store.hosted_rooms.member_authorized(_guard(**changes), epoch=epoch, operation="submit") is False, changes


def test_member_authorized_fences_the_epoch_first(backend, scope_a):
    store = backend.open(scope_a)
    stale = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(stale, "alice", rule=None))
    current = _epoch(backend, scope_a)
    # CONTRACT: the check runs behind the epoch fence inside its read.
    with pytest.raises(FenceStale):
        store.hosted_rooms.member_authorized(_guard(), epoch=stale, operation="submit")
    assert store.hosted_rooms.member_authorized(_guard(), epoch=current, operation="submit") is True


def test_member_authorized_needs_the_owner_key_and_a_live_room(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store)
    # CONTRACT: a room nobody claimed has no owner key, so no member of it is authorized.
    assert store.hosted_rooms.member_authorized(_guard(), epoch=epoch, operation="submit") is False
    _room(store, "owned", owner=HostedOwnerClaim(epoch, "alice", rule=None))
    store.hosted_rooms.disband_room(placement="profile", room_id="owned", expected_gateway_id="gw", expected_epoch=1, now=150.0)
    assert store.hosted_rooms.member_authorized(_guard(room_id="owned"), epoch=epoch, operation="submit") is False


def test_member_authorized_with_a_task_needs_its_task_row(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None))
    # A guard names a task only after the driver admitted it, so the driver tables exist.
    assert store.hosted_room_driver.list_tasks(placement="profile", room_id="room") == ()
    task = HostedTaskIdentity("room", "task-1", "thread-1", "turn-1")
    # CONTRACT: with a task, the member check also needs the room's task row; none is stored here.
    for operation in ("submit", "execute", "control"):
        assert store.hosted_rooms.member_authorized(_guard(task=task, generation=1), epoch=epoch, operation=operation) is False


def test_member_authorized_reads_the_profile_placement_only(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)
    # CONTRACT: the canonical member check reads the bound profile's state.db; a shared-placement room does not seat anyone there.
    with pytest.raises(AuthorityRefusal):
        store.hosted_rooms.claim_owner(epoch=epoch, room_id="room", actor="alice", owner=None)
    assert store.hosted_rooms.member_authorized(_guard(), epoch=epoch, operation="submit") is False


# ── router-call claims ────────────────────────────────────────────────────────


def test_a_router_call_claim_has_one_holder_until_it_expires_or_is_released(backend, scope_a, scope_b):
    rooms, other = backend.open(scope_a).hosted_rooms, backend.open(scope_b).hosted_rooms

    def claim(store, owner: str, now: float, placement: str = "shared", discussion: str = "user:m1") -> bool:
        return store.claim_router_call(placement=placement, room_id="room", discussion_event_id=discussion, owner=owner, expires_at=now + 120.0, now=now)

    assert claim(rooms, "a", 1000.0) is True
    # CONTRACT: the shared file's claim is seen from another profile of the install; the holder itself does not re-claim.
    assert claim(other, "b", 1050.0) is False
    assert claim(rooms, "a", 1050.0) is False
    assert claim(rooms, "b", 1000.0, discussion="user:m2") is True
    assert claim(rooms, "b", 1000.0, placement="profile") is True
    rooms.release_router_call(placement="shared", room_id="room", discussion_event_id="user:m1", owner="b")  # not the holder
    assert claim(other, "b", 1060.0) is False
    assert claim(other, "b", 1120.0) is True  # expired claims are taken over
    other.release_router_call(placement="shared", room_id="room", discussion_event_id="user:m1", owner="b")
    assert claim(rooms, "c", 1121.0) is True


# ── member homes and Disband ──────────────────────────────────────────────────


def test_member_homes_append_in_first_recorded_order_once_each(backend, scope_a, scope_b):
    rooms = backend.open(scope_a).hosted_rooms
    assert rooms.member_homes(room_id="room", member_id="m1") == ()
    for home in ("/homes/a", "/homes/b", "/homes/a"):
        rooms.record_member_home(room_id="room", member_id="m1", home=home)
    assert rooms.member_homes(room_id="room", member_id="m1") == ("/homes/a", "/homes/b")
    assert rooms.member_homes(room_id="room", member_id="m2") == ()
    assert backend.open(scope_b).hosted_rooms.member_homes(room_id="room", member_id="m1") == ()


def test_disband_forgets_only_that_rooms_member_homes(backend, scope_a):
    store = backend.open(scope_a)
    _room(store, "room")
    for room_id, member_id in (("room", "m1"), ("room", "m2"), ("room-2", "m1"), ("roo", "m1")):
        store.hosted_rooms.record_member_home(room_id=room_id, member_id=member_id, home=f"/homes/{room_id}/{member_id}")
        store.hosted_rooms.record_attempt_home(room_id=room_id, member_id=member_id, task_id="t", generation=1, home=f"/homes/{room_id}/{member_id}/t")
    store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=150.0)
    assert store.hosted_rooms.member_homes(room_id="room", member_id="m1") == ()
    assert store.hosted_rooms.member_homes(room_id="room", member_id="m2") == ()
    assert store.hosted_rooms.member_homes(room_id="room-2", member_id="m1") == ("/homes/room-2/m1",)
    assert store.hosted_rooms.member_homes(room_id="roo", member_id="m1") == ("/homes/roo/m1",)
    # CONTRACT: the room's attempt homes go with its member homes; other rooms keep theirs.
    assert store.hosted_rooms.attempt_home(room_id="room", member_id="m1", task_id="t", generation=1) is None
    assert store.hosted_rooms.attempt_home(room_id="room-2", member_id="m1", task_id="t", generation=1) == "/homes/room-2/m1/t"
    assert store.hosted_rooms.attempt_home(room_id="roo", member_id="m1", task_id="t", generation=1) == "/homes/roo/m1/t"


def _driver_task(store, n: int, member_id: str = "m1"):
    """Task ``task-<n>`` of ``room`` for *member_id*, admitted; returns its identity."""
    identity = HostedTaskIdentity("room", f"task-{n}", f"thread-{n}", f"turn-{n}")
    store.hosted_room_driver.admit_task(identity, placement="profile", payload={
        "target_profile": "p1", "target_member_id": member_id, "prompt": "hi", "source_event_seq": n}, now=100.0)
    return identity


def test_attempt_homes_keep_the_first_home_and_name_the_stopping_attempts(backend, scope_a):
    # CONTRACT: ``record_attempt_home``: the first home recorded for an attempt wins; ``attempt_home`` reads it whatever the task's status (None when never recorded); ``stopping_attempt_homes`` holds the homes of this member's attempts whose task row is still that attempt (same generation) in ``stopping`` or ``deferred``.
    store = backend.open(scope_a)
    rooms, driver = store.hosted_rooms, store.hosted_room_driver
    _room(store, "room")
    lease = driver.acquire_lease(placement="profile", room_id="room", gateway_id="gw", authority_epoch=1, process_generation="proc", ttl_seconds=600.0, now=100.0)
    task = _driver_task(store, 1)
    attempt = driver.start_task(task, lease, placement="profile", expected_cancel_generation=0, now=110.0)
    generation = attempt.execution_generation
    rooms.record_attempt_home(room_id="room", member_id="m1", task_id="task-1", generation=generation, home="/homes/old")
    rooms.record_attempt_home(room_id="room", member_id="m1", task_id="task-1", generation=generation, home="/homes/new")
    assert rooms.attempt_home(room_id="room", member_id="m1", task_id="task-1", generation=generation) == "/homes/old"
    assert rooms.attempt_home(room_id="room", member_id="m1", task_id="task-1", generation=generation + 1) is None
    assert rooms.stopping_attempt_homes(room_id="room", member_id="m1") == ()
    stopping = driver.begin_task_cancel(task, placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=115.0)
    assert rooms.stopping_attempt_homes(room_id="room", member_id="m1") == ("/homes/old",)
    assert rooms.stopping_attempt_homes(room_id="room", member_id="m2") == ()
    driver.defer_unconfirmed_stop(task, lease, placement="profile", expected_execution_generation=generation,
                                  expected_cancel_generation=stopping["cancel_generation"], reason="unconfirmed", now=116.0)
    assert rooms.stopping_attempt_homes(room_id="room", member_id="m1") == ("/homes/old",)
    driver.cancel_task(task, placement="profile", cancel_id="cancel-1", expected_cancel_generation=stopping["cancel_generation"], now=117.0)
    assert rooms.stopping_attempt_homes(room_id="room", member_id="m1") == ()
    assert rooms.attempt_home(room_id="room", member_id="m1", task_id="task-1", generation=generation) == "/homes/old"


def test_recording_an_attempt_home_drops_the_members_attempts_that_are_over(backend, scope_a):
    # CONTRACT: a ``record_attempt_home`` write first drops this member's attempt records whose task row is gone, at another generation or outside ``running``, ``stopping``, ``deferred`` and ``indeterminate``; live attempts and other members' records stay.
    store = backend.open(scope_a)
    rooms, driver = store.hosted_rooms, store.hosted_room_driver
    _room(store, "room")
    lease = driver.acquire_lease(placement="profile", room_id="room", gateway_id="gw", authority_epoch=1, process_generation="proc", ttl_seconds=600.0, now=100.0)
    over = driver.start_task(_driver_task(store, 1), lease, placement="profile", expected_cancel_generation=0, now=110.0)
    driver.settle_task(over, placement="profile", settlement_id="settle-1", status="settled", result={"text": "done"}, now=111.0)
    live = driver.start_task(_driver_task(store, 2), lease, placement="profile", expected_cancel_generation=0, now=112.0)
    attempts = (("task-1", over.execution_generation), ("task-2", live.execution_generation), ("task-2", live.execution_generation + 1), ("gone", 1))
    for task_id, generation in attempts:
        rooms.record_attempt_home(room_id="room", member_id="m1", task_id=task_id, generation=generation, home=f"/homes/{task_id}/{generation}")
    rooms.record_attempt_home(room_id="room", member_id="m2", task_id="gone", generation=1, home="/homes/m2")
    rooms.record_attempt_home(room_id="room", member_id="m1", task_id="task-3", generation=1, home="/homes/next")
    kept = {attempt: rooms.attempt_home(room_id="room", member_id="m1", task_id=attempt[0], generation=attempt[1]) for attempt in (*attempts, ("task-3", 1))}
    assert kept == {("task-1", over.execution_generation): None, ("task-2", live.execution_generation): f"/homes/task-2/{live.execution_generation}",
                    ("task-2", live.execution_generation + 1): None, ("gone", 1): None, ("task-3", 1): "/homes/next"}
    assert rooms.attempt_home(room_id="room", member_id="m2", task_id="gone", generation=1) == "/homes/m2"
    assert rooms.member_homes(room_id="room", member_id="m1") == ()


def test_disband_waits_for_unexpired_held_grants_and_drops_expired_ones(backend, scope_a):
    store = backend.open(scope_a)
    _room(store, "room")
    peers = store.hosted_room_peers
    peers.hold_room_grant_revoke(placement="profile", room_id="room", member_id="m2", target_url="u", grant="g-soon", expires_at=160.0, now=110.0)
    peers.hold_room_grant_revoke(placement="profile", room_id="room", member_id="m2", target_url="u", grant="g-late", expires_at=200.0, now=110.0)
    store.hosted_rooms.record_member_home(room_id="room", member_id="m1", home="/homes/a")
    with pytest.raises(RoomHeldGrantsPendingError, match="Group chat grants still need revoking; retry Disband.") as refused:
        store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=170.0)
    assert refused.value.reason == "room_disbanding_retry"
    assert store.hosted_rooms.room_state(placement="profile", room_id="room")["latest_seq"] == 0
    assert store.hosted_rooms.member_homes(room_id="room", member_id="m1") == ("/homes/a",)
    assert sorted(row["grant"] for row in peers.held_room_grant_revokes(placement="profile", room_id="room", now=150.0)) == ["g-late", "g-soon"]
    result = store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=200.0)
    assert result["idempotent"] is False
    assert peers.owes_room_grant_revokes(placement="profile") is False
    assert store.hosted_rooms.member_homes(room_id="room", member_id="m1") == ()
