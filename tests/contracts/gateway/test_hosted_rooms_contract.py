"""Contract cases for the profile-bound verbs of ``persistence.gateway.hosted_rooms.HostedRoomRepository``: owner claims, transport bindings, the two member-session verbs and the hosted member check.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs; epochs are begun through the backend's ``begin_epoch`` seam. The cases cover the owner claim, the member check, the binding, the label, the member session and room creation.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, Conflict, FenceStale, ProfileScope, SessionId
from persistence.conversation.sessions import NewSession, TitleConflict
from persistence.gateway.hosted_rooms import HostedMemberGuard, HostedOwnerClaim, HostedRoomError, HostedRoomRefusal, HostedTaskIdentity
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
