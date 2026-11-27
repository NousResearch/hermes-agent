"""Contract cases for the hand-off verbs of ``persistence.gateway.hosted_rooms.HostedRoomRepository``: the rooms a standalone backend fenced in ``shared-state.db``, the hand-off record, and the one-write copy and claim of those rooms into the bound profile's ``state.db``.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs: standalone rooms are created and fenced on ``placement='shared'`` with ``create_room``, ``append_event`` and ``demote_room``; epochs are begun through the backend's ``begin_epoch`` seam.
"""

from __future__ import annotations

import pytest

from persistence import FenceStale, ProfileScope
from persistence.core.errors import PermanentStoreError
from persistence.gateway.hosted_rooms import RoomLinkRecord

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
USER = {"kind": "user", "id": "u1"}
LOCAL = "install:local"
HANDOFF = "install:local:gateway-store"


def _standalone(store, room_id: str, *, now: float = 100.0) -> None:
    """A room the standalone backend created and wrote in the shared file under its own authority."""
    store.hosted_rooms.create_room(
        placement="shared", room_id=room_id, name=room_id, members=MEMBERS, authority_gateway_id=LOCAL, now=now)
    store.hosted_rooms.append_event(
        placement="shared", room_id=room_id, event_id=f"user:{room_id}", kind="message.user", actor=USER,
        payload={"text": "hello"}, authority_gateway_id=LOCAL, authority_epoch=1, now=now)


def _fence(store, room_id: str) -> None:
    store.hosted_room_replicas.demote_room(
        placement="shared", room_id=room_id, observed_gateway_id=HANDOFF, observed_epoch=2, local_gateway_id=LOCAL,
        now=110.0)


def _import(store, room_ids, *, epoch: int | None, now: float = 200.0) -> int:
    return store.hosted_rooms.import_fenced_rooms(
        epoch=epoch, room_ids=frozenset(room_ids), owner="alice", local_gateway_id=LOCAL, handoff_gateway_id=HANDOFF,
        now=now)


def _epoch(backend, scope: ProfileScope) -> int:
    return backend.begin_epoch(scope, "instance-contract")


# ── the hand-off record ───────────────────────────────────────────────────────


def test_fenced_rooms_are_the_live_rooms_on_the_handoff_authority_not_yet_recorded(backend, scope_a):
    store = backend.open(scope_a)
    for room_id in ("fenced", "gone", "standalone"):
        _standalone(store, room_id)
    _fence(store, "fenced")
    _fence(store, "gone")
    store.hosted_rooms.disband_room(placement="shared", room_id="gone", expected_gateway_id=HANDOFF, expected_epoch=2,
                                    now=120.0)
    store.hosted_rooms.create_room(placement="profile", room_id="elsewhere", name="Elsewhere", members=MEMBERS,
                                   authority_gateway_id=HANDOFF, now=100.0)
    # CONTRACT: both reads act on shared-state.db only; a disbanded fenced room is handed but never offered again.
    assert store.hosted_rooms.fenced_room_ids(handoff_gateway_id=HANDOFF) == frozenset({"fenced"})
    assert store.hosted_rooms.handed_room_ids(handoff_gateway_id=HANDOFF) == frozenset({"fenced", "gone"})
    assert store.hosted_rooms.fenced_room_ids(handoff_gateway_id=LOCAL) == frozenset({"standalone"})
    store.hosted_rooms.record_room_handoff(room_ids=frozenset({"fenced"}), now=130.0)
    # CONTRACT: a recorded room leaves the fenced set and stays handed; recording it again is a no-op.
    assert store.hosted_rooms.fenced_room_ids(handoff_gateway_id=HANDOFF) == frozenset()
    store.hosted_rooms.record_room_handoff(room_ids=frozenset({"fenced"}), now=140.0)
    assert store.hosted_rooms.handed_room_ids(handoff_gateway_id=HANDOFF) == frozenset({"fenced", "gone"})


def test_a_new_install_has_nothing_fenced_or_handed(backend, scope_a):
    store = backend.open(scope_a)
    assert store.hosted_rooms.fenced_room_ids(handoff_gateway_id=HANDOFF) == frozenset()
    assert store.hosted_rooms.handed_room_ids(handoff_gateway_id=HANDOFF) == frozenset()
    store.hosted_rooms.record_room_handoff(room_ids=frozenset(), now=100.0)


# ── the copy and claim ────────────────────────────────────────────────────────


def test_import_copies_the_room_with_its_history_then_claims_and_owns_it(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    _standalone(store, "desk")
    _fence(store, "desk")

    assert _import(store, {"desk"}, epoch=epoch) == 1

    room = store.hosted_rooms.room_state(placement="profile", room_id="desk")
    claim = {
        "room_id": "desk", "seq": 3, "event_id": "system:authority-claimed:3", "kind": "authority.claimed",
        "actor": {"kind": "system", "id": "authority-control"}, "authority_epoch": 3,
        "payload": {"previous_gateway_id": HANDOFF, "authority_gateway_id": LOCAL, "authority_epoch": 3},
        "created_at": 200.0, "idempotent": False}
    # CONTRACT: the copy carries the shared history; the claim is the next event at epoch + 1 and moves the room to the local authority.
    assert room == {
        "room_id": "desk", "name": "desk", "members": MEMBERS, "authority_gateway_id": LOCAL, "authority_epoch": 3,
        "revision": 3, "created_at": 100.0, "updated_at": 200.0, "idempotent": False, "latest_seq": 3,
        "authority_claim": claim}
    events = store.hosted_rooms.read_events(placement="profile", room_id="desk")["events"]
    assert [event["kind"] for event in events] == ["message.user", "authority.lost", "authority.claimed"]
    assert events[-1] == claim
    assert store.hosted_rooms.owner("desk") == "alice"
    # CONTRACT: shared-state.db is only read: the source stays fenced to the hand-off authority.
    source = store.hosted_rooms.room_state(placement="shared", room_id="desk")
    assert (source["authority_gateway_id"], source["authority_epoch"], source["latest_seq"]) == (HANDOFF, 2, 2)


def test_import_skips_every_room_the_profile_has_known(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _epoch(backend, scope_a)
    for room_id in ("desk", "gone"):
        _standalone(store, room_id)
        _fence(store, room_id)
    assert _import(store, {"desk"}, epoch=epoch) == 1
    before = store.hosted_rooms.read_events(placement="profile", room_id="desk")
    # CONTRACT: a room the profile holds is never copied or claimed again.
    assert _import(store, {"desk"}, epoch=epoch, now=300.0) == 0
    assert store.hosted_rooms.read_events(placement="profile", room_id="desk") == before
    store.hosted_rooms.create_room(placement="profile", room_id="gone", name="gone", members=MEMBERS,
                                   authority_gateway_id=LOCAL, now=100.0)
    store.hosted_rooms.disband_room(placement="profile", room_id="gone", expected_gateway_id=LOCAL, expected_epoch=1,
                                    now=150.0)
    assert store.hosted_rooms.prune_disbanded_rooms(placement="profile", now=150.0 + 400 * 86400) == 1
    # CONTRACT: a retired id with no row left is known too: a room disbanded in the profile never comes back.
    assert _import(store, {"gone"}, epoch=epoch) == 0
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True)] == ["desk"]
    assert store.hosted_rooms.owner("gone") is None


def test_import_is_fenced_on_the_writer_epoch_and_none_skips_the_fence(backend, scope_a):
    store = backend.open(scope_a)
    stale = _epoch(backend, scope_a)
    _epoch(backend, scope_a)
    _standalone(store, "desk")
    _fence(store, "desk")
    # CONTRACT: a stale epoch refuses before anything is copied or owned.
    with pytest.raises(FenceStale):
        _import(store, {"desk"}, epoch=stale)
    assert store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True) == []
    assert store.hosted_rooms.owner("desk") is None
    assert _import(store, {"desk"}, epoch=None) == 1
    assert store.hosted_rooms.owner("desk") == "alice"


def test_a_row_the_profile_already_holds_fails_the_whole_import(backend, scope_a):
    store = backend.open(scope_a)
    _standalone(store, "desk")
    link = RoomLinkRecord(
        room_id="desk", member_id="m1", target_url="https://peer.example/rooms", target_profile="p1", grant="grant",
        catalog_json="[]", cancellation_scope_id="scope-1", trace_id="trace-1", transport_security="tls",
        status="ready", updated_at=100.0)
    for placement in ("shared", "profile"):
        store.hosted_room_peers.upsert_room_link_record(placement=placement, record=link, max_links=8)
    _fence(store, "desk")
    # CONTRACT: the copy and the claim are one write: a conflicting child row leaves the profile as it was.
    with pytest.raises(PermanentStoreError):
        _import(store, {"desk"}, epoch=None)
    assert store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True) == []
    assert store.hosted_rooms.owner("desk") is None


def test_import_copies_only_the_named_rooms(backend, scope_a):
    store = backend.open(scope_a)
    for room_id in ("desk", "other"):
        _standalone(store, room_id)
        _fence(store, room_id)
    assert _import(store, {"desk", "unknown"}, epoch=None) == 1
    # CONTRACT: an id shared-state.db does not hold copies nothing; a room not named stays behind.
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="profile")] == ["desk"]
