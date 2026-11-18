"""Contract cases for the room and event-log verbs of ``persistence.gateway.hosted_rooms.HostedRoomRepository`` on both placements.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin the shared row steps, the retention prune and the verbs: messages, replay rules, the order of the checks, and the two files the placements name (the bound profile's ``state.db``; the install's root ``shared-state.db``, which every profile of the install reaches).
"""

from __future__ import annotations

import pytest

from persistence import Conflict, NotFound
from persistence.gateway.hosted_rooms import (
    AttachmentError,
    AuthorityConflictError,
    EventAttachmentConflictError,
    EventConflictError,
    EventCursorConflictError,
    HostedRoomError,
    RoomConflictError,
    RoomHistoryExpiredError,
    RoomNotFoundError,
)

USER = {"kind": "user", "id": "u1"}
MEMBERS = [{"member_id": "m1", "profile": "p1"}]
RETENTION_S = 90 * 24 * 60 * 60


def _create(store, room_id: str = "room", *, placement: str = "profile", name: str = "Room", members=None, gateway: str = "gw", now: float = 100.0):
    return store.hosted_rooms.create_room(placement=placement, room_id=room_id, name=name, members=MEMBERS if members is None else members, authority_gateway_id=gateway, now=now)


def _append(store, event_id: str, *, room_id: str = "room", placement: str = "profile", kind: str = "message.user", actor=None, payload=None, gateway: str = "gw", epoch: int = 1, expected=None, now: float = 110.0):
    return store.hosted_rooms.append_event(
        placement=placement, room_id=room_id, event_id=event_id, kind=kind, actor=USER if actor is None else actor,
        payload={"text": event_id} if payload is None else payload, authority_gateway_id=gateway, authority_epoch=epoch,
        expected_latest_seq=expected, now=now)


def _disband(store, room_id: str = "room", *, placement: str = "profile", gateway: str = "gw", epoch: int = 1, now: float = 200.0):
    return store.hosted_rooms.disband_room(placement=placement, room_id=room_id, expected_gateway_id=gateway, expected_epoch=epoch, now=now)


# ── create_room ───────────────────────────────────────────────────────────────


def test_create_room_replays_the_identical_room_and_refuses_any_other_state(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    replay = _create(store, now=500.0)
    assert replay["idempotent"] is True and replay["latest_seq"] == 0 and replay["created_at"] == 100.0
    for changes, message in (
        ({"name": "Other"}, "room_id already exists with different state"),
        ({"members": [{"member_id": "m9", "profile": "p9"}]}, "room_id already exists with different state"),
        ({"gateway": "gw-2"}, "room_id already belongs to a different authority"),
    ):
        with pytest.raises(RoomConflictError, match=message) as refused:
            _create(store, **changes)
        assert isinstance(refused.value, Conflict) and isinstance(refused.value, ValueError)


def test_create_room_validates_its_payload_first(backend, scope_a):
    store = backend.open(scope_a)
    for changes, message in (
        ({"room_id": "bad room"}, "invalid room_id"), ({"name": "   "}, "invalid room name"),
        ({"members": "m1"}, "members must be a list"), ({"members": ["m1"]}, "each room member must be an object"),
        ({"gateway": ""}, "invalid authority_gateway_id"),
    ):
        with pytest.raises(HostedRoomError, match=message):
            _create(store, **changes)
    assert store.hosted_rooms.list_rooms(placement="profile") == []


def test_create_room_adopts_a_legacy_room_at_the_next_epoch(backend, scope_a):
    store = backend.open(scope_a)
    _create(store, gateway="legacy", members=[{"member_id": "m1", "profile": "p1", "target": {}}])
    members = [{"member_id": "m1", "profile": "p1", "target": {"kind": "local"}}]
    adopted = _create(store, members=members, now=300.0)
    # CONTRACT: adoption may add routing metadata, claims epoch 2 with a control event and reports the room as idempotent.
    assert (adopted["authority_gateway_id"], adopted["authority_epoch"], adopted["revision"], adopted["latest_seq"]) == ("gw", 2, 2, 1)
    assert adopted["idempotent"] is True and adopted["adopted"] is True and adopted["members"] == members
    claim = adopted["claim_event"]
    assert (claim["seq"], claim["event_id"], claim["kind"], claim["authority_epoch"], claim["idempotent"]) == (1, "system:authority-adopted", "authority.claimed", 2, False)
    assert claim["actor"] == {"kind": "system", "id": "authority-control"}
    assert claim["payload"] == {"previous_gateway_id": "legacy", "authority_gateway_id": "gw", "authority_epoch": 2}
    assert store.hosted_rooms.room_state(placement="profile", room_id="room")["authority_claim"] == claim


def test_the_two_placements_are_two_files_and_the_shared_one_spans_profiles(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    _create(store_a, "shared-room", placement="shared")
    _create(store_a, "profile-room")
    # CONTRACT: 'shared' is the install's root shared-state.db, reached by every profile; 'profile' is the bound profile's state.db.
    assert [room["room_id"] for room in store_b.hosted_rooms.list_rooms(placement="shared")] == ["shared-room"]
    assert store_b.hosted_rooms.list_rooms(placement="profile") == []
    assert [room["room_id"] for room in store_a.hosted_rooms.list_rooms(placement="profile")] == ["profile-room"]
    # CONTRACT: one room id lives independently in each file.
    _create(store_b, "shared-room", name="Profile copy")
    assert store_b.hosted_rooms.room_state(placement="profile", room_id="shared-room")["name"] == "Profile copy"
    assert store_b.hosted_rooms.room_state(placement="shared", room_id="shared-room")["name"] == "Room"
    _append(store_b, "e1", room_id="shared-room", placement="shared")
    assert store_a.hosted_rooms.read_events(placement="shared", room_id="shared-room")["latest_seq"] == 1
    assert store_b.hosted_rooms.read_events(placement="profile", room_id="shared-room")["latest_seq"] == 0


# ── rename_room ───────────────────────────────────────────────────────────────


def test_rename_room_updates_the_room_and_appends_its_event(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    renamed = store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="rename-1", name="Renamed", now=120.0)
    assert (renamed["name"], renamed["revision"], renamed["latest_seq"], renamed["updated_at"], renamed["idempotent"]) == ("Renamed", 2, 1, 120.0, False)
    event = renamed["event"]
    assert (event["seq"], event["kind"], event["actor"], event["payload"], event["authority_epoch"]) == (1, "room.renamed", {"kind": "system", "id": "room-control"}, {"name": "Renamed"}, 1)
    # CONTRACT: the same rename replays the current room with the stored event; another name under that event id conflicts.
    replay = store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="rename-1", name="Renamed", now=130.0)
    assert replay["idempotent"] is True and replay["event"] == {**event, "idempotent": True} and replay["updated_at"] == 120.0
    with pytest.raises(EventConflictError, match="event_id already exists with different immutable content"):
        store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="rename-1", name="Other", now=130.0)
    with pytest.raises(EventConflictError):
        _append(store, "rename-1")


def test_rename_room_refuses_missing_disbanded_and_expired_rooms(backend, scope_a):
    store = backend.open(scope_a)
    with pytest.raises(RoomNotFoundError, match="hosted room not found") as missing:
        store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="r", name="N")
    assert isinstance(missing.value, NotFound)
    _create(store)
    _disband(store, now=200.0)
    with pytest.raises(RoomNotFoundError, match="hosted room not found") as disbanded:
        store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="r", name="N")
    assert not isinstance(disbanded.value, RoomHistoryExpiredError)
    assert store.hosted_rooms.prune_disbanded_rooms(placement="profile", now=200.0 + RETENTION_S) == 1
    with pytest.raises(RoomHistoryExpiredError, match="Group Chat history expired; room_id remains permanently retired") as expired:
        store.hosted_rooms.rename_room(placement="profile", room_id="room", event_id="r", name="N")
    assert expired.value.reason == "room_history_expired"


# ── append_event ──────────────────────────────────────────────────────────────


def test_append_event_allocates_sequences_and_replays_identical_content(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    actor = {"kind": "user", "id": "u1", "display_name": "Ünï"}
    first = _append(store, "e1", actor=actor, payload={"text": "héllo", "n": 1}, now=110.0)
    assert first == {
        "room_id": "room", "seq": 1, "event_id": "e1", "kind": "message.user", "actor": actor, "authority_epoch": 1,
        "payload": {"text": "héllo", "n": 1}, "created_at": 110.0, "idempotent": False}
    assert _append(store, "e2", now=111.0)["seq"] == 2
    replay = _append(store, "e1", actor=actor, payload={"n": 1, "text": "héllo"}, now=500.0)
    assert replay == {**first, "idempotent": True}
    with pytest.raises(EventConflictError, match="event_id already exists with different content"):
        _append(store, "e1", actor=actor, payload={"text": "other"})
    room = store.hosted_rooms.room_state(placement="profile", room_id="room")
    assert (room["latest_seq"], room["updated_at"], room["revision"]) == (2, 111.0, 1)


def test_append_event_replays_after_the_room_is_disbanded(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    first = _append(store, "e1")
    _disband(store)
    # CONTRACT: the stored event is read before the room, so an exact replay survives the disband; a new event does not.
    assert _append(store, "e1") == {**first, "idempotent": True}
    with pytest.raises(RoomNotFoundError, match="hosted room not found"):
        _append(store, "e2")


def test_append_event_checks_authority_before_the_cursor(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    _append(store, "e1")
    for changes in ({"gateway": "gw-2", "actor": USER}, {"epoch": 2}):
        with pytest.raises(AuthorityConflictError, match="stale hosted room authority") as stale:
            _append(store, "e2", expected=7, **changes)
        assert stale.value.reason == "authority_conflict"
    with pytest.raises(EventCursorConflictError, match="room changed before event publication"):
        _append(store, "e2", expected=0)
    assert _append(store, "e2", expected=1)["seq"] == 2


def _attachment(store, *, event_id: str = "e1", viewer: bool = True, upload_id: str = "up-1", now: float = 105.0) -> dict:
    """One upload committed to ``event_id`` (room-visible and held until the event when ``viewer``), as ``append_user_event`` commits it."""
    meta = store.hosted_room_attachments.put(placement="profile", room_id="room", upload_id=upload_id, kind="file", name="a.txt", mime="text/plain", data=b"hello", now=now)
    entry = {key: meta[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
    store.hosted_room_attachments.commit_message_with_receipt(
        placement="profile", room_id="room", event_id=event_id, manifest=[entry], recipient_member_ids=["m1"], viewer_access=viewer, hold_until_event=viewer, now=now)
    return entry


def test_append_event_refuses_message_attachments_that_are_not_committed(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    # The user-message path opens the attachment store before it appends, so the attachment tables exist.
    store.hosted_room_attachments.put(placement="profile", room_id="room", upload_id="up-other", kind="file", name="b.txt", mime="text/plain", data=b"other", now=105.0)
    attachment = {"attachment_id": "att_" + "0" * 32, "kind": "file", "name": "a.txt", "size": 1, "mime": "text/plain"}
    with pytest.raises(EventAttachmentConflictError, match="attachment commitment changed before event publication") as refused:
        _append(store, "e1", payload={"text": "x", "attachments": [attachment]})
    assert isinstance(refused.value, EventCursorConflictError)
    assert store.hosted_rooms.read_events(placement="profile", room_id="room")["events"] == []
    assert _append(store, "e1", payload={"text": "x", "attachments": []})["seq"] == 1


def test_append_event_retains_the_commitments_its_message_publishes(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    entry = _attachment(store)
    event = _append(store, "e1", payload={"text": "x", "attachments": [entry]})
    assert event["payload"]["attachments"] == [entry]
    # CONTRACT: the held commitment loses its expiry in the append's own write.
    read = store.hosted_room_attachments.read(placement="profile", room_id="room", attachment_id=entry["attachment_id"], recipient_member_id="m1", event_id="e1", now=10.0 ** 9)
    assert read.data == b"hello"
    assert _append(store, "e1", payload={"text": "x", "attachments": [entry]}, now=10.0 ** 9)["idempotent"] is True


@pytest.mark.parametrize("kind", ["message.user", "message.member"])
def test_append_event_refuses_a_message_whose_commitment_does_not_match(backend, scope_a, kind):
    store = backend.open(scope_a)
    _create(store)
    other_event, hidden = _attachment(store, event_id="e0"), _attachment(store, viewer=False, upload_id="up-2")
    held = _attachment(store, upload_id="up-3")
    for entries, now in (([other_event], 110.0), ([hidden], 110.0), ([{**held, "name": "b.txt"}], 110.0), ([held], 105.0 + 60 * 60)):
        with pytest.raises(EventAttachmentConflictError, match="attachment commitment changed before event publication"):
            _append(store, "e1", kind=kind, payload={"text": "x", "attachments": entries}, now=now)
    assert store.hosted_rooms.read_events(placement="profile", room_id="room")["events"] == []


def test_append_event_validates_the_published_manifest_and_requires_its_canonical_form(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    entry = _attachment(store)
    cases = (
        ("x", "attachments must be a list"),
        ([{**entry, "attachment_id": "att-1"}], "invalid attachment_id"),
        ([{**entry, "extra": 1}], "attachment manifest has unknown fields: extra"),
        ([entry, entry], "attachment ids must be unique within one message"),
        ([{**entry, "name": " a.txt"}], "published attachment metadata must be canonical"),
        ([{**entry, "mime": "TEXT/PLAIN"}], "published attachment metadata must be canonical"))
    for attachments, message in cases:
        with pytest.raises(AttachmentError, match=message) as refused:
            _append(store, "e1", payload={"text": "x", "attachments": attachments})
        assert not isinstance(refused.value, EventCursorConflictError)
    # Other kinds carry no attachment step.
    assert _append(store, "e2", kind="room.activity", payload={"attachments": "x"})["seq"] == 1


def test_append_event_checks_authority_and_cursor_before_attachments(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    payload = {"text": "x", "attachments": "not a list"}
    with pytest.raises(AuthorityConflictError, match="stale hosted room authority"):
        _append(store, "e1", payload=payload, epoch=2)
    with pytest.raises(EventCursorConflictError, match="room changed before event publication"):
        _append(store, "e1", payload=payload, expected=3)
    with pytest.raises(AttachmentError, match="attachments must be a list"):
        _append(store, "e1", payload=payload, expected=0)


def test_append_event_on_a_missing_or_expired_room(backend, scope_a):
    store = backend.open(scope_a)
    with pytest.raises(RoomNotFoundError, match="hosted room not found"):
        _append(store, "e1")
    _create(store)
    _disband(store, now=200.0)
    store.hosted_rooms.prune_disbanded_rooms(placement="profile", now=200.0 + RETENTION_S)
    with pytest.raises(RoomHistoryExpiredError):
        _append(store, "e1")


def test_append_event_control_reserve_admits_a_gateway_notice_into_a_full_file(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    # Fill the file's event budget: halve the payload at each refusal until a one-character message is refused.
    size, index = 256 * 1024 - 64, 0
    while size:
        try:
            _append(store, f"fill-{index}", payload={"text": "x" * size}, now=110.0 + index)
            index += 1
        except HostedRoomError as exc:
            assert "Group Chat storage is full on this host" in str(exc)
            size //= 2
    gateway = {"kind": "gateway", "id": "gw"}
    notice = {"error": "Members could not be woken: boom", "reason_code": "prepare_failed"}
    with pytest.raises(HostedRoomError, match="Group Chat storage is full on this host"):
        _append(store, "notice", kind="member.unavailable", actor=gateway, payload=notice)
    # CONTRACT: control_reserve lets a gateway actor's event use the control headroom; another actor's stays refused.
    with pytest.raises(HostedRoomError, match="Group Chat storage is full on this host"):
        store.hosted_rooms.append_event(
            placement="profile", room_id="room", event_id="user-reserve", kind="message.user", actor=USER,
            payload={"text": "x"}, authority_gateway_id="gw", authority_epoch=1, control_reserve=True)
    event = store.hosted_rooms.append_event(
        placement="profile", room_id="room", event_id="notice", kind="member.unavailable", actor=gateway,
        payload=notice, authority_gateway_id="gw", authority_epoch=1, control_reserve=True, now=500.0)
    assert (event["seq"], event["kind"], event["actor"], event["payload"]) == (index + 1, "member.unavailable", gateway, notice)


# ── disband_room and prune_disbanded_rooms ────────────────────────────────────


def test_disband_room_tombstones_once_and_replays_without_the_fence(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    _append(store, "e1")
    with pytest.raises(AuthorityConflictError, match="stale hosted room authority"):
        _disband(store, epoch=2)
    result = _disband(store, now=200.0)
    event = result["event"]
    assert (result["room_id"], result["disbanded_at"], result["idempotent"]) == ("room", 200.0, False)
    assert (event["seq"], event["event_id"], event["kind"], event["payload"], event["actor"]) == (2, "system:room-disbanded", "room.disbanded", {"room_id": "room"}, {"kind": "system", "id": "room-control"})
    # CONTRACT: replays bypass the authority fence and return the tombstone with the stored event.
    replay = _disband(store, gateway="other", epoch=9, now=300.0)
    assert replay == {"room_id": "room", "disbanded_at": 200.0, "idempotent": True, "event": {**event, "idempotent": True}}
    state = store.hosted_rooms.room_state(placement="profile", room_id="room", include_disbanded=True)
    assert (state["disbanded_at"], state["latest_seq"], state["revision"]) == (200.0, 2, 2)
    with pytest.raises(RoomNotFoundError):
        store.hosted_rooms.room_state(placement="profile", room_id="room")
    with pytest.raises(RoomConflictError, match="room_id belongs to a disbanded room"):
        _create(store)
    with pytest.raises(RoomNotFoundError, match="hosted room not found"):
        _disband(store, "never")


def test_prune_purges_old_tombstones_and_keeps_their_ids_retired(backend, scope_a):
    store = backend.open(scope_a)
    for room_id, disbanded_at in (("old", 1000.0), ("young", 5000.0)):
        _create(store, room_id)
        _disband(store, room_id, now=disbanded_at)
    _create(store, "live")
    assert store.hosted_rooms.prune_disbanded_rooms(placement="profile", now=1000.0 + RETENTION_S - 1) == 0
    assert store.hosted_rooms.prune_disbanded_rooms(placement="profile", now=1000.0 + RETENTION_S) == 1
    assert sorted(room["room_id"] for room in store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True)) == ["live", "young"]
    # CONTRACT: the purged id stays retired: its history is expired, its disband replays as expired, it is never created again.
    with pytest.raises(RoomHistoryExpiredError):
        store.hosted_rooms.room_state(placement="profile", room_id="old", include_disbanded=True)
    assert _disband(store, "old", now=9.0) == {"room_id": "old", "disbanded_at": 1000.0, "idempotent": True, "history_expired": True}
    with pytest.raises(RoomConflictError, match="room_id belongs to a disbanded room"):
        _create(store, "old")
    assert store.hosted_rooms.prune_disbanded_rooms(placement="shared", now=1000.0 + RETENTION_S) == 0


def test_disband_keeps_only_the_newest_tombstones(backend, scope_a):
    store = backend.open(scope_a)
    for index in range(513):
        _create(store, f"room-{index:03d}", members=[], now=float(index))
        _disband(store, f"room-{index:03d}", now=1000.0 + index)
    # CONTRACT: the disband's own prune keeps the newest 512 tombstones; the oldest loses its history and keeps its id.
    with pytest.raises(RoomHistoryExpiredError):
        store.hosted_rooms.room_state(placement="profile", room_id="room-000", include_disbanded=True)
    assert store.hosted_rooms.room_state(placement="profile", room_id="room-001", include_disbanded=True)["disbanded_at"] == 1001.0


# ── reads ─────────────────────────────────────────────────────────────────────


def test_list_rooms_pages_by_latest_change_then_room_id(backend, scope_a):
    store = backend.open(scope_a)
    for room_id, now in (("b", 100.0), ("a", 100.0), ("c", 300.0), ("d", 50.0)):
        _create(store, room_id, now=now)
    _disband(store, "d", now=400.0)
    rooms = store.hosted_rooms.list_rooms(placement="profile")
    assert [room["room_id"] for room in rooms] == ["c", "a", "b"]
    assert rooms[0] == {
        "room_id": "c", "name": "Room", "members": MEMBERS, "authority_gateway_id": "gw", "authority_epoch": 1, "revision": 1,
        "created_at": 300.0, "updated_at": 300.0, "idempotent": False, "latest_seq": 0}
    every = store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True)
    assert [room["room_id"] for room in every] == ["d", "c", "a", "b"] and every[0]["disbanded_at"] == 400.0
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True, limit=2, offset=1)] == ["c", "a"]
    assert store.hosted_rooms.list_rooms(placement="profile", offset=3) == []


def test_room_state_carries_the_claim_of_the_current_epoch_only(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    state = store.hosted_rooms.room_state(placement="profile", room_id="room")
    assert "authority_claim" not in state and state["latest_seq"] == 0
    _create(store, "legacy-room", gateway="legacy")
    _create(store, "legacy-room", now=150.0)
    claimed = store.hosted_rooms.room_state(placement="profile", room_id="legacy-room")
    assert claimed["authority_epoch"] == 2 and claimed["authority_claim"]["event_id"] == "system:authority-adopted"
    with pytest.raises(RoomNotFoundError, match="hosted room not found"):
        store.hosted_rooms.room_state(placement="shared", room_id="room")


def test_read_events_returns_a_monotonic_page(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    for index in range(1, 6):
        _append(store, f"e{index}", now=110.0 + index)
    page = store.hosted_rooms.read_events(placement="profile", room_id="room", since_seq=1, limit=2)
    assert [event["seq"] for event in page["events"]] == [2, 3]
    assert (page["cursor"], page["latest_seq"], page["has_more"], page["authority"]) == (3, 5, True, {"gateway_id": "gw", "epoch": 1})
    tail = store.hosted_rooms.read_events(placement="profile", room_id="room", since_seq=5)
    assert (tail["events"], tail["cursor"], tail["has_more"]) == ([], 5, False)
    with pytest.raises(HostedRoomError, match="since_seq is ahead of the hosted room log"):
        store.hosted_rooms.read_events(placement="profile", room_id="room", since_seq=6)
    _disband(store)
    with pytest.raises(RoomNotFoundError):
        store.hosted_rooms.read_events(placement="profile", room_id="room")
    assert store.hosted_rooms.read_events(placement="profile", room_id="room", include_disbanded=True)["latest_seq"] == 6


def test_read_events_stays_within_the_page_byte_budget(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    blob = "x" * (250 * 1024)
    for index in range(1, 10):
        _append(store, f"big-{index}", payload={"text": blob})
    page = store.hosted_rooms.read_events(placement="profile", room_id="room", limit=100)
    # CONTRACT: a page holds at most 2 MiB of events; the rest follows from its cursor.
    assert 0 < len(page["events"]) < 9 and page["has_more"] is True and page["cursor"] == len(page["events"])
    rest = store.hosted_rooms.read_events(placement="profile", room_id="room", since_seq=page["cursor"], limit=100)
    assert rest["events"][0]["seq"] == page["cursor"] + 1


def test_probe_hosted_room_sees_live_rooms_of_its_file_only(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    # CONTRACT: a file never opened answers False without being created.
    assert store_b.hosted_rooms.probe_hosted_room(placement="profile", room_id="room") is False
    assert store_a.hosted_rooms.probe_hosted_room(placement="shared", room_id="room") is False
    _create(store_a, placement="shared")
    assert store_b.hosted_rooms.probe_hosted_room(placement="shared", room_id="room") is True
    assert store_a.hosted_rooms.probe_hosted_room(placement="profile", room_id="room") is False
    _disband(store_a, placement="shared")
    assert store_b.hosted_rooms.probe_hosted_room(placement="shared", room_id="room") is False


# ── room settings and single-event reads ──────────────────────────────────────

SYSTEM = {"kind": "system", "id": "room-control"}
SETTINGS = "room.context_set"


def _settings(store, event_id: str, *, purpose: str = "Ship", notes: str = "", routing=None, lead=None, seated=("m1",), gateway: str = "gw", epoch: int = 1, now: float = 120.0):
    return store.hosted_rooms.save_room_settings(
        placement="profile", room_id="room", event_id=event_id, purpose=purpose, notes=notes, routing=routing,
        lead_member_id=lead, seated_member_ids=seated, authority_gateway_id=gateway, authority_epoch=epoch, now=now)


def _latest(store, **changes):
    return store.hosted_rooms.latest_event_payload(**{"placement": "profile", "room_id": "room", "kind": SETTINGS, **changes})


def test_latest_event_payload_reads_the_newest_event_of_a_kind_before_a_sequence(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    assert _latest(store) is None
    _append(store, "c1", kind=SETTINGS, actor=SYSTEM, payload={"purpose": "one", "notes": ""})
    _append(store, "u1")
    _append(store, "c2", kind=SETTINGS, actor=SYSTEM, payload={"purpose": "two", "notes": ""})
    assert _latest(store) == {"purpose": "two", "notes": ""}
    # CONTRACT: before_seq reads the newest event strictly below that sequence.
    assert _latest(store, before_seq=3) == {"purpose": "one", "notes": ""}
    assert _latest(store, before_seq=1) is None
    assert _latest(store, kind="message.user") == {"text": "u1"}
    # CONTRACT: the read needs no room row: a room it does not know has no such event.
    assert _latest(store, room_id="nope") is None


def test_event_by_id_reads_one_stored_event_of_a_live_or_disbanded_room(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    event = _append(store, "e1")
    assert store.hosted_rooms.event_by_id(placement="profile", room_id="room", event_id="e1") == event
    assert store.hosted_rooms.event_by_id(placement="profile", room_id="room", event_id="e2") is None
    _disband(store)
    assert store.hosted_rooms.event_by_id(placement="profile", room_id="room", event_id="e1") == event
    assert store.hosted_rooms.event_by_id(placement="shared", room_id="room", event_id="e1") is None


def test_save_room_settings_appends_one_settings_event_with_the_resolved_record(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    saved = _settings(store, "c1", purpose="  Ship   the release ", notes=" Friday \n")
    # CONTRACT: a room with no settings resolves omitted routing to everyone with no lead; the record is normalized.
    record = {"purpose": "Ship the release", "notes": "Friday", "routing": "everyone", "lead_member_id": ""}
    assert saved["context"] == record
    event = saved["event"]
    assert (event["seq"], event["kind"], event["actor"], event["payload"], event["authority_epoch"], event["idempotent"]) == (1, SETTINGS, SYSTEM, record, 1, False)
    assert _latest(store) == record
    assert store.hosted_rooms.event_by_id(placement="profile", room_id="room", event_id="c1") == event


def test_save_room_settings_keeps_the_routing_a_save_omits(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    _settings(store, "c1", routing="lead", lead="m1")
    # CONTRACT: omitted routing fields take the newest settings event's values inside the write.
    saved = _settings(store, "c2", purpose="New purpose")
    assert saved["context"] == {"purpose": "New purpose", "notes": "", "routing": "lead", "lead_member_id": "m1"}
    # CONTRACT: everyone routing never keeps a lead.
    assert _settings(store, "c3", routing="everyone")["context"]["lead_member_id"] == ""


def test_save_room_settings_accepts_auto_routing_with_no_lead(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    saved = _settings(store, "c1", routing="auto", lead="m1")
    # CONTRACT: auto is a routing mode; like everyone it keeps no lead.
    assert saved["context"] == {"purpose": "Ship", "notes": "", "routing": "auto", "lead_member_id": ""}
    assert _latest(store) == saved["context"]


def test_save_room_settings_validates_the_resolved_record_and_the_lead(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    for changes, message in (
        ({"purpose": 1}, "purpose, notes, routing and lead_member_id must be strings"),
        ({"purpose": "x" * 501}, "purpose must be at most 500 characters"),
        ({"notes": "x" * (4 * 1024 + 1)}, "notes must be at most 4096 bytes"),
        ({"routing": "some"}, "routing must be 'everyone', 'lead' or 'auto'"),
        ({"routing": "lead"}, "lead routing needs a lead_member_id"),
        ({"routing": "lead", "lead": "m9"}, "the lead must be a member of this Group Chat"),
    ):
        with pytest.raises(HostedRoomError, match=f"^{message}$"):
            _settings(store, "c1", **changes)
    assert _latest(store) is None
    with pytest.raises(AuthorityConflictError, match="stale hosted room authority"):
        _settings(store, "c1", epoch=2)


def test_a_retried_settings_save_returns_its_first_event_in_its_own_fields(backend, scope_a):
    store = backend.open(scope_a)
    _create(store)
    first = _settings(store, "c-notes", purpose="Notes only")
    _settings(store, "c-lead", routing="lead", lead="m1")
    # CONTRACT: a retry resolves omitted fields from its own first event, so it replays after a routing change.
    retry = _settings(store, "c-notes", purpose="Notes only", seated=())
    assert retry["event"] == {**first["event"], "idempotent": True}
    assert retry["context"] == first["context"]
    with pytest.raises(EventConflictError):
        _settings(store, "c-notes", purpose="Other")
    # CONTRACT: a first event saved before routing existed is compared in its own fields.
    old = _append(store, "c-old", kind=SETTINGS, actor=SYSTEM, payload={"purpose": "Old", "notes": ""})
    assert _settings(store, "c-old", purpose="Old")["event"] == {**old, "idempotent": True}
    with pytest.raises(EventConflictError):
        _settings(store, "c-old", purpose="Old", routing="lead", lead="m1")
