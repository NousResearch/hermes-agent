"""Tests for gateway/hosted_room_replicas.py — replica ingest, promotion, and
stale-authority demotion for hosted Group Chat rooms."""

import json

import pytest

import gateway.hosted_room_replicas as replicas
from persistence.gateway.hosted_rooms import (
    HostedRoomError, ReplicaEpochRegressionError, ReplicaError, ReplicaGapError, RoomConflictError)
from persistence_assembly import open_persistence, profile_scope

USER = {"kind": "user", "id": "tek"}
MEMBERS = [{"kind": "bot", "id": "planner"}, {"kind": "bot", "id": "coder"}]

AUTH_A = "install:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
AUTH_B = "install:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

PLACEMENT = "shared"


def _open_store(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    return open_persistence(profile_scope(home=home))


@pytest.fixture
def authority_store(tmp_path):
    store = _open_store(tmp_path, "authority")
    yield store
    store.close()


@pytest.fixture
def replica_store(tmp_path):
    store = _open_store(tmp_path, "replica")
    yield store
    store.close()


def _append_event(store, *, event_id, payload, gateway_id, epoch, room_id="room-1"):
    return store.hosted_rooms.append_event(
        placement=PLACEMENT,
        room_id=room_id,
        event_id=event_id,
        kind="message.user",
        actor=USER,
        payload=payload,
        authority_gateway_id=gateway_id,
        authority_epoch=epoch,
    )


def _read_events(store, *, since_seq=0, room_id="room-1"):
    return store.hosted_rooms.read_events(placement=PLACEMENT, room_id=room_id, since_seq=since_seq, limit=100)


def _seed_room(store, *, gateway_id=AUTH_A, n_events=3, room_id="room-1"):
    store.hosted_rooms.create_room(
        placement=PLACEMENT,
        room_id=room_id,
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=gateway_id,
    )
    for index in range(n_events):
        _append_event(
            store,
            room_id=room_id,
            event_id=f"e{index}",
            payload={"text": f"msg {index} 😀"},
            gateway_id=gateway_id,
            epoch=1,
        )
    return _read_events(store, room_id=room_id)


def test_ingest_page_persists_events_and_lineage(authority_store, replica_store):
    page = _seed_room(authority_store)
    result = replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    assert result["ingested"] == 3
    assert result["stored_seq"] == 3
    assert result["caught_up"] is True
    state = replicas.replica_state(replica_store, placement=PLACEMENT, room_id="room-1")
    assert state["last_seq"] == 3
    assert state["authority"] == page["authority"]
    assert state["members"] == MEMBERS


def test_ingest_page_is_idempotent(authority_store, replica_store):
    page = _seed_room(authority_store)
    replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    again = replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    assert again["ingested"] == 0
    assert again["stored_seq"] == 3


def test_ingest_rejects_sequence_gap(authority_store, replica_store):
    _seed_room(authority_store, n_events=5)
    later = _read_events(authority_store, since_seq=2)
    with pytest.raises(ReplicaGapError):
        replicas.ingest_page(
            replica_store,
            placement=PLACEMENT,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=later,
        )


def test_ingest_rejects_epoch_regression(authority_store, replica_store):
    page = _seed_room(authority_store)
    newer = json.loads(json.dumps(page))
    newer["authority"]["epoch"] = 3
    replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=newer
    )
    stale = json.loads(json.dumps(page))
    stale["authority"]["epoch"] = 2
    with pytest.raises(ReplicaEpochRegressionError):
        replicas.ingest_page(
            replica_store,
            placement=PLACEMENT,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=stale,
        )


def test_ingest_requires_authority_stamp(authority_store, replica_store):
    page = _seed_room(authority_store)
    page.pop("authority")
    with pytest.raises(ReplicaError):
        replicas.ingest_page(
            replica_store,
            placement=PLACEMENT,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_promote_replica_continues_room_at_next_epoch(authority_store, replica_store, monkeypatch):
    page = _seed_room(authority_store)
    replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    promoted = replicas.promote_replica(replica_store, placement=PLACEMENT, room_id="room-1")
    assert promoted["authority_gateway_id"] == AUTH_B
    assert promoted["authority_epoch"] == 2
    assert promoted["previous_gateway_id"] == AUTH_A
    assert promoted["claim_seq"] == 4

    # The room is now locally authoritative with the full history + claim.
    replay = _read_events(replica_store)
    assert [e["seq"] for e in replay["events"]] == [1, 2, 3, 4]
    claim = replay["events"][-1]
    assert claim["kind"] == "authority.claimed"
    assert claim["payload"]["previous_gateway_id"] == AUTH_A
    assert claim["payload"]["authority_epoch"] == 2
    assert replay["authority"] == {"gateway_id": AUTH_B, "epoch": 2}

    # New work continues under the new epoch.
    _append_event(
        replica_store,
        event_id="post-takeover",
        payload={"text": "continuing"},
        gateway_id=AUTH_B,
        epoch=2,
    )

    # The old authority's identity/epoch is fenced out.
    with pytest.raises(HostedRoomError):
        _append_event(
            replica_store,
            event_id="stale-write",
            payload={"text": "stale"},
            gateway_id=AUTH_A,
            epoch=1,
        )

    # Replica bookkeeping is consumed by promotion.
    with pytest.raises(ReplicaError):
        replicas.replica_state(replica_store, placement=PLACEMENT, room_id="room-1")


def test_promote_refuses_when_room_exists_locally(authority_store, monkeypatch):
    page = _seed_room(authority_store)
    # Same store also holds a replica row for the same id — conflict must win.
    replicas.ingest_page(
        authority_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    with pytest.raises(RoomConflictError):
        replicas.promote_replica(authority_store, placement=PLACEMENT, room_id="room-1")


def test_promote_refuses_when_already_authority(authority_store, replica_store, monkeypatch):
    page = _seed_room(authority_store)
    replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    with pytest.raises(ReplicaError):
        replicas.promote_replica(replica_store, placement=PLACEMENT, room_id="room-1")


def test_demote_fences_stale_local_authority(authority_store, monkeypatch):
    _seed_room(authority_store)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)

    result = replicas.demote_room(
        authority_store, placement=PLACEMENT, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
    )
    assert result["idempotent"] is False
    assert result["authority_gateway_id"] == AUTH_B
    assert result["authority_epoch"] == 2

    replay = _read_events(authority_store)
    lost = replay["events"][-1]
    assert lost["kind"] == "authority.lost"
    assert lost["payload"]["authority_gateway_id"] == AUTH_B
    assert replay["authority"] == {"gateway_id": AUTH_B, "epoch": 2}

    # Local sends at the stale identity/epoch are now rejected.
    with pytest.raises(HostedRoomError):
        _append_event(
            authority_store,
            event_id="after-demote",
            payload={"text": "stale"},
            gateway_id=AUTH_A,
            epoch=1,
        )

    # Repeating the same observation is idempotent.
    again = replicas.demote_room(
        authority_store, placement=PLACEMENT, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
    )
    assert again["idempotent"] is True


def test_demote_rejects_non_superseding_epoch(authority_store, monkeypatch):
    _seed_room(authority_store)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    with pytest.raises(ReplicaEpochRegressionError):
        replicas.demote_room(
            authority_store, placement=PLACEMENT, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=1
        )


def test_full_failover_round_trip(authority_store, replica_store, monkeypatch):
    """Authority A hosts, replica B follows, A dies, B promotes, A returns
    and is fenced + demoted; the room's history survives intact throughout."""
    page = _seed_room(authority_store, n_events=4)
    replicas.ingest_page(
        replica_store, placement=PLACEMENT, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )

    # A "dies"; B takes over.
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    promoted = replicas.promote_replica(replica_store, placement=PLACEMENT, room_id="room-1")
    _append_event(
        replica_store,
        event_id="b-work",
        payload={"text": "work continues on B"},
        gateway_id=AUTH_B,
        epoch=promoted["authority_epoch"],
    )

    # A comes back, observes B's claim, and fences itself.
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    replicas.demote_room(
        authority_store,
        placement=PLACEMENT,
        room_id="room-1",
        observed_gateway_id=AUTH_B,
        observed_epoch=promoted["authority_epoch"],
    )
    with pytest.raises(HostedRoomError):
        _append_event(
            authority_store,
            event_id="a-stale",
            payload={"text": "split brain attempt"},
            gateway_id=AUTH_A,
            epoch=1,
        )

    # B's room holds the complete history: 4 original + claim + new work.
    replay = _read_events(replica_store)
    kinds = [e["kind"] for e in replay["events"]]
    assert kinds == ["message.user"] * 4 + ["authority.claimed", "message.user"]
    assert replay["authority"]["gateway_id"] == AUTH_B
