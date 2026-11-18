"""Contract cases for ``persistence.gateway.hosted_rooms.HostedRoomReplicaRepository`` and the work-record capture's escaping checks.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin idempotent, gap- and regression-safe ingest, promotion at ``epoch + 1`` with a lineage-proving claim, demotion with ``authority.lost``. The capture cases pin what escapes the work-record capture inside a receipt write: an ``AuthorityHistoryError`` rolls the receipt back.
"""

from __future__ import annotations

import pytest

from persistence import Conflict
from persistence.gateway.hosted_rooms import (
    AuthorityHistoryError,
    RemoteRunIdentity,
    RemoteRunReceipt,
    ReplicaEpochRegressionError,
    ReplicaError,
    ReplicaGapError,
    ReplicaPage,
    RoomConflictError,
)

MEMBERS_JSON = '[{"member_id":"m1","profile":"p1"}]'


def _event(seq: int, *, epoch: int = 1, created_at: float | None = 50.0) -> dict:
    event = {"seq": seq, "event_id": f"e{seq}", "kind": "message.user", "actor": {"kind": "user", "id": "u1"}, "authority_epoch": epoch, "payload": {"text": f"t{seq}"}}
    if created_at is not None:
        event["created_at"] = created_at + seq
    return event


def _page(*seqs: int, gateway: str = "gw-a", epoch: int = 1, latest_seq: int | None = None) -> ReplicaPage:
    return ReplicaPage(authority_gateway_id=gateway, authority_epoch=epoch, events=tuple(_event(seq, epoch=epoch) for seq in seqs), latest_seq=latest_seq)


def _ingest(store, page: ReplicaPage, *, room_id: str = "room", now: float = 100.0):
    return store.hosted_room_replicas.ingest_page(placement="shared", room_id=room_id, room_name="Room", members_json=MEMBERS_JSON, page=page, now=now)


def _receipt(room_id: str = "room", epoch: int = 1) -> RemoteRunReceipt:
    return RemoteRunReceipt(
        room_id=room_id, home_install_id="home", authority_gateway_id="gw", authority_epoch=epoch, member_id="m1",
        target_install_id="target", target_profile="p1", task_id="task-1", execution_generation=1, run_id="run-1", session_id="sess-1")


def _identity(receipt: RemoteRunReceipt) -> RemoteRunIdentity:
    return RemoteRunIdentity(**{name: getattr(receipt, name) for name in RemoteRunIdentity.__dataclass_fields__})


# ── ingest_page and replica_state ─────────────────────────────────────────────


def test_ingest_stores_new_events_once_and_reports_coverage(backend, scope_a):
    store = backend.open(scope_a)
    assert _ingest(store, _page(1, 2, latest_seq=4)) == {"room_id": "room", "stored_seq": 2, "ingested": 2, "authority": {"gateway_id": "gw-a", "epoch": 1}, "caught_up": False}
    # CONTRACT: an overlapping page ingests only the events past the stored sequence.
    assert _ingest(store, _page(2, 3, 4, latest_seq=4), now=120.0)["ingested"] == 2
    assert _ingest(store, _page(1, 2, 3, 4), now=130.0) == {"room_id": "room", "stored_seq": 4, "ingested": 0, "authority": {"gateway_id": "gw-a", "epoch": 1}, "caught_up": True}
    state = store.hosted_room_replicas.replica_state(placement="shared", room_id="room")
    assert {key: state[key] for key in ("room_id", "name", "members", "authority", "last_seq", "latest_seq", "created_at", "updated_at")} == {
        "room_id": "room", "name": "Room", "members": [{"member_id": "m1", "profile": "p1"}],
        "authority": {"gateway_id": "gw-a", "epoch": 1}, "last_seq": 4, "latest_seq": 4, "created_at": 100.0, "updated_at": 130.0}
    assert state["event_bytes"] > 0
    with pytest.raises(ReplicaError, match="replica not found"):
        store.hosted_room_replicas.replica_state(placement="profile", room_id="room")


def test_ingest_refuses_gaps_and_epoch_regressions(backend, scope_a):
    store = backend.open(scope_a)
    _ingest(store, _page(1, 2, epoch=2))
    with pytest.raises(ReplicaGapError, match="page skips sequences the replica has not stored") as gap:
        _ingest(store, _page(4, 5, epoch=2))
    assert isinstance(gap.value, Conflict)
    with pytest.raises(ReplicaEpochRegressionError, match="page authority epoch is older than the stored replica epoch"):
        _ingest(store, _page(3, epoch=1))
    # CONTRACT: a newer epoch moves the stored authority forward.
    assert _ingest(store, _page(3, gateway="gw-b", epoch=3))["authority"] == {"gateway_id": "gw-b", "epoch": 3}
    assert store.hosted_room_replicas.replica_state(placement="shared", room_id="room")["last_seq"] == 3


def test_ingest_admits_at_most_256_replica_rooms(backend, scope_a):
    store = backend.open(scope_a)
    for index in range(256):
        _ingest(store, _page(), room_id=f"room-{index}")
    with pytest.raises(ReplicaError, match="replica room capacity exhausted"):
        _ingest(store, _page(1), room_id="one-more")
    assert _ingest(store, _page(1), room_id="room-0")["ingested"] == 1


# ── promote_replica ───────────────────────────────────────────────────────────


def test_promote_continues_the_replica_as_a_local_room_at_the_next_epoch(backend, scope_a, scope_b):
    store = backend.open(scope_a)
    _ingest(store, _page(1, 2))
    promoted = store.hosted_room_replicas.promote_replica(placement="shared", room_id="room", local_gateway_id="gw-b", reason="operator", now=200.0)
    assert promoted == {
        "room_id": "room", "authority_gateway_id": "gw-b", "authority_epoch": 2, "previous_gateway_id": "gw-a",
        "previous_epoch": 1, "claim_seq": 3, "latest_seq": 3}
    # CONTRACT: the promoted room lives in the same file, so every profile of the install sees it.
    rooms = backend.open(scope_b).hosted_rooms
    state = rooms.room_state(placement="shared", room_id="room")
    assert (state["authority_gateway_id"], state["authority_epoch"], state["latest_seq"], state["revision"], state["created_at"]) == ("gw-b", 2, 3, 1, 200.0)
    claim = state["authority_claim"]
    assert (claim["seq"], claim["event_id"], claim["kind"], claim["actor"]) == (3, "system:authority-claimed:2", "authority.claimed", {"kind": "system", "id": "authority-control"})
    assert claim["payload"] == {"previous_gateway_id": "gw-a", "authority_gateway_id": "gw-b", "authority_epoch": 2, "promoted_from_replica": True, "reason": "operator"}
    log = rooms.read_events(placement="shared", room_id="room")
    assert [(event["seq"], event["event_id"], event["created_at"]) for event in log["events"][:2]] == [(1, "e1", 51.0), (2, "e2", 52.0)]
    with pytest.raises(ReplicaError, match="replica not found"):
        store.hosted_room_replicas.replica_state(placement="shared", room_id="room")
    # CONTRACT: the promoted room continues at its new authority.
    appended = rooms.append_event(placement="shared", room_id="room", event_id="e4", kind="message.user", actor={"kind": "user", "id": "u1"}, payload={"text": "next"}, authority_gateway_id="gw-b", authority_epoch=2, now=210.0)
    assert appended["seq"] == 4


def test_promote_refuses_a_missing_replica_then_a_held_authority_then_a_local_or_disbanded_room(backend, scope_a):
    store = backend.open(scope_a)
    replicas = store.hosted_room_replicas
    with pytest.raises(ReplicaError, match="replica not found"):
        replicas.promote_replica(placement="shared", room_id="room", local_gateway_id="gw-b")
    _ingest(store, _page(1))
    with pytest.raises(ReplicaError, match="this gateway already holds the room authority"):
        replicas.promote_replica(placement="shared", room_id="room", local_gateway_id="gw-a")
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw-b", now=90.0)
    with pytest.raises(RoomConflictError, match="room_id already exists in the local authoritative store"):
        replicas.promote_replica(placement="shared", room_id="room", local_gateway_id="gw-b")
    store.hosted_rooms.disband_room(placement="shared", room_id="room", expected_gateway_id="gw-b", expected_epoch=1, now=95.0)
    store.hosted_rooms.prune_disbanded_rooms(placement="shared", now=95.0 + 90 * 24 * 60 * 60)
    with pytest.raises(RoomConflictError, match="room_id belongs to a disbanded room"):
        replicas.promote_replica(placement="shared", room_id="room", local_gateway_id="gw-b")
    assert replicas.replica_state(placement="shared", room_id="room")["last_seq"] == 1


# ── demote_room ───────────────────────────────────────────────────────────────


def test_demote_records_the_lost_authority_once(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw", now=90.0)
    demoted = store.hosted_room_replicas.demote_room(placement="shared", room_id="room", observed_gateway_id="gw-x", observed_epoch=3, local_gateway_id="gw", now=100.0)
    assert demoted == {"room_id": "room", "authority_gateway_id": "gw-x", "authority_epoch": 3, "idempotent": False}
    state = store.hosted_rooms.room_state(placement="shared", room_id="room")
    assert (state["authority_gateway_id"], state["authority_epoch"], state["latest_seq"], state["revision"], state["updated_at"]) == ("gw-x", 3, 1, 2, 100.0)
    (lost,) = store.hosted_rooms.read_events(placement="shared", room_id="room")["events"]
    assert (lost["event_id"], lost["kind"], lost["authority_epoch"]) == ("system:authority-lost:3", "authority.lost", 3)
    assert lost["payload"] == {"previous_gateway_id": "gw", "authority_gateway_id": "gw-x", "authority_epoch": 3}
    assert store.hosted_room_replicas.demote_room(placement="shared", room_id="room", observed_gateway_id="gw-x", observed_epoch=3, local_gateway_id="gw", now=110.0) == {**demoted, "idempotent": True}
    with pytest.raises(ReplicaEpochRegressionError, match="observed epoch does not supersede the stored authority"):
        store.hosted_room_replicas.demote_room(placement="shared", room_id="room", observed_gateway_id="gw-y", observed_epoch=3, local_gateway_id="gw")
    with pytest.raises(ReplicaError, match="room is not locally authoritative; nothing to demote"):
        store.hosted_room_replicas.demote_room(placement="shared", room_id="room", observed_gateway_id="gw-y", observed_epoch=4, local_gateway_id="gw")
    with pytest.raises(ReplicaError, match="room not found in the local authoritative store"):
        store.hosted_room_replicas.demote_room(placement="profile", room_id="room", observed_gateway_id="gw-y", observed_epoch=4, local_gateway_id="gw")


# ── the work-record capture inside a receipt write ────────────────────────────


def test_a_receipt_for_a_room_without_a_committed_lineage_rolls_back(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw", now=90.0)
    store.hosted_room_replicas.demote_room(placement="shared", room_id="room", observed_gateway_id="gw-x", observed_epoch=3, local_gateway_id="gw", now=100.0)
    # CONTRACT: the room's own fence is its first transfer, and a fence to epoch 3 does not extend the first span.
    with pytest.raises(AuthorityHistoryError, match="authority claim does not extend its predecessor") as escaped:
        store.hosted_room_peers.upsert_remote_run_receipt(placement="shared", receipt=_receipt(), now=110.0)
    assert isinstance(escaped.value, ValueError)
    assert store.hosted_room_peers.remote_run_receipt(placement="shared", identity=_identity(_receipt())) is None


def test_a_receipt_checks_the_claim_lineage_of_a_promoted_room(backend, scope_a):
    store = backend.open(scope_a)
    _ingest(store, _page(1, epoch=1), room_id="valid")
    _ingest(store, _page(1, epoch=3), room_id="skipped")
    for room_id in ("valid", "skipped"):
        store.hosted_room_replicas.promote_replica(placement="shared", room_id=room_id, local_gateway_id="gw-b", now=200.0)
    # CONTRACT: a claim at epoch 2 extends the first owner's span, so the receipt commits.
    store.hosted_room_peers.upsert_remote_run_receipt(placement="shared", receipt=_receipt("valid"), now=210.0)
    assert store.hosted_room_peers.remote_run_receipt(placement="shared", identity=_identity(_receipt("valid"))) is not None
    # CONTRACT: a claim at epoch 4 does not extend a first span, so the history error escapes and the receipt rolls back.
    with pytest.raises(AuthorityHistoryError, match="authority claim does not extend its predecessor"):
        store.hosted_room_peers.upsert_remote_run_receipt(placement="shared", receipt=_receipt("skipped"), now=210.0)
    assert store.hosted_room_peers.remote_run_receipt(placement="shared", identity=_identity(_receipt("skipped"))) is None
