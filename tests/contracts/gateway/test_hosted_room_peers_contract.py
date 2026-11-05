"""Contract cases for ``persistence.gateway.hosted_rooms.HostedRoomPeerRepository``: room links, grant revocations, peer reservations and remote-run receipts.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin the link cap, the revocation fence (``expires_at`` and ``revoked_before`` only grow), the reservation ranking (a newer epoch, or the same epoch under another gateway, outranks a claim), and the receipt replay rule.
"""

from __future__ import annotations

import dataclasses

import pytest

from persistence.gateway.hosted_rooms import (
    AuthorityConflictError,
    HostedRoomError,
    RemoteRunIdentity,
    RemoteRunReceipt,
    RoomGrantClaims,
    RoomLinkRecord,
)

CLAIMS = RoomGrantClaims(
    room_id="room", home_install_id="home", authority_gateway_id="gw", authority_epoch=1, member_id="m1",
    target_install_id="target", target_profile="p1", issued_at=90.0)
RECEIPT = RemoteRunReceipt(
    room_id="room", home_install_id="home", authority_gateway_id="gw", authority_epoch=1, member_id="m1",
    target_install_id="target", target_profile="p1", task_id="task-1", execution_generation=1, run_id="run-1", session_id="sess-1")


def _link(room_id: str = "room", member_id: str = "m1", **changes) -> RoomLinkRecord:
    values = {
        "room_id": room_id, "member_id": member_id, "target_url": "https://peer.example/rooms", "target_profile": "p1",
        "grant": "secret-grant", "catalog_json": "[]", "cancellation_scope_id": "scope-1", "trace_id": "trace-1",
        "transport_security": "tls", "status": "ready", "updated_at": 100.0}
    return RoomLinkRecord(**{**values, **changes})


def _identity(receipt: RemoteRunReceipt) -> RemoteRunIdentity:
    return RemoteRunIdentity(**{field.name: getattr(receipt, field.name) for field in dataclasses.fields(RemoteRunIdentity)})


# ── links ─────────────────────────────────────────────────────────────────────


def test_room_links_upsert_by_room_and_member_and_list_in_key_order(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.upsert_room_link_record(placement="profile", record=_link("room-b"), max_links=512)
    peers.upsert_room_link_record(placement="profile", record=_link("room-a", "m2"), max_links=512)
    peers.upsert_room_link_record(placement="profile", record=_link("room-a", "m1"), max_links=512)
    peers.upsert_room_link_record(placement="profile", record=_link("room-b", status="degraded", updated_at=120.0), max_links=512)
    rows = peers.list_room_link_records(placement="profile")
    assert [(row["room_id"], row["member_id"]) for row in rows] == [("room-a", "m1"), ("room-a", "m2"), ("room-b", "m1")]
    assert rows[2] == {**dataclasses.asdict(_link("room-b")), "status": "degraded", "updated_at": 120.0}
    assert "secret-grant" not in repr(_link())
    assert peers.list_room_link_records(placement="shared") == []


def test_the_link_cap_refuses_only_a_new_key(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.upsert_room_link_record(placement="shared", record=_link("room", "m1"), max_links=2)
    peers.upsert_room_link_record(placement="shared", record=_link("room", "m2"), max_links=2)
    with pytest.raises(HostedRoomError, match="too many stored room links"):
        peers.upsert_room_link_record(placement="shared", record=_link("room", "m3"), max_links=2)
    # CONTRACT: replacing a stored link at the cap is allowed.
    peers.upsert_room_link_record(placement="shared", record=_link("room", "m2", trace_id="trace-2"), max_links=2)
    assert [row["trace_id"] for row in peers.list_room_link_records(placement="shared")] == ["trace-1", "trace-2"]


def test_link_status_updates_report_whether_the_link_exists(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.upsert_room_link_record(placement="profile", record=_link(), max_links=512)
    assert peers.update_room_link_status(placement="profile", room_id="room", member_id="m1", status="unreachable", now=150.0) is True
    assert peers.update_room_link_status(placement="profile", room_id="room", member_id="m9", status="unreachable", now=150.0) is False
    assert peers.update_room_link_status(placement="shared", room_id="room", member_id="m1", status="unreachable", now=150.0) is False
    (row,) = peers.list_room_link_records(placement="profile")
    assert (row["status"], row["updated_at"]) == ("unreachable", 150.0)


def test_delete_room_links_removes_one_room_and_counts_it(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    for room_id, member_id in (("room", "m1"), ("room", "m2"), ("other", "m1")):
        peers.upsert_room_link_record(placement="profile", record=_link(room_id, member_id), max_links=512)
    assert peers.delete_room_link_records(placement="profile", room_id="room") == 2
    assert peers.delete_room_link_records(placement="profile", room_id="room") == 0
    assert [row["room_id"] for row in peers.list_room_link_records(placement="profile")] == ["other"]


# ── grant revocations ─────────────────────────────────────────────────────────


def test_a_revoked_scope_refuses_grants_issued_before_the_revocation_until_it_expires(backend, scope_a, scope_b):
    peers = backend.open(scope_a).hosted_room_peers
    peers.revoke_room_grant_scope(placement="shared", claims=CLAIMS, expires_at=200.0, now=100.0)
    assert peers.room_grant_is_revoked(placement="shared", claims=CLAIMS, now=150.0) is True
    assert peers.room_grant_is_revoked(placement="shared", claims=dataclasses.replace(CLAIMS, issued_at=100.0), now=150.0) is True
    assert peers.room_grant_is_revoked(placement="shared", claims=dataclasses.replace(CLAIMS, issued_at=101.0), now=150.0) is False
    assert peers.room_grant_is_revoked(placement="shared", claims=CLAIMS, now=200.0) is False
    assert peers.room_grant_is_revoked(placement="shared", claims=dataclasses.replace(CLAIMS, target_profile="p2"), now=150.0) is False
    assert peers.room_grant_is_revoked(placement="profile", claims=CLAIMS, now=150.0) is False
    # CONTRACT: the shared revocation is read by every profile of the install.
    assert backend.open(scope_b).hosted_room_peers.room_grant_is_revoked(placement="shared", claims=CLAIMS, now=150.0) is True


def test_revocations_only_grow_and_an_expired_one_writes_nothing(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.revoke_room_grant_scope(placement="shared", claims=CLAIMS, expires_at=500.0, now=100.0)
    peers.revoke_room_grant_scope(placement="shared", claims=CLAIMS, expires_at=300.0, now=50.0)
    late = dataclasses.replace(CLAIMS, issued_at=100.0)
    # CONTRACT: revoked_before keeps the later instant and expires_at the later expiry.
    assert peers.room_grant_is_revoked(placement="shared", claims=late, now=400.0) is True
    peers.revoke_room_grant_scope(placement="shared", claims=CLAIMS, expires_at=600.0, now=600.0)
    assert peers.room_grant_is_revoked(placement="shared", claims=dataclasses.replace(CLAIMS, issued_at=550.0), now=450.0) is False


def test_an_incomplete_grant_scope_is_refused(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    incomplete = dataclasses.replace(CLAIMS, target_install_id="")
    with pytest.raises(HostedRoomError, match="room grant scope is incomplete"):
        peers.revoke_room_grant_scope(placement="shared", claims=incomplete, expires_at=200.0, now=100.0)
    with pytest.raises(HostedRoomError, match="room grant scope is incomplete"):
        peers.room_grant_is_revoked(placement="shared", claims=incomplete, now=100.0)


# ── peer reservations ─────────────────────────────────────────────────────────


def test_a_reservation_makes_its_grant_current_until_it_expires(backend, scope_a, scope_b):
    peers = backend.open(scope_a).hosted_room_peers
    # CONTRACT: a file never opened answers the probe False without being created.
    assert peers.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=100.0) is False
    with pytest.raises(HostedRoomError, match="peer room reservation must expire in the future"):
        peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=100.0, now=100.0)
    peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=200.0, now=100.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=150.0) is True
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=200.0) is False
    assert peers.peer_room_grant_is_current(placement="shared", claims=dataclasses.replace(CLAIMS, authority_epoch=2), now=150.0) is False
    assert peers.peer_room_grant_is_current(placement="profile", claims=CLAIMS, now=150.0) is False
    other = backend.open(scope_b).hosted_room_peers
    assert other.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=150.0) is True
    assert other.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p2", now=150.0) is False
    assert other.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=200.0) is False
    # CONTRACT: a renewal keeps the later expiry.
    peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=150.0, now=120.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=180.0) is True


def test_a_newer_authority_outranks_and_revokes_older_reservations(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=500.0, now=100.0)
    rival = dataclasses.replace(CLAIMS, member_id="m2", authority_gateway_id="gw-2")
    # CONTRACT: the same epoch under another gateway is outranked by the live reservation.
    with pytest.raises(AuthorityConflictError, match="peer room reservation authority changed") as refused:
        peers.reserve_peer_room(placement="shared", claims=rival, expires_at=500.0, now=110.0)
    assert refused.value.reason == "authority_conflict"
    newer = dataclasses.replace(rival, authority_epoch=2)
    peers.reserve_peer_room(placement="shared", claims=newer, expires_at=500.0, now=120.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=newer, now=130.0) is True
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=130.0) is False
    # CONTRACT: the member's own row outranks a stale claim even after it was revoked.
    peers.revoke_room_grant_scope(placement="shared", claims=newer, expires_at=500.0, now=135.0)
    with pytest.raises(AuthorityConflictError, match="peer room reservation authority changed"):
        peers.reserve_peer_room(placement="shared", claims=dataclasses.replace(newer, authority_epoch=1, authority_gateway_id="gw"), expires_at=500.0, now=140.0)
    other_profile = dataclasses.replace(CLAIMS, target_profile="p2")
    peers.reserve_peer_room(placement="shared", claims=other_profile, expires_at=500.0, now=150.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=other_profile, now=160.0) is True


def test_revoking_a_scope_revokes_its_reservation_and_a_new_reservation_restores_it(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=500.0, now=100.0)
    peers.revoke_room_grant_scope(placement="shared", claims=CLAIMS, expires_at=500.0, now=110.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=120.0) is False
    assert peers.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=120.0) is False
    peers.reserve_peer_room(placement="shared", claims=CLAIMS, expires_at=500.0, now=130.0)
    assert peers.peer_room_grant_is_current(placement="shared", claims=CLAIMS, now=140.0) is True


# ── remote-run receipts ───────────────────────────────────────────────────────


def test_a_receipt_binds_one_attempt_and_replays_only_identically(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    assert peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT)) is None
    peers.upsert_remote_run_receipt(placement="profile", receipt=RECEIPT, now=100.0)
    stored = peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT))
    assert stored == {**dataclasses.asdict(RECEIPT), "created_at": 100.0, "updated_at": 100.0}
    peers.upsert_remote_run_receipt(placement="profile", receipt=RECEIPT, now=150.0)
    assert peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT)) == {**stored, "updated_at": 150.0}
    for changes in ({"run_id": "run-2"}, {"session_id": "sess-2"}):
        with pytest.raises(HostedRoomError, match="remote run receipt conflicts with its logical task"):
            peers.upsert_remote_run_receipt(placement="profile", receipt=dataclasses.replace(RECEIPT, **changes), now=160.0)
    assert peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT))["updated_at"] == 150.0
    retry = dataclasses.replace(RECEIPT, execution_generation=2, run_id="run-2")
    peers.upsert_remote_run_receipt(placement="profile", receipt=retry, now=170.0)
    assert peers.remote_run_receipt(placement="profile", identity=_identity(retry))["run_id"] == "run-2"
    assert peers.remote_run_receipt(placement="shared", identity=_identity(RECEIPT)) is None


def test_a_receipt_for_a_live_first_epoch_room_commits(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[{"member_id": "m1", "profile": "p1"}], authority_gateway_id="gw", now=90.0)
    store.hosted_room_peers.upsert_remote_run_receipt(placement="shared", receipt=RECEIPT, now=100.0)
    assert store.hosted_room_peers.remote_run_receipt(placement="shared", identity=_identity(RECEIPT))["run_id"] == "run-1"
