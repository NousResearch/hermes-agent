"""Contract cases for ``persistence.gateway.hosted_rooms.HostedRoomPeerRepository``: room links, grant revocations, owed and held grant revokes, peer reservations and remote-run receipts.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin the link cap, the revocation fence (``expires_at`` and ``revoked_before`` only grow), the reservation ranking (a newer epoch, or the same epoch under another gateway, outranks a claim), and the receipt replay rule.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import time

import pytest

from persistence.gateway.hosted_rooms import (
    MAX_OWED_REVOKE_SECONDS,
    MAX_OWED_REVOKES,
    MAX_OWED_REVOKES_PER_MEMBER,
    AuthorityConflictError,
    HostedRoomError,
    OwedRevokesFullError,
    RemoteRunIdentity,
    RemoteRunReceipt,
    RoomGrantClaims,
    RoomLinkRecord,
    RoomNotFoundError,
    grant_digest,
    room_grant_hard_expiry,
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


def test_a_receipt_drops_only_while_it_names_the_run(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    peers.upsert_remote_run_receipt(placement="profile", receipt=RECEIPT, now=100.0)
    assert peers.drop_remote_run_receipt(placement="profile", identity=_identity(RECEIPT), run_id="run-2") is False
    assert peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT)) is not None
    assert peers.drop_remote_run_receipt(placement="profile", identity=_identity(RECEIPT), run_id="run-1") is True
    assert peers.remote_run_receipt(placement="profile", identity=_identity(RECEIPT)) is None
    assert peers.drop_remote_run_receipt(placement="profile", identity=_identity(RECEIPT), run_id="run-1") is False


# ── owed and held grant revokes ───────────────────────────────────────────────


def _grant(expires_at: float, status_expires_at: float | None = None, nonce: str = "") -> str:
    """A grant token whose unsigned payload carries the expiries ``room_grant_hard_expiry`` reads."""
    claims = {"expires_at": expires_at, "nonce": nonce, **({} if status_expires_at is None else {"status_expires_at": status_expires_at})}
    payload, signature = (base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") for raw in (json.dumps(claims).encode(), b"signature"))
    return f"{payload}.{signature}"


def _owe(peers, member_id: str = "m1", *, grant: str, expires_at: float, now: float, room_id: str = "room") -> None:
    peers.owe_room_grant_revoke(placement="shared", room_id=room_id, member_id=member_id, target_url="https://peer.example/rooms",
                                grant=grant, expires_at=expires_at, now=now)


def test_the_hard_expiry_is_the_status_horizon_and_an_unreadable_grant_lives_longest():
    assert room_grant_hard_expiry(_grant(100.0)) == 100.0
    assert room_grant_hard_expiry(_grant(100.0, 500.0)) == 500.0
    for unreadable in ("not-a-grant", "a.!!!", _grant(100.0).split(".")[0] + ".é"):
        assert room_grant_hard_expiry(unreadable, now=7.0) == 7.0 + MAX_OWED_REVOKE_SECONDS
    assert grant_digest("g") == hashlib.sha256(b"g").hexdigest()


def test_an_owed_revoke_is_due_oldest_first_until_settled_or_expired(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    assert peers.owes_room_grant_revokes(placement="shared") is False
    _owe(peers, "m1", grant="g-late", expires_at=1000.0, now=20.0)
    _owe(peers, "m2", grant="g-early", expires_at=1000.0, now=10.0)
    _owe(peers, "m3", grant="g-expiring", expires_at=50.0, now=30.0)
    _owe(peers, "m1", grant="g-late", expires_at=9000.0, now=40.0)  # a recorded key stays as it is
    assert peers.owes_room_grant_revokes(placement="shared") is True
    due = peers.due_room_grant_revokes(placement="shared", limit=10, now=40.0)
    assert [row["grant"] for row in due] == ["g-early", "g-late", "g-expiring"]
    assert due[0] == {"room_id": "room", "member_id": "m2", "grant_digest": grant_digest("g-early"),
                      "target_url": "https://peer.example/rooms", "grant": "g-early", "attempts": 0}
    assert [row["grant"] for row in peers.due_room_grant_revokes(placement="shared", limit=1, now=40.0)] == ["g-early"]
    peers.settle_room_grant_revoke(placement="shared", room_id="room", member_id="m2", grant_digest=grant_digest("g-early"), retry_at=70.0)
    assert [row["grant"] for row in peers.due_room_grant_revokes(placement="shared", limit=10, now=60.0)] == ["g-late"]
    retried = peers.due_room_grant_revokes(placement="shared", limit=10, now=70.0)
    assert [(row["grant"], row["attempts"]) for row in retried] == [("g-late", 0), ("g-early", 1)]
    peers.settle_room_grant_revoke(placement="shared", room_id="room", member_id="m1", grant_digest=grant_digest("g-late"))
    peers.settle_room_grant_revoke(placement="shared", room_id="room", member_id="m2", grant_digest=grant_digest("g-early"))
    assert peers.owes_room_grant_revokes(placement="shared") is False
    assert peers.owes_room_grant_revokes(placement="profile") is False


def test_an_expired_owed_revoke_is_dropped_by_the_due_read(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    _owe(peers, grant="g", expires_at=50.0, now=10.0)
    assert peers.due_room_grant_revokes(placement="shared", limit=10, now=50.0) == []
    assert peers.owes_room_grant_revokes(placement="shared") is False


def test_an_owed_revoke_expiry_is_capped_at_the_longest_status_horizon(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    _owe(peers, grant="g", expires_at=10.0 * MAX_OWED_REVOKE_SECONDS, now=0.0)
    assert peers.room_has_peer_obligations(placement="shared", room_id="room", now=MAX_OWED_REVOKE_SECONDS - 1.0) is True
    assert peers.room_has_peer_obligations(placement="shared", room_id="room", now=float(MAX_OWED_REVOKE_SECONDS)) is False


def test_owed_revokes_are_bounded_per_member_and_in_total(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    for index in range(MAX_OWED_REVOKES_PER_MEMBER):
        _owe(peers, "m0", grant=f"g{index}", expires_at=1000.0, now=10.0)
    with pytest.raises(OwedRevokesFullError, match="too many owed revokes for this group chat member"):
        _owe(peers, "m0", grant="one-more", expires_at=1000.0, now=10.0)
    _owe(peers, "m0", grant="g0", expires_at=1000.0, now=10.0)  # a recorded key needs no room
    members = MAX_OWED_REVOKES // MAX_OWED_REVOKES_PER_MEMBER
    for member in range(1, members):
        for index in range(MAX_OWED_REVOKES_PER_MEMBER):
            _owe(peers, f"m{member}", grant=f"g{index}", expires_at=20.0 if member == 1 else 1000.0, now=10.0)
    with pytest.raises(OwedRevokesFullError, match="too many owed room grant revokes"):
        _owe(peers, "fresh", grant="g", expires_at=1000.0, now=15.0)
    for index in range(MAX_OWED_REVOKES_PER_MEMBER):  # the expired rows make room first
        _owe(peers, "fresh", grant=f"g{index}", expires_at=1000.0, now=20.0)
    with pytest.raises(OwedRevokesFullError, match="too many owed room grant revokes"):
        _owe(peers, "fresh-2", grant="g", expires_at=1000.0, now=20.0)


def test_a_replaced_grant_that_outlives_its_replacement_is_held(backend, scope_a):
    peers = backend.open(scope_a).hosted_room_peers
    soon = time.time() + 3600.0
    long_grant, short_grant, longer_grant = _grant(soon, soon + 600.0), _grant(soon, nonce="short"), _grant(soon, soon + 900.0)
    peers.upsert_room_link_record(placement="shared", record=_link(grant=long_grant), max_links=8)
    peers.upsert_room_link_record(placement="shared", record=_link(grant=short_grant), max_links=8)
    assert peers.held_room_grant_revokes(placement="shared", room_id="room") == []  # not asked to hold
    peers.upsert_room_link_record(placement="shared", record=_link(grant=long_grant), max_links=8, hold_replaced=True)
    assert peers.held_room_grant_revokes(placement="shared", room_id="room") == []  # the replacement outlives it
    peers.upsert_room_link_record(placement="shared", record=_link(grant=short_grant), max_links=8, hold_replaced=True)
    assert peers.held_room_grant_revokes(placement="shared", room_id="room") == [
        {"target_url": "https://peer.example/rooms", "grant": long_grant}]
    assert peers.due_room_grant_revokes(placement="shared", limit=10) == []  # held, not due
    peers.upsert_room_link_record(placement="shared", record=_link(grant=long_grant), max_links=8)
    assert peers.held_room_grant_revokes(placement="shared", room_id="room") == []  # the stored grant is never held
    peers.upsert_room_link_record(placement="shared", record=_link(grant=longer_grant), max_links=8, hold_replaced=True)
    for index in range(MAX_OWED_REVOKES_PER_MEMBER):
        _owe(peers, grant=f"owed-{index}", expires_at=soon, now=time.time())
    with pytest.raises(HostedRoomError, match="too many held group chat grant revokes"):
        peers.upsert_room_link_record(placement="shared", record=_link(grant=short_grant), max_links=8, hold_replaced=True)
    assert peers.list_room_link_records(placement="shared")[0]["grant"] == longer_grant


def test_held_grants_past_their_expiry_are_dropped_by_the_held_read(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw", now=5.0)
    peers = store.hosted_room_peers
    peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="g1", expires_at=50.0, now=10.0)
    peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m2", target_url="u", grant="g2", expires_at=90.0, now=10.0)
    assert peers.held_room_grant_revokes(placement="shared", room_id="room", now=50.0) == [{"target_url": "u", "grant": "g2"}]
    assert peers.room_has_peer_obligations(placement="shared", room_id="room", now=60.0) is True
    assert peers.held_room_grant_revokes(placement="shared", room_id="room", now=90.0) == []
    assert peers.owes_room_grant_revokes(placement="shared") is False


def test_a_grant_held_on_a_live_room_waits_and_on_a_disbanded_room_is_owed(backend, scope_a):
    store = backend.open(scope_a)
    rooms, peers = store.hosted_rooms, store.hosted_room_peers
    with pytest.raises(RoomNotFoundError):
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="g", expires_at=500.0, now=10.0)
    rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw", now=5.0)
    peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="held", expires_at=500.0, now=10.0)
    assert peers.due_room_grant_revokes(placement="shared", limit=10, now=20.0) == []
    for index in range(1, MAX_OWED_REVOKES_PER_MEMBER):
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant=f"held-{index}", expires_at=500.0, now=10.0)
    with pytest.raises(HostedRoomError, match="too many held group chat grant revokes") as refused:
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="full", expires_at=500.0, now=10.0)
    assert not isinstance(refused.value, OwedRevokesFullError)
    peers.retire_room_link_records(placement="shared", room_id="room", now=20.0)
    rooms.disband_room(placement="shared", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=30.0)
    peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="after", expires_at=500.0, now=40.0)
    assert [row["grant"] for row in peers.due_room_grant_revokes(placement="shared", limit=10, now=40.0)] == ["after"]
    for index in range(1, MAX_OWED_REVOKES_PER_MEMBER):
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant=f"after-{index}", expires_at=500.0, now=40.0)
    with pytest.raises(OwedRevokesFullError):
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m1", target_url="u", grant="full", expires_at=500.0, now=40.0)


def test_retiring_a_room_owes_its_routes_or_forgets_its_revoked_held_grants(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="shared", room_id="room", name="Room", members=[], authority_gateway_id="gw", now=5.0)
    peers = store.hosted_room_peers
    soon = time.time() + 3600.0
    peers.upsert_room_link_record(placement="shared", record=_link(grant=_grant(soon)), max_links=8)
    peers.upsert_room_link_record(placement="shared", record=_link(member_id="m2", grant=_grant(soon, nonce="2")), max_links=8)
    peers.upsert_room_link_record(placement="shared", record=_link("other", grant=_grant(soon, nonce="o")), max_links=8)
    for grant in ("held-a", "held-b"):
        peers.hold_room_grant_revoke(placement="shared", room_id="room", member_id="m3", target_url="u", grant=grant, expires_at=soon, now=10.0)
    assert peers.room_has_peer_obligations(placement="shared", room_id="room", now=10.0) is True
    assert peers.retire_room_link_records(placement="shared", room_id="room", revoked_digests={grant_digest("held-a")}, now=20.0) == 2
    assert peers.held_room_grant_revokes(placement="shared", room_id="room", now=20.0) == [{"target_url": "u", "grant": "held-b"}]
    assert peers.due_room_grant_revokes(placement="shared", limit=10, now=20.0) == []
    assert peers.retire_room_link_records(placement="shared", room_id="room", now=20.0) == 0
    assert peers.room_has_peer_obligations(placement="shared", room_id="room", now=20.0) is False
    store.hosted_rooms.create_room(placement="shared", room_id="other", name="Other", members=[], authority_gateway_id="gw", now=5.0)
    peers.hold_room_grant_revoke(placement="shared", room_id="other", member_id="m3", target_url="u", grant="held-o", expires_at=soon, now=10.0)
    assert peers.retire_room_link_records(placement="shared", room_id="other", owe_routes=True, now=30.0) == 1
    assert peers.list_room_link_records(placement="shared") == []
    due = peers.due_room_grant_revokes(placement="shared", limit=10, now=30.0)
    assert sorted(row["grant"] for row in due) == sorted([_grant(soon, nonce="o"), "held-o"])
    assert peers.held_room_grant_revokes(placement="shared", room_id="other", now=30.0) == []
