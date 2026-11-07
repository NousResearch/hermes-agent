"""The SQLite ``HostedRoomPeerRepository`` against main's ``gateway.hosted_rooms`` link, grant, reservation and
receipt functions, read from the pinned main checkout: the placement's file is seeded through the storage verbs and
copied, main's function runs on the copy, the backend verb on the placement, then the outcomes and the hosted-room rows
of both files are compared."""

from __future__ import annotations

import dataclasses

import pytest

from persistence.gateway.hosted_rooms import RemoteRunIdentity, RemoteRunReceipt, RoomGrantClaims, RoomLinkRecord
from persistence.sqlite.gateway.hosted_room_peers import SQLiteHostedRoomPeerRepository
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted.conftest import NOW, outcome, same_rows, seeded_pair

__all__ = ["legacy_store"]

pytestmark = pytest.mark.usefixtures("clock")

CLAIMS = RoomGrantClaims(
    room_id="room", home_install_id="home", authority_gateway_id="gw", authority_epoch=1, member_id="m1",
    target_install_id="target", target_profile="p1", issued_at=NOW - 100)
RECEIPT = RemoteRunReceipt(
    room_id="room", home_install_id="home", authority_gateway_id="gw", authority_epoch=1, member_id="m1",
    target_install_id="target", target_profile="p1", task_id="task-1", execution_generation=1, run_id="run-1",
    session_id="sess-1")


def _link(member_id: str, **changes) -> RoomLinkRecord:
    values = {
        "room_id": "room", "member_id": member_id, "target_url": "https://peer.example/rooms", "target_profile": "p1",
        "grant": "secret-grant", "catalog_json": "[]", "cancellation_scope_id": "scope-1", "trace_id": "trace-1",
        "transport_security": "tls", "status": "ready", "updated_at": NOW - 50}
    return RoomLinkRecord(**{**values, **changes})


def _identity(receipt: RemoteRunReceipt) -> RemoteRunIdentity:
    return RemoteRunIdentity(**{field.name: getattr(receipt, field.name) for field in dataclasses.fields(RemoteRunIdentity)})


def _asdict(value) -> dict:
    return dataclasses.asdict(value)


@pytest.fixture
def m1(legacy_store):
    """Main's ``gateway.hosted_rooms`` from the pinned checkout, importable for the whole test."""
    return legacy_import("gateway.hosted_rooms")


def _seed(store, placement: str) -> None:
    store.hosted_rooms.create_room(placement=placement, room_id="room", name="Room",
                                   members=[{"member_id": "m1", "profile": "p1"}], authority_gateway_id="gw",
                                   now=NOW - 90)
    peers = SQLiteHostedRoomPeerRepository(store)
    for member in ("m1", "m2"):
        peers.upsert_room_link_record(placement=placement, record=_link(member), max_links=512)
    peers.revoke_room_grant_scope(placement=placement, claims=dataclasses.replace(CLAIMS, member_id="m2"),
                                  expires_at=NOW + 500, now=NOW - 60)
    peers.revoke_room_grant_scope(placement=placement, claims=dataclasses.replace(CLAIMS, member_id="m3"),
                                  expires_at=NOW - 1, now=NOW - 70)
    peers.reserve_peer_room(placement=placement, claims=CLAIMS, expires_at=NOW + 500, now=NOW - 40)
    peers.reserve_peer_room(placement=placement, claims=dataclasses.replace(CLAIMS, member_id="m9", target_profile="p9"),
                            expires_at=NOW - 2, now=NOW - 30)
    peers.upsert_remote_run_receipt(placement=placement, receipt=RECEIPT, now=NOW - 20)


NEWER = dataclasses.replace(CLAIMS, authority_epoch=2, authority_gateway_id="gw-2")
RIVAL = dataclasses.replace(CLAIMS, member_id="m4", authority_gateway_id="gw-3")
CASES = {
    "upsert_new": (
        lambda m1, p: m1.upsert_room_link_record(p, record=_asdict(_link("m3")), max_links=512),
        lambda r, pl: r.upsert_room_link_record(placement=pl, record=_link("m3"), max_links=512)),
    "upsert_update_at_cap": (
        lambda m1, p: m1.upsert_room_link_record(p, record=_asdict(_link("m2", trace_id="t2")), max_links=2),
        lambda r, pl: r.upsert_room_link_record(placement=pl, record=_link("m2", trace_id="t2"), max_links=2)),
    "upsert_new_over_cap": (
        lambda m1, p: m1.upsert_room_link_record(p, record=_asdict(_link("m3")), max_links=2),
        lambda r, pl: r.upsert_room_link_record(placement=pl, record=_link("m3"), max_links=2)),
    "status_hit": (
        lambda m1, p: m1.update_room_link_status(p, room_id="room", member_id="m1", status="unreachable", now=NOW),
        lambda r, pl: r.update_room_link_status(placement=pl, room_id="room", member_id="m1", status="unreachable", now=NOW)),
    "status_miss": (
        lambda m1, p: m1.update_room_link_status(p, room_id="room", member_id="m7", status="unreachable", now=NOW),
        lambda r, pl: r.update_room_link_status(placement=pl, room_id="room", member_id="m7", status="unreachable", now=NOW)),
    "delete_links": (
        lambda m1, p: m1.delete_room_link_records(p, room_id="room"),
        lambda r, pl: r.delete_room_link_records(placement=pl, room_id="room")),
    "revoke_scope": (
        lambda m1, p: m1.revoke_room_grant_scope(p, claims=_asdict(CLAIMS), expires_at=NOW + 100, now=NOW),
        lambda r, pl: r.revoke_room_grant_scope(placement=pl, claims=CLAIMS, expires_at=NOW + 100, now=NOW)),
    "revoke_grows_only": (
        lambda m1, p: m1.revoke_room_grant_scope(p, claims=_asdict(dataclasses.replace(CLAIMS, member_id="m2")), expires_at=NOW + 10, now=NOW - 80),
        lambda r, pl: r.revoke_room_grant_scope(placement=pl, claims=dataclasses.replace(CLAIMS, member_id="m2"), expires_at=NOW + 10, now=NOW - 80)),
    "revoke_already_expired": (
        lambda m1, p: m1.revoke_room_grant_scope(p, claims=_asdict(CLAIMS), expires_at=NOW, now=NOW),
        lambda r, pl: r.revoke_room_grant_scope(placement=pl, claims=CLAIMS, expires_at=NOW, now=NOW)),
    "revoke_incomplete": (
        lambda m1, p: m1.revoke_room_grant_scope(p, claims=_asdict(dataclasses.replace(CLAIMS, home_install_id="")), expires_at=NOW + 1, now=NOW),
        lambda r, pl: r.revoke_room_grant_scope(placement=pl, claims=dataclasses.replace(CLAIMS, home_install_id=""), expires_at=NOW + 1, now=NOW)),
    "reserve_renew": (
        lambda m1, p: m1.reserve_peer_room(p, claims=_asdict(CLAIMS), expires_at=NOW + 900, now=NOW),
        lambda r, pl: r.reserve_peer_room(placement=pl, claims=CLAIMS, expires_at=NOW + 900, now=NOW)),
    "reserve_newer_epoch": (
        lambda m1, p: m1.reserve_peer_room(p, claims=_asdict(NEWER), expires_at=NOW + 900, now=NOW),
        lambda r, pl: r.reserve_peer_room(placement=pl, claims=NEWER, expires_at=NOW + 900, now=NOW)),
    "reserve_outranked": (
        lambda m1, p: m1.reserve_peer_room(p, claims=_asdict(RIVAL), expires_at=NOW + 900, now=NOW),
        lambda r, pl: r.reserve_peer_room(placement=pl, claims=RIVAL, expires_at=NOW + 900, now=NOW)),
    "reserve_in_the_past": (
        lambda m1, p: m1.reserve_peer_room(p, claims=_asdict(CLAIMS), expires_at=NOW, now=NOW),
        lambda r, pl: r.reserve_peer_room(placement=pl, claims=CLAIMS, expires_at=NOW, now=NOW)),
    "receipt_new": (
        lambda m1, p: m1.upsert_remote_run_receipt(p, record=_asdict(dataclasses.replace(RECEIPT, execution_generation=2, run_id="run-2")), now=NOW),
        lambda r, pl: r.upsert_remote_run_receipt(placement=pl, receipt=dataclasses.replace(RECEIPT, execution_generation=2, run_id="run-2"), now=NOW)),
    "receipt_replay": (
        lambda m1, p: m1.upsert_remote_run_receipt(p, record=_asdict(RECEIPT), now=NOW),
        lambda r, pl: r.upsert_remote_run_receipt(placement=pl, receipt=RECEIPT, now=NOW)),
    "receipt_conflict": (
        lambda m1, p: m1.upsert_remote_run_receipt(p, record=_asdict(dataclasses.replace(RECEIPT, session_id="sess-2")), now=NOW),
        lambda r, pl: r.upsert_remote_run_receipt(placement=pl, receipt=dataclasses.replace(RECEIPT, session_id="sess-2"), now=NOW)),
    "list_links": (
        lambda m1, p: m1.list_room_link_records(p),
        lambda r, pl: r.list_room_link_records(placement=pl)),
    "revoked_before": (
        lambda m1, p: m1.room_grant_is_revoked(p, claims=_asdict(dataclasses.replace(CLAIMS, member_id="m2")), now=NOW),
        lambda r, pl: r.room_grant_is_revoked(placement=pl, claims=dataclasses.replace(CLAIMS, member_id="m2"), now=NOW)),
    "revoked_issued_after": (
        lambda m1, p: m1.room_grant_is_revoked(p, claims=_asdict(dataclasses.replace(CLAIMS, member_id="m2", issued_at=NOW - 10)), now=NOW),
        lambda r, pl: r.room_grant_is_revoked(placement=pl, claims=dataclasses.replace(CLAIMS, member_id="m2", issued_at=NOW - 10), now=NOW)),
    "revoked_expired": (
        lambda m1, p: m1.room_grant_is_revoked(p, claims=_asdict(dataclasses.replace(CLAIMS, member_id="m3")), now=NOW),
        lambda r, pl: r.room_grant_is_revoked(placement=pl, claims=dataclasses.replace(CLAIMS, member_id="m3"), now=NOW)),
    "grant_current": (
        lambda m1, p: m1.peer_room_grant_is_current(p, claims=_asdict(CLAIMS), now=NOW),
        lambda r, pl: r.peer_room_grant_is_current(placement=pl, claims=CLAIMS, now=NOW)),
    "grant_other_epoch": (
        lambda m1, p: m1.peer_room_grant_is_current(p, claims=_asdict(NEWER), now=NOW),
        lambda r, pl: r.peer_room_grant_is_current(placement=pl, claims=NEWER, now=NOW)),
    "probe_live": (
        lambda m1, p: m1.probe_peer_room_reservation(p, room_id="room", target_profile="p1", now=NOW),
        lambda r, pl: r.probe_peer_room_reservation(placement=pl, room_id="room", target_profile="p1", now=NOW)),
    "probe_expired": (
        lambda m1, p: m1.probe_peer_room_reservation(p, room_id="room", target_profile="p9", now=NOW),
        lambda r, pl: r.probe_peer_room_reservation(placement=pl, room_id="room", target_profile="p9", now=NOW)),
    "receipt_read": (
        lambda m1, p: m1.remote_run_receipt(p, record=_asdict(_identity(RECEIPT))),
        lambda r, pl: r.remote_run_receipt(placement=pl, identity=_identity(RECEIPT))),
    "receipt_read_miss": (
        lambda m1, p: m1.remote_run_receipt(p, record=_asdict(_identity(dataclasses.replace(RECEIPT, task_id="other")))),
        lambda r, pl: r.remote_run_receipt(placement=pl, identity=_identity(dataclasses.replace(RECEIPT, task_id="other")))),
}


@pytest.mark.parametrize("placement", ["shared", "profile"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_peer_verb_matches_m1(case, placement, backend, scopes, tmp_path, m1):
    legacy, target = seeded_pair(
        backend, scopes["a"], placement, tmp_path, lambda path: _seed(backend.open(scopes["a"]), placement))
    legacy_call, verb = CASES[case]
    expected = outcome(lambda: legacy_call(m1, legacy))
    actual = outcome(lambda: verb(SQLiteHostedRoomPeerRepository(backend.open(scopes["a"])), placement))
    assert actual == expected
    backend_rows, legacy_rows = same_rows(target, legacy)
    assert backend_rows == legacy_rows


def test_probe_never_creates_the_file(backend, scopes, root):
    peers = SQLiteHostedRoomPeerRepository(backend.open(scopes["a"]))
    assert peers.probe_peer_room_reservation(placement="shared", room_id="room", target_profile="p1", now=NOW) is False
    assert not (root / "shared-state.db").exists()
