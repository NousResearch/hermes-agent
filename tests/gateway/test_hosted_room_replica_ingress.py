"""A participant stores only history sent with a live ``replicate`` grant it issued."""

import copy
import time

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_ingress import ingest_granted_page
from tests.gateway.fixtures.passive_copy import HOME, MEMBERS, SECRET, TARGET, append, grant, reserve


@pytest.fixture
def stores(tmp_path):
    source, target = tmp_path / "home.db", tmp_path / "participant.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    append(source, "hello")
    return source, target


def issued(target_db, **overrides):
    token = grant(**overrides)
    return token, reserve(target_db, token, now=overrides.get("now"))


def ingest(stores, token, **overrides):
    source, target = stores
    fields = dict(
        room_id="room", room_name="Workshop", members=MEMBERS,
        page=overrides.pop("page") if "page" in overrides else rooms.read_events(source, room_id="room"),
        token=token, secret=SECRET, target_install_id=TARGET, target_profile="default")
    fields.update(overrides)
    return ingest_granted_page(target, **fields)


def test_scoped_page_is_durable_idempotent_and_passive(stores):
    token, _ = issued(stores[1])
    assert ingest(stores, token)["stored_seq"] == 1
    assert ingest(stores, token)["ingested"] == 0
    assert replicas.replica_state(stores[1], room_id="room")["safety_status"] == "passive"
    with pytest.raises(rooms.RoomNotFoundError):
        rooms.room_state(stores[1], room_id="room")


@pytest.mark.parametrize("permissions", [("status",), ("dispatch",), ("approve", "stop")])
def test_grants_without_the_opt_in_never_gain_replication(stores, permissions):
    token, _ = issued(stores[1], permissions=permissions)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(stores, token)


@pytest.mark.parametrize("overrides", [
    {"room_id": "other"}, {"authority_gateway_id": "install:other"}, {"authority_epoch": 2},
    {"target": "install:other"}, {"profile": "other"}, {"member_id": "writer"},
])
def test_grant_scope_cannot_be_rebound(stores, overrides):
    token, _ = issued(stores[1], **overrides)
    with pytest.raises((peer.HostedRoomGrantError, replicas.ReplicaError)):
        ingest(stores, token)


def test_revoked_grant_cannot_write_even_with_an_authentic_page(stores):
    token, claims = issued(stores[1])
    rooms.revoke_room_grant_scope(stores[1], claims=claims, expires_at=claims["status_expires_at"])
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(stores, token)
    with pytest.raises(replicas.ReplicaNotFoundError, match="not found"):
        replicas.replica_state(stores[1], room_id="room")


def test_revocation_between_validation_and_write_wins(stores, monkeypatch):
    token, claims = issued(stores[1])
    original = replicas._validate_page

    def revoke_before_transaction(page):
        result = original(page)
        rooms.revoke_room_grant_scope(stores[1], claims=claims, expires_at=claims["status_expires_at"])
        return result

    monkeypatch.setattr(replicas, "_validate_page", revoke_before_transaction)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(stores, token)


def test_withdrawn_reservation_is_rechecked_inside_the_writer(stores, monkeypatch):
    token, claims = issued(stores[1])
    original = replicas._validate_page

    def drop_reservation(page):
        result = original(page)
        rooms.reserve_peer_room(stores[1], claims={**claims, "authority_epoch": 2},
                                expires_at=claims["status_expires_at"])
        return result

    monkeypatch.setattr(replicas, "_validate_page", drop_reservation)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(stores, token)


def test_authorized_sender_cannot_rewrite_existing_history(stores):
    token, _ = issued(stores[1])
    ingest(stores, token)
    page = copy.deepcopy(rooms.read_events(stores[0], room_id="room"))
    page["events"][0]["payload"]["text"] = "rewritten"
    with pytest.raises(replicas.ReplicaError):
        ingest(stores, token, page=page)


def test_expired_replication_horizon_cannot_write(stores, monkeypatch):
    now = time.time()
    token, _ = issued(stores[1], now=now)
    monkeypatch.setattr(time, "time", lambda: now + 3601)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(stores, token)


def test_replication_outlives_dispatch_until_the_status_horizon(stores, monkeypatch):
    now = time.time()
    token, _ = issued(stores[1], now=now)
    monkeypatch.setattr(time, "time", lambda: now + 601)
    with pytest.raises(peer.HostedRoomGrantError):
        peer.decode_room_grant(SECRET, token, permission="dispatch")
    assert ingest(stores, token)["stored_seq"] == 1


def test_renamed_room_keeps_copying_and_follows_its_rename_event(stores):
    token, _ = issued(stores[1])
    ingest(stores, token)
    rooms.rename_room(stores[0], room_id="room", event_id="rename", name="Revised workshop")
    # A sender may already know a later name; only the page's own events relabel the copy.
    assert ingest(stores, token, room_name="Ahead of its page", page=rooms.read_events(
        stores[0], room_id="room", since_seq=1))["stored_seq"] == 2
    assert replicas.replica_state(stores[1], room_id="room")["name"] == "Revised workshop"


def test_authenticated_sender_cannot_change_fixed_membership(stores):
    token, _ = issued(stores[1])
    ingest(stores, token)
    with pytest.raises(replicas.ReplicaError, match="metadata"):
        ingest(stores, token, members=[*MEMBERS, {"member_id": "late", "handle": "late", "profile": "default",
                                                  "target": {"kind": "local", "profile": "default"}}])


def test_replicated_disband_remains_terminal(stores):
    token, _ = issued(stores[1])
    ingest(stores, token)
    rooms.disband_room(stores[0], room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    page = rooms.read_events(stores[0], room_id="room", include_disbanded=True)
    ingest(stores, token, page=page)
    assert replicas.replica_state(stores[1], room_id="room")["disbanded_at"] is not None
    page = copy.deepcopy(page)
    later = copy.deepcopy(page["events"][0])
    later.update(seq=3, event_id="late")
    page["events"].append(later)
    page.update(cursor=3, latest_seq=3, has_more=False)
    with pytest.raises(replicas.ReplicaError):
        ingest(stores, token, page=page)
