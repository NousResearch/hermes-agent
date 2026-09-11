"""Authenticated replication uses real signatures, reservations and SQLite writes."""

import copy
import time

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_ingress import ingest_granted_page


SECRET = b"replica-test-secret-not-a-real-credential"
HOME = "install:home"
TARGET = "install:target"
MEMBERS = [
    {"member_id": "writer", "profile": "default", "handle": "writer", "target": {"kind": "local", "profile": "default"}},
    {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer", "target": {
        "kind": "peer", "peer_id": "peer-reviewer", "installation_id": TARGET,
        "profile": "reviewer", "capability_digest": "b" * 64,
    }},
]


@pytest.fixture
def pair(tmp_path):
    source, target = tmp_path / "home.db", tmp_path / "peer.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(
        source, room_id="room", event_id="hello", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "Hello"},
        authority_gateway_id=HOME, authority_epoch=1,
    )
    return source, target


def grant(target, *, permissions=("replicate",), now=None, **overrides):
    now = time.time() if now is None else now
    fields = dict(
        grant_id="replication-test", room_id="room", home_install_id=HOME,
        authority_gateway_id=HOME, authority_epoch=1, member_id="reviewer",
        target_install_id=TARGET, target_profile="reviewer", execution_policy_digest="a" * 64,
        issued_at=now - 1, ttl_seconds=60, status_ttl_seconds=600, permissions=permissions,
    )
    fields.update(overrides)
    token = peer.issue_room_grant(SECRET, **fields)
    claims = peer.decode_room_grant(SECRET, token, permission=permissions[0], now=now)
    rooms.reserve_peer_room(target, claims=claims, expires_at=claims["status_expires_at"], now=now)
    return token, claims


def ingest(pair, token, **overrides):
    source, target = pair
    fields = dict(
        room_id="room", room_name="Workshop", members=MEMBERS,
        page=overrides.pop("page") if "page" in overrides else rooms.read_events(source, room_id="room"),
        token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer",
    )
    fields.update(overrides)
    return ingest_granted_page(target, **fields)


def test_scoped_page_is_durable_idempotent_and_passive(pair):
    token, _ = grant(pair[1])
    assert ingest(pair, token)["stored_seq"] == 1
    assert ingest(pair, token)["ingested"] == 0
    assert replicas.replica_state(pair[1], room_id="room")["safety_status"] == "passive"
    with pytest.raises(rooms.RoomNotFoundError):
        rooms.room_state(pair[1], room_id="room")


@pytest.mark.parametrize("permissions", [("status",), ("dispatch",), ("approve", "stop")])
def test_old_grants_never_gain_replication_rights(pair, permissions):
    token, _ = grant(pair[1], permissions=permissions)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(pair, token)


@pytest.mark.parametrize("overrides", [
    {"room_id": "other"}, {"authority_gateway_id": "install:other"},
    {"authority_epoch": 2}, {"target_install_id": "install:other"},
    {"target_profile": "other"}, {"member_id": "writer"},
])
def test_grant_scope_cannot_be_rebound(pair, overrides):
    token, _ = grant(pair[1], **overrides)
    with pytest.raises((peer.HostedRoomGrantError, replicas.ReplicaError)):
        ingest(pair, token)


def test_revoked_grant_cannot_write_even_with_authentic_page(pair):
    token, claims = grant(pair[1])
    rooms.revoke_room_grant_scope(pair[1], claims=claims, expires_at=claims["status_expires_at"])
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(pair, token)
    with pytest.raises(replicas.ReplicaError, match="not found"):
        replicas.replica_state(pair[1], room_id="room")


def test_revoke_between_validation_and_write_wins(pair, monkeypatch):
    token, claims = grant(pair[1])
    original = replicas._validate_page

    def revoke_before_transaction(page):
        result = original(page)
        rooms.revoke_room_grant_scope(pair[1], claims=claims, expires_at=claims["status_expires_at"])
        return result

    monkeypatch.setattr(replicas, "_validate_page", revoke_before_transaction)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(pair, token)


def test_authorized_sender_cannot_rewrite_existing_history(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    page = copy.deepcopy(rooms.read_events(pair[0], room_id="room"))
    page["events"][0]["payload"]["text"] = "rewritten"
    with pytest.raises(replicas.ReplicaError):
        ingest(pair, token, page=page)


def test_expired_replication_horizon_cannot_write(pair, monkeypatch):
    now = time.time()
    token, _ = grant(pair[1], now=now)
    monkeypatch.setattr(time, "time", lambda: now + 601)
    with pytest.raises(peer.HostedRoomGrantError):
        ingest(pair, token)


def test_authenticated_rename_does_not_stall_contiguous_history(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    rooms.rename_room(pair[0], room_id="room", event_id="rename", name="Revised workshop")
    result = ingest(pair, token, room_name="Revised workshop")
    assert result["stored_seq"] == 2
    assert replicas.replica_state(pair[1], room_id="room")["name"] == "Revised workshop"


def test_authenticated_name_update_cannot_change_fixed_membership(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    with pytest.raises(replicas.ReplicaError, match="metadata"):
        ingest(pair, token, members=MEMBERS[1:])


def test_replicated_disband_remains_terminal(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    rooms.disband_room(pair[0], room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    page = rooms.read_events(pair[0], room_id="room", include_disbanded=True)
    ingest(pair, token, page=page)
    assert replicas.replica_state(pair[1], room_id="room")["disbanded_at"] is not None
    page = copy.deepcopy(page)
    later = copy.deepcopy(page["events"][0])
    later.update(seq=3, event_id="late")
    page["events"].append(later)
    page.update(cursor=3, latest_seq=3, has_more=False)
    with pytest.raises(replicas.ReplicaError):
        ingest(pair, token, page=page)
