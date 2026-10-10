"""RoomLink grant authority: policy drift, gateway-owned secret, superseded authority."""

import time
from unittest.mock import MagicMock

import pytest

from gateway.platforms import api_server

def test_room_grant_secret_stays_gateway_owned_on_named_profile(
    tmp_path, monkeypatch
):
    from gateway.hosted_room_peer import gateway_room_grant_secret

    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._api_key = "gateway-api-key-1234567890"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_token = api_server._api_request_profile.set("reviewer")
    try:
        assert adapter._room_grant_secret() == gateway_room_grant_secret()
    finally:
        api_server._api_request_profile.reset(profile_token)

def test_superseded_room_authority_cannot_reuse_its_grant(tmp_path, monkeypatch):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import (
        gateway_room_grant_secret,
        issue_room_grant,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret = gateway_room_grant_secret()
    now = time.time()
    common = {
        "grant_id": "grant-old",
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-old",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": hosted_rooms.local_authority_gateway_id(),
        "target_profile": "reviewer",
        "capability_digest": "c" * 64,
        "execution_policy_digest": "d" * 64,
        "issued_at": now,
        "ttl_seconds": 3600,
    }
    old_grant = issue_room_grant(secret, **common)
    old_claims = {
        key: value
        for key, value in common.items()
        if key
        in {
            "room_id",
            "home_install_id",
            "authority_gateway_id",
            "authority_epoch",
            "member_id",
            "target_install_id",
            "target_profile",
        }
    }
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims=old_claims,
        expires_at=now + 3600,
        now=now,
    )

    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    request = MagicMock(headers={"Authorization": f"HermesRoom {old_grant}"})
    assert adapter._room_grant_claims(request, permission="status")[
        "authority_gateway_id"
    ] == "gateway-old"

    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims={
            **old_claims,
            "authority_gateway_id": "gateway-new",
            "authority_epoch": 2,
            "member_id": "member-new",
        },
        expires_at=now + 3600,
        now=now,
    )

    with pytest.raises(ValueError, match="no longer current"):
        adapter._room_grant_claims(request, permission="status")


def test_revoked_older_grant_stays_revoked_after_shorter_grant_horizon(
    tmp_path, monkeypatch
):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import (
        decode_room_grant,
        gateway_room_grant_secret,
        issue_room_grant,
    )
    from gateway.platforms.api_server_room_grants import (
        RoomGrantReauthorizationRequired,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = {"value": 100.0}
    monkeypatch.setattr(
        "gateway.hosted_rooms_common.time.time", lambda: now["value"]
    )
    secret = gateway_room_grant_secret()
    scope = {
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": hosted_rooms.local_authority_gateway_id(),
        "target_profile": "reviewer",
        "execution_policy_digest": "d" * 64,
    }

    def issue(
        grant_id, issued_at, ttl_seconds, status_expires_at, capability_digest
    ):
        return issue_room_grant(
            secret,
            grant_id=grant_id,
            **scope,
            capability_digest=capability_digest,
            issued_at=issued_at,
            ttl_seconds=ttl_seconds,
            status_expires_at=status_expires_at,
        )

    old_grant = issue("grant-old-long", 100, 300, 1000, "a" * 64)
    old_claims = decode_room_grant(
        secret, old_grant, permission="status", now=100
    )
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims=old_claims,
        expires_at=1000,
        now=100,
    )

    short_grant = issue("grant-short", 150, 60, 250, "b" * 64)
    short_claims = decode_room_grant(
        secret, short_grant, permission="status", now=150
    )
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims=short_claims,
        expires_at=250,
        now=150,
    )
    hosted_rooms.revoke_room_grant_scope(
        hosted_rooms.default_db_path(),
        claims=short_claims,
        expires_at=250,
        now=200,
    )

    new_grant = issue("grant-new", 201, 300, 1000, "b" * 64)
    new_claims = decode_room_grant(
        secret, new_grant, permission="status", now=201
    )
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims=new_claims,
        expires_at=1000,
        now=201,
    )

    now["value"] = 251
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    old_request = MagicMock(
        headers={"Authorization": f"HermesRoom {old_grant}"}
    )
    with pytest.raises(RoomGrantReauthorizationRequired, match="revoked"):
        adapter._room_grant_claims(old_request, permission="status")

    new_request = MagicMock(
        headers={"Authorization": f"HermesRoom {new_grant}"}
    )
    assert adapter._room_grant_claims(new_request, permission="status")[
        "grant_id"
    ] == "grant-new"
