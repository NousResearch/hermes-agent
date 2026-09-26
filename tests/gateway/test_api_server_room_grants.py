"""RoomLink grant authority: policy drift, gateway-owned secret, superseded authority."""

import time
from unittest.mock import MagicMock

import pytest

from gateway.platforms import api_server
from gateway.platforms import api_server_room_grants as room_grants

def test_grant_refresh_rejects_execution_policy_drift():
    claims = {"execution_policy_digest": "a" * 64}

    with pytest.raises(
        room_grants.RoomGrantReauthorizationRequired,
        match="execution policy changed",
    ):
        room_grants._require_unchanged_execution_policy(
            claims,
            {"policy_digest": "b" * 64},
        )

def test_grant_refresh_accepts_the_authorized_execution_policy():
    claims = {"execution_policy_digest": "a" * 64}

    assert (
        room_grants._require_unchanged_execution_policy(
            claims,
            {"policy_digest": "a" * 64},
        )
        is None
    )

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
