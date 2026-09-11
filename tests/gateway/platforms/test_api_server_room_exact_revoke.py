"""The source HTTP endpoint retires one signed bearer in both enforcing stores."""

import base64
import contextvars
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_room_grant_state as state
from gateway import hosted_rooms
from gateway.hosted_room_peer import (
    HostedRoomGrantError,
    decode_room_grant,
    issue_room_grant,
)
from gateway.platforms import api_server, api_server_room_grants as grants


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    secret = b"source-exact-grant-test-secret-only"
    stores = (tmp_path / "state.db", tmp_path / "profiles" / "reviewer" / "state.db")
    monkeypatch.setattr(state, "grant_state_db_paths", lambda: stores)
    monkeypatch.setattr(
        hosted_rooms, "local_authority_gateway_id", lambda: "install:peer"
    )
    profile = contextvars.ContextVar("exact-profile", default="reviewer")
    monkeypatch.setattr(api_server, "_api_request_profile", profile)
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_secret = lambda: secret
    adapter._read_json_body = AsyncMock(return_value=({}, None))
    handler = next(
        handler
        for method, path, handler in grants._http_routes(adapter)
        if (method, path) == ("POST", "/v1/room-members/grants/revoke-exact")
    )

    def token(name="loser", **overrides):
        args = dict(
            grant_id=name,
            room_id="room-1",
            home_install_id="install:home",
            authority_gateway_id="install:home",
            authority_epoch=1,
            member_id="member",
            target_install_id="install:peer",
            target_profile="reviewer",
            issued_at=time.time() - 10,
            ttl_seconds=3600,
        )
        return issue_room_grant(secret, **{**args, **overrides})

    async def revoke(value, scheme="HermesRoom"):
        return await handler(
            SimpleNamespace(headers={"Authorization": f"{scheme} {value}"})
        )

    return SimpleNamespace(
        secret=secret,
        stores=stores,
        token=token,
        revoke=revoke,
        profile=profile,
        adapter=adapter,
    )


@pytest.mark.asyncio
async def test_exact_route_is_idempotent_without_retiring_sibling_scope(endpoint):
    losing, winning = endpoint.token(), endpoint.token("winner")
    losing_claims = decode_room_grant(endpoint.secret, losing, permission="status")
    winning_claims = decode_room_grant(endpoint.secret, winning, permission="status")
    for db in endpoint.stores:
        hosted_rooms.reserve_peer_room(
            db, claims=winning_claims, expires_at=winning_claims["expires_at"]
        )
    for _ in range(2):
        assert (await endpoint.revoke(losing)).status == 200
    for db in endpoint.stores:
        assert hosted_rooms.room_grant_is_revoked(db, claims=losing_claims)
        assert not hosted_rooms.room_grant_is_revoked(db, claims=winning_claims)
        assert hosted_rooms.peer_room_grant_is_current(db, claims=winning_claims)


def test_exact_endpoint_is_in_the_real_adapter_route_table(endpoint):
    routes = [
        (method, path) for method, path, _ in endpoint.adapter._http_route_table()
    ]
    assert routes.count(("POST", "/v1/room-members/grants/revoke-exact")) == 1
    assert routes.count(("POST", "/v1/room-members/grants/revoke")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("grant_id", ["grant/user", "grant@home", "g" * 256])
async def test_exact_revoke_preserves_the_existing_wire_grant_id_grammar(
    endpoint, grant_id
):
    value = endpoint.token(grant_id)
    claims = decode_room_grant(endpoint.secret, value, permission="status")
    assert (await endpoint.revoke(value)).status == 200
    for db in endpoint.stores:
        assert hosted_rooms.room_grant_is_revoked(db, claims=claims)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", ["bearer", "profile", "installation", "signature", "permission", "body"]
)
async def test_exact_route_rejects_invalid_authority_without_state_changes(
    endpoint, invalid
):
    kwargs = {}
    if invalid == "profile":
        kwargs["target_profile"] = "other"
    elif invalid == "installation":
        kwargs["target_install_id"] = "install:other"
    elif invalid == "permission":
        kwargs["permissions"] = ("dispatch",)
    value = endpoint.token(**kwargs)
    if invalid == "signature":
        value = value.rsplit(".", 1)[0] + ".invalid"
    if invalid == "body":
        endpoint.adapter._read_json_body = AsyncMock(
            return_value=({"scope": "all"}, None)
        )
    response = await endpoint.revoke(
        value, "Bearer" if invalid == "bearer" else "HermesRoom"
    )
    assert response.status == (400 if invalid == "body" else 401)
    assert not any(db.exists() for db in endpoint.stores)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_revocation_only_accepts_expired_signed_tokens_without_restoring_authority(
    endpoint, legacy
):
    value = endpoint.token(issued_at=time.time() - 7200)
    if legacy:
        encoded = value.partition(".")[0]
        payload = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        payload.pop("status_expires_at")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        encode = lambda data: (
            base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
        )
        value = (
            encode(raw)
            + "."
            + encode(hmac.new(endpoint.secret, raw, hashlib.sha256).digest())
        )
    for permission in ("dispatch", "status"):
        with pytest.raises(HostedRoomGrantError):
            decode_room_grant(endpoint.secret, value, permission=permission)
    with pytest.raises(HostedRoomGrantError, match="only for revocation"):
        decode_room_grant(
            endpoint.secret,
            value,
            permission="dispatch",
            allow_expired_for_revocation=True,
        )
    assert (await endpoint.revoke(value)).status == 200
    with pytest.raises(HostedRoomGrantError):
        decode_room_grant(endpoint.secret, value, permission="status")


def test_exact_cleanup_attempts_profile_store_even_when_shared_store_fails(
    endpoint, monkeypatch
):
    claims = decode_room_grant(endpoint.secret, endpoint.token(), permission="status")
    calls = []
    revoke = hosted_rooms.revoke_room_grant_id

    def fail_shared(db, **kwargs):
        calls.append(db)
        if db == endpoint.stores[0]:
            raise OSError("shared store unavailable")
        return revoke(db, **kwargs)

    monkeypatch.setattr(hosted_rooms, "revoke_room_grant_id", fail_shared)
    with pytest.raises(OSError, match="shared store unavailable"):
        state.revoke_grant_state(
            endpoint.stores, claims=claims, expires_at=claims["expires_at"], exact=True
        )
    assert calls == list(endpoint.stores)
    assert hosted_rooms.room_grant_is_revoked(endpoint.stores[1], claims=claims)
