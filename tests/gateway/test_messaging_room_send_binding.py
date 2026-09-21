"""Explicit native-owner consent for one private canonical Group Send."""
import json

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_messaging_inventory_binding import (
    bound,
    enrolled,
    grant_params,
    recipient,
    rpc,
)
from tests.gateway.test_messaging_room_read_binding import room_grant_params, room_rpc


def _native(c, *, subject="native-alice", submit=True):
    capabilities = ["session:read", "session:create", "session:control"]
    if submit:
        capabilities.append("session:submit")
    return AuthorityConnection(
        c.authority,
        object(),
        {
            "user_id": subject,
            "provider": "local",
            "profile_id": str(c.home),
            "instance_id": c.authority.instance_id,
            "capabilities": capabilities,
            "native_bootstrap": True,
        },
        operator=True,
    )


async def _read_grant(c):
    inventory = await enrolled(c)
    response = await room_rpc(c.alice, params=room_grant_params(inventory))
    assert "result" in response, response
    return inventory, response["result"]


def send_grant_params(read_grant, **changes):
    return {
        "request_id": "send-grant-one",
        "recipient": recipient(),
        "room_id": "alice-room",
        "room_read_binding_id": read_grant["binding_id"],
        "room_read_generation": read_grant["generation"],
        "expected_generation": 0,
    } | changes


def send_revoke_params(read_grant, send_grant, **changes):
    return send_grant_params(read_grant) | {
        "request_id": "send-revoke-one",
        "binding_id": send_grant["binding_id"],
        "expected_generation": send_grant["generation"],
    } | changes


async def send_rpc(connection, method="grant", params=None):
    assert params is not None
    return await connection.dispatch(
        {
            "id": 3,
            "method": "groups.messaging.room.send." + method,
            "params": params,
        }
    )


def send_rows(c):
    with c.db._read_ctx() as conn:
        return tuple(
            tuple(row)
            for row in conn.execute(
                "SELECT key,value FROM state_meta "
                "WHERE key LIKE 'gateway.messaging.send.v1.%' ORDER BY key"
            )
        )


@pytest.mark.asyncio
async def test_exact_submit_owner_grants_revokes_and_regrants_one_send_lineage(bound):
    inventory, read_grant = await _read_grant(bound)
    owner = _native(bound)
    params = send_grant_params(read_grant)

    response = await send_rpc(owner, params=params)
    assert "result" in response, response  # Semantic RED: method is not registered yet.
    first = response["result"]
    assert first == {
        "binding_id": first["binding_id"],
        "generation": 1,
        "active": True,
    }
    assert first["binding_id"].startswith("mrs-")
    assert (await send_rpc(owner, params=params))["result"] == first

    records = [
        json.loads(value)
        for key, value in send_rows(bound)
        if ".binding." in key
    ]
    assert records == [
        {
            "version": 1,
            "recipient": recipient(),
            "profile_id": str(bound.home),
            "owner": "native-alice",
            "room_id": "alice-room",
            "scope": "groups.send",
            "inventory_binding_id": inventory["binding_id"],
            "room_read_binding_id": read_grant["binding_id"],
            "room_read_generation": read_grant["generation"],
            "room_ref": read_grant["room_ref"],
            "binding_id": first["binding_id"],
            "generation": 1,
            "active": True,
        }
    ]

    revoked = (await send_rpc(
        owner, "revoke", send_revoke_params(read_grant, first)
    ))["result"]
    assert revoked == {
        "binding_id": first["binding_id"],
        "generation": 2,
        "active": False,
    }
    assert (await send_rpc(
        owner, "revoke", send_revoke_params(read_grant, first)
    ))["result"] == revoked

    second = (await send_rpc(
        owner,
        params=send_grant_params(
            read_grant,
            request_id="send-grant-two",
            expected_generation=revoked["generation"],
        ),
    ))["result"]
    assert second["binding_id"] != first["binding_id"]
    assert second["generation"] == 3


@pytest.mark.asyncio
async def test_send_grant_requires_submit_and_exact_owner_recipient_read_and_service(bound):
    _inventory, read_grant = await _read_grant(bound)
    owner = _native(bound)
    good = send_grant_params(read_grant)
    before = send_rows(bound)

    attempts = [
        (_native(bound, submit=False), good),
        (_native(bound, subject="native-bob"), good),
        (owner, good | {"recipient": recipient(user_id="other")}),
        (owner, good | {"room_id": "bob-room"}),
        (owner, good | {"room_read_binding_id": "mrr-" + "0" * 32}),
        (owner, good | {"room_read_generation": read_grant["generation"] + 1}),
        (owner, good | {"expected_generation": 1}),
        (owner, good | {"unexpected": True}),
    ]
    for connection, params in attempts:
        assert "error" in await send_rpc(connection, params=params)
        assert send_rows(bound) == before

    service = bound.authority.hosted_room_service
    bound.authority.hosted_room_service = None
    try:
        assert "error" in await send_rpc(owner, params=good)
        assert send_rows(bound) == before
    finally:
        bound.authority.hosted_room_service = service


@pytest.mark.asyncio
async def test_send_capacity_reserves_one_future_revoke(bound, monkeypatch):
    from gateway import session_group_messaging_send as send_binding

    _inventory, read_grant = await _read_grant(bound)
    owner = _native(bound)
    before = send_rows(bound)
    monkeypatch.setattr(send_binding, "_MAX_REQUESTS", 1)
    assert "error" in await send_rpc(owner, params=send_grant_params(read_grant))
    assert send_rows(bound) == before

    monkeypatch.setattr(send_binding, "_MAX_REQUESTS", 2)
    granted = (await send_rpc(owner, params=send_grant_params(read_grant)))["result"]
    assert "result" in await send_rpc(
        owner, "revoke", send_revoke_params(read_grant, granted)
    )


@pytest.mark.asyncio
async def test_read_regrant_invalidates_send_and_stale_context_never_inherits_replacement(bound):
    inventory, read_grant = await _read_grant(bound)
    owner = _native(bound)
    send_grant = (await send_rpc(owner, params=send_grant_params(read_grant)))["result"]

    event = bound.event()
    event.text = "/group 1 send review this"
    from gateway.session_group_messaging_send import attest_room_send

    captured = attest_room_send(bound.runner, event, read_grant["room_ref"], "review this")
    assert captured.actor.subject.startswith("messaging:")
    assert captured.actor.subject != captured.owner == owner.actor.subject
    assert captured.actor.capabilities == frozenset({"session:read"})
    from gateway.session_group_controls import dispatch_group_control
    from hermes_state_runtime import RuntimeStoreError

    exact = {
        "room_id": captured.room_id,
        "event_id": captured.client_event_id,
        "payload": {"text": "review this", "thread_id": captured.client_event_id},
    }
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await dispatch_group_control(
            captured, "groups.send", exact | {"payload": exact["payload"] | {"text": "changed"}}
        )
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await dispatch_group_control(
            captured, "groups.stop", {"room_id": captured.room_id, "cancel_id": "no"}
        )

    list_revoke = {
        "request_id": "list-revoke",
        "recipient": recipient(),
        "binding_id": inventory["binding_id"],
        "expected_generation": inventory["generation"],
    }
    assert "result" in await rpc(bound.alice, "revoke", list_revoke)
    replacement_inventory = (await rpc(
        bound.alice,
        params=grant_params(request_id="list-regrant", expected_generation=2),
    ))["result"]
    replacement_read = (await room_rpc(
        bound.alice,
        params=room_grant_params(
            replacement_inventory,
            request_id="room-read-regrant",
            expected_generation=read_grant["generation"],
        ),
    ))["result"]
    replacement_send = (await send_rpc(
        owner,
        params=send_grant_params(
            replacement_read,
            request_id="send-after-read-regrant",
            expected_generation=send_grant["generation"],
        ),
    ))["result"]
    assert replacement_send["binding_id"] != send_grant["binding_id"]
    with pytest.raises(Exception, match="stale|permission_denied"):
        captured.require_current()
