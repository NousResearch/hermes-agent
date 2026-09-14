"""Actual HTTP lookup framing, exact selection, and fresh reciprocal grants."""

import copy

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_controls as controls
from gateway import hosted_room_file_lookup as lookup
from gateway.hosted_room_file_contract import FileAccessError
from tests.gateway import test_hosted_room_control_files_client as wire
from tests.gateway.test_hosted_room_file_access import file_state, publish
from tests.gateway.platforms.test_api_server_room_control_files import file_api


def invoke(client, operation):
    if operation == "resolve":
        return client.resolve_file(
            target_profile="reviewer",
            code=lookup.selection_digest(wire.SCOPE, wire.METADATA)[:8],
        )
    return client.latest_shared_message(target_profile="reviewer")


@pytest.fixture
def payloads():
    return {
        "resolve": {
            "scope": wire.SCOPE,
            "selection": {
                **wire.page()["items"][0],
                "manifest_index": 0,
            },
        },
        "reply": {
            "scope": wire.SCOPE,
            "reply": {
                "event_id": "event-1",
                "seq": 1,
                "shared_at": 1.0,
                "text": "# Complete reply",
                "producer": {"kind": "member", "id": "peer", "label": "Reviewer"},
            },
        },
    }


@pytest.mark.parametrize("operation", ["resolve", "reply"])
def test_lookup_client_accepts_exact_scoped_metadata(payloads, monkeypatch, operation):
    monkeypatch.setattr(wire, "page", lambda: copy.deepcopy(payloads[operation]))
    with wire.host(catalog=True) as (client, requests):
        result = invoke(client, operation)
    assert (
        result
        == payloads[operation]["selection" if operation == "resolve" else "reply"]
    )
    assert len(requests) == 1 and "A" * 43 not in requests[0][0]


@pytest.mark.parametrize("operation", ["resolve", "reply"])
@pytest.mark.parametrize(
    "mode",
    [
        "old",
        "redirect",
        "malformed",
        "truncated",
        "large_catalog",
        "encoding",
        "scope_room_id",
        "scope_member_id",
        "scope_target_profile",
        "scope_authority_gateway_id",
        "bool_epoch",
    ],
)
def test_new_lookup_routes_keep_http_protections(
    payloads, monkeypatch, operation, mode
):
    monkeypatch.setattr(wire, "page", lambda: copy.deepcopy(payloads[operation]))
    with wire.host(mode, catalog=True) as (client, requests):
        with pytest.raises(FileAccessError) as error:
            invoke(client, operation)
    assert len(requests) == 1
    if mode == "old":
        assert error.value.code == "file_access_unsupported"


@pytest.mark.parametrize(
    "damage",
    ["wrong_code", "manifest_index", "private_path", "collision", "many", "text"],
)
def test_lookup_response_cannot_retarget_or_expand_selection(
    payloads, monkeypatch, damage
):
    value = copy.deepcopy(payloads["resolve"])
    item = value["selection"]
    if damage == "wrong_code":
        item["event_id"] = "another-event"
    elif damage == "manifest_index":
        item["manifest_index"] = True
    elif damage == "private_path":
        item["path"] = "/private/file"
    elif damage in {"collision", "many"}:
        value = {
            "scope": wire.SCOPE,
            "matches": [item] * (2 if damage == "collision" else 9),
        }
    elif damage == "text":
        value = {
            "scope": wire.SCOPE,
            "reply": {**payloads["reply"]["reply"], "text": "x" * 65537},
        }
    monkeypatch.setattr(wire, "page", lambda: copy.deepcopy(value))
    with wire.host(catalog=True) as (client, _requests):
        with pytest.raises(FileAccessError) as error:
            invoke(client, "reply" if damage == "text" else "resolve")
    assert error.value.code == "file_invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["resolve", "reply"])
async def test_authority_revocation_after_lookup_returns_no_metadata(
    file_api, monkeypatch, operation
):
    state = file_api
    item = publish(state, "secret-name.txt", actor={"kind": "member", "id": "peer"})
    name = "resolve_local_file" if operation == "resolve" else "latest_local_reply"
    original = getattr(lookup, name)

    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        controls.revoke_home_control_tokens(state.db, room_id="room-1")
        return result

    monkeypatch.setattr(lookup, name, revoke)
    suffix = (
        "files/resolve?code=" + lookup.selection_digest(state.room, item)[:8]
        if operation == "resolve"
        else "latest-reply"
    )
    async with TestClient(TestServer(state.app)) as client:
        result = await client.get(
            "/v1/room-controls/room-1/" + suffix, headers=state.headers
        )
        assert result.status == 403
        text = await result.text()
        assert "secret-name" not in text and item["event_id"] not in text


@pytest.mark.asyncio
async def test_lookup_rejects_query_expansion_and_bodies(file_api):
    state = file_api
    publish(state)
    async with TestClient(TestServer(state.app)) as client:
        for suffix in (
            "files/resolve?code=12345678&code=abcdefab",
            "files/resolve?code=12345678&recipient_member_id=ops",
            "files/resolve?code=not-hex",
            "latest-reply?member_id=ops",
        ):
            result = await client.get(
                "/v1/room-controls/room-1/" + suffix, headers=state.headers
            )
            assert result.status == 400
        result = await client.get(
            "/v1/room-controls/room-1/latest-reply", headers=state.headers, data=b"body"
        )
        assert result.status == 400
