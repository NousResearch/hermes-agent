"""Bounded Matrix reads keep thread identity and report unavailable decryption."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context


def test_gateway_binds_receiving_adapter_and_gateway_loop_for_matrix_reads():
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext, SessionSource
    from gateway.session_context import get_session_transport

    receiving = SimpleNamespace(supports_async_delivery=True)
    gateway_loop = object()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: SimpleNamespace()}
    runner._gateway_loop = gateway_loop
    runner._delivery_adapter_for = lambda source: receiving
    context = SessionContext(
        source=SessionSource(platform=Platform.MATRIX, chat_id="!room:server", user_id="@alice:server"),
        connected_platforms=[], home_channels={},
    )

    tokens = runner._set_session_env(context)
    try:
        bound = get_session_transport()
    finally:
        runner._clear_session_env(tokens)

    assert (bound, get_session_transport()) == ((receiving, gateway_loop), (None, None))


@pytest.mark.asyncio
async def test_read_thread_filters_unrelated_events_and_reports_missing_keys():
    raw = [
        {"event_id": "$other", "sender": "@bob:server", "type": "m.room.message",
         "content": {"msgtype": "m.text", "body": "other", "m.relates_to": {"rel_type": "m.thread", "event_id": "$else"}}},
        {"event_id": "$encrypted", "sender": "@alice:server", "type": "m.room.encrypted", "content": {}},
        {"event_id": "$reply", "sender": "@alice:server", "type": "m.room.message",
         "content": {"msgtype": "m.text", "body": "reply", "m.relates_to": {"rel_type": "m.thread", "event_id": "$root"}}},
    ]
    root = {"event_id": "$root", "sender": "@alice:server", "type": "m.room.message",
            "content": {"msgtype": "m.text", "body": "start"}}
    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value={"chunk": raw})),
        get_event=AsyncMock(return_value=root), crypto=None,
    )
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: user == "@alice:server",
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )

    result = await read_matrix_context(adapter, "thread", "!room:server", "$root", 5,
                                       requester="@alice:server")

    assert result == {
        "events": [
            {"event_id": "$root", "sender": "@alice:server", "body": "start",
             "msgtype": "m.text", "thread_id": None, "timestamp": None, "sender_authorized": True},
            {"event_id": "$reply", "sender": "@alice:server", "body": "reply",
             "msgtype": "m.text", "thread_id": "$root", "timestamp": None, "sender_authorized": True},
        ],
        "errors": [{"event_id": "$encrypted", "error": "missing decryption keys"}],
    }


@pytest.mark.asyncio
async def test_read_rejects_unauthorized_requester_before_network():
    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock()))
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: False,
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )

    result = await read_matrix_context(adapter, "room", "!room:server", None, 5,
                                       requester="@alice:server")

    assert result == {"error": "Matrix requester is not authorized for this room"}
    client.api.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_room_uses_sync_token_and_decrypts_with_owning_client(monkeypatch):
    encrypted = {"event_id": "$secret", "sender": "@alice:server", "type": "m.room.encrypted", "content": {}}
    types_module = ModuleType("mautrix.types")
    types_module.EncryptedEvent = SimpleNamespace(deserialize=lambda raw: raw)
    types_module.JSON = lambda raw: raw
    mautrix_module = ModuleType("mautrix")
    mautrix_module.types = types_module
    monkeypatch.setitem(sys.modules, "mautrix", mautrix_module)
    monkeypatch.setitem(sys.modules, "mautrix.types", types_module)
    crypto = SimpleNamespace(decrypt_megolm_event=AsyncMock(return_value={
        "content": {"msgtype": "m.text", "body": "secret"}
    }))
    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value={"chunk": [encrypted]})),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="s42")),
        crypto=crypto,
    )
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )

    result = await read_matrix_context(adapter, "room", "!room:server", None, 5,
                                       requester="@alice:server")

    assert result == {"events": [{"event_id": "$secret", "sender": "@alice:server",
                                  "body": "secret", "msgtype": "m.text", "thread_id": None,
                                  "timestamp": None, "sender_authorized": True}], "errors": []}
    assert client.api.request.await_args.kwargs["query_params"] == {"from": "s42", "dir": "b", "limit": "5"}
    crypto.decrypt_megolm_event.assert_awaited_once_with(encrypted)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["membership", "room", "requester"])
async def test_read_policy_is_current_after_identity_resolution(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    room = "!room:server"
    requester = "@alice:server"
    adapter = object.__new__(MatrixAdapter)
    adapter._client = None
    adapter._joined_rooms = {room}
    adapter._allowed_room_ids = {room}
    adapter._authorization_check = lambda *_args, **_kwargs: True

    async def resolve_identity(_room: str) -> bool:
        if policy == "membership":
            adapter._joined_rooms.clear()
        elif policy == "room":
            adapter._allowed_room_ids = {"!other:server"}
        else:
            adapter._authorization_check = lambda *_args, **_kwargs: False
        return False

    identity = AsyncMock(side_effect=resolve_identity)
    monkeypatch.setattr(adapter, "_is_dm_room", identity)
    result = await adapter.read_matrix_context(
        "room",
        room,
        None,
        1,
        requester=requester,
    )
    error = (
        "Matrix requester is not authorized for this room"
        if policy == "requester"
        else "Matrix room is not allowed or joined"
    )
    assert (result, identity.await_count) == ({"error": error}, 1)
