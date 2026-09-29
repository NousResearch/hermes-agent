"""Bounded Matrix reads keep thread identity and report unavailable decryption."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context


def _message(event_id, body, *, ts=None, thread=None, sender="@alice:server"):
    content = {"msgtype": "m.text", "body": body}
    if thread:
        content["m.relates_to"] = {"rel_type": "m.thread", "event_id": thread}
    return {"event_id": event_id, "sender": sender, "type": "m.room.message",
            "origin_server_ts": ts, "content": content}


def _visible(event_id, body, *, ts=None, thread=None):
    return {"event_id": event_id, "sender": "@alice:server", "body": body, "msgtype": "m.text",
            "thread_id": thread, "timestamp": ts, "sender_authorized": True}


def _client(chunk=(), *, event=None, crypto=None):
    return SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value={"chunk": list(chunk)})),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="s42")),
        get_event=AsyncMock(return_value=event), crypto=crypto,
    )


def _adapter(client, **overrides):
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    values = dict(
        _allowed_room_ids=set(),
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: user == "@alice:server",
    )
    values.update(overrides)
    adapter = SimpleNamespace(**values)
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )
    return adapter


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
        _message("$other", "other", thread="$else", sender="@bob:server"),
        {"event_id": "$encrypted", "sender": "@alice:server", "type": "m.room.encrypted", "content": {}},
        _message("$reply", "reply", thread="$root"),
    ]
    client = _client(raw, event=_message("$root", "start"))

    result = await read_matrix_context(_adapter(client), "thread", "!room:server", "$root", 5,
                                       requester="@alice:server")

    assert result == {
        "events": [_visible("$root", "start"), _visible("$reply", "reply", thread="$root")],
        "errors": [{"event_id": "$encrypted", "error": "missing decryption keys"}],
        "skipped": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "expected_ids"), [
    ("room", ["$r1", "$r2", "$r3"]),
    ("thread", ["$root", "$r1", "$r2", "$r3"]),
])
async def test_room_and_thread_reads_return_events_oldest_first(kind, expected_ids):
    newest_first = [_message(f"$r{n}", f"reply {n}", ts=n + 1, thread="$root") for n in (3, 2, 1)]
    client = _client(newest_first, event=_message("$root", "question", ts=1))

    result = await read_matrix_context(_adapter(client), kind, "!room:server", "$root", 10,
                                       requester="@alice:server")

    replies = [_visible(f"$r{n}", f"reply {n}", ts=n + 1, thread="$root") for n in (1, 2, 3)]
    expected = [_visible("$root", "question", ts=1)] + replies if kind == "thread" else replies
    assert [event["event_id"] for event in expected] == expected_ids
    assert result == {"events": expected, "errors": [], "skipped": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "expected_ids"), [
    ("room", ["$r1", "$r2", "$r3"]),
    ("thread", ["$root", "$r1", "$r2", "$r3"]),
])
async def test_room_and_thread_reads_return_events_oldest_first(kind, expected_ids):
    newest_first = [_message(f"$r{n}", f"reply {n}", ts=n + 1, thread="$root") for n in (3, 2, 1)]
    client = _client(newest_first, event=_message("$root", "question", ts=1))

    result = await read_matrix_context(_adapter(client), kind, "!room:server", "$root", 10,
                                       requester="@alice:server")

    replies = [_visible(f"$r{n}", f"reply {n}", ts=n + 1, thread="$root") for n in (1, 2, 3)]
    expected = [_visible("$root", "question", ts=1)] + replies if kind == "thread" else replies
    assert [event["event_id"] for event in expected] == expected_ids
    assert result == {"events": expected, "errors": [], "skipped": 0}


@pytest.mark.asyncio
async def test_read_rejects_unauthorized_requester_before_network():
    client = _client()
    adapter = _adapter(client, _is_sender_authorized=lambda user, **kw: False)

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
    reaction = {"event_id": "$reaction", "sender": "@alice:server", "type": "m.reaction",
                "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$secret", "key": "+1"}}}
    client = _client([reaction, encrypted], crypto=crypto)

    result = await read_matrix_context(_adapter(client), "room", "!room:server", None, 5,
                                       requester="@alice:server")

    assert result == {"events": [_visible("$secret", "secret")], "errors": [], "skipped": 1}
    query = client.api.request.await_args.kwargs["query_params"]
    assert {**query, "filter": json.loads(query["filter"])} == {
        "from": "s42", "dir": "b", "limit": "5",
        "filter": {"types": ["m.room.message", "m.room.encrypted", "m.sticker"]},
    }
    crypto.decrypt_megolm_event.assert_awaited_once_with(encrypted)


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    {"event_id": "$target", "sender": "@alice:server", "type": "m.reaction",
     "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$other", "key": "+1"}}},
    {"event_id": "$target", "sender": "@alice:server", "type": "m.room.topic", "state_key": "",
     "content": {"topic": "Planning"}},
    {"event_id": "$target", "sender": "@alice:server", "type": "m.room.message", "content": {},
     "unsigned": {"redacted_because": {"type": "m.room.redaction"}}},
], ids=["reaction", "state", "redacted"])
async def test_event_read_without_message_content_is_an_error(event):
    client = _client(event=event)

    result = await read_matrix_context(_adapter(client), "event", "!room:server", "$target", 1,
                                       requester="@alice:server")

    assert result == {"error": "Matrix event has no message content"}


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
