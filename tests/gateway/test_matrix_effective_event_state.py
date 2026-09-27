"""Matrix history exposes the current event state without changing event identity."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.effective_event import MatrixEffectiveEvent, effective_event
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache
from plugins.platforms.matrix.room_context import fetch_room_entries
from plugins.platforms.matrix.thread_context import fetch_thread_entries


ROOM = "!room:example.org"
SENDER = "@alice:example.org"


def _original(event_id: str, body: str, *, root: str | None = None) -> dict:
    content = {"msgtype": "m.text", "body": body}
    if root:
        content["m.relates_to"] = {"rel_type": "m.thread", "event_id": root}
    return {
        "room_id": ROOM, "event_id": event_id, "sender": SENDER,
        "type": "m.room.message", "content": content,
    }


def _edited(original: dict, body: str) -> dict:
    event_id = original["event_id"]
    return {
        **original,
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.message",
            "content": {
                "msgtype": "m.text", "body": f"* {body}",
                "m.new_content": {
                    "msgtype": "m.text", "body": body,
                    "m.relates_to": {"rel_type": "m.thread", "event_id": "$wrong"},
                },
                "m.relates_to": {"rel_type": "m.replace", "event_id": event_id},
            },
        }}},
    }


def _adapter(client) -> SimpleNamespace:
    return SimpleNamespace(
        _client=client, _joined_rooms={ROOM}, _user_id="@bot:example.org",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )


@pytest.mark.asyncio
async def test_event_read_uses_latest_valid_edit_and_keeps_original_thread_relation():
    original = _edited(_original("$child", "before", root="$root"), "after")

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return original
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)

    result = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)

    assert result == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "after", "msgtype": "m.text",
        "thread_id": "$root", "timestamp": None, "sender_authorized": True,
        "edited": True,
    }], "errors": []}


@pytest.mark.asyncio
async def test_event_read_reports_original_redaction_without_exposing_bundled_edit():
    original = _edited(_original("$child", "before"), "after")
    original["unsigned"]["redacted_because"] = {"event_id": "$redaction"}

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return original
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)

    result = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)

    assert result == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "[redacted]", "msgtype": None,
        "thread_id": None, "timestamp": None, "sender_authorized": True,
        "redacted": True,
    }], "errors": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["sender", "type", "target", "redacted"])
async def test_invalid_replacement_keeps_original_content(invalid: str):
    original = _edited(_original("$child", "before"), "after")
    replacement = original["unsigned"]["m.relations"]["m.replace"]
    if invalid == "sender":
        replacement["sender"] = "@mallory:example.org"
    elif invalid == "type":
        replacement["type"] = "m.reaction"
    elif invalid == "target":
        replacement["content"]["m.relates_to"]["event_id"] = "$other"
    else:
        replacement["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}

    async def request(_method, path, **_kwargs):
        return original if "/event/" in path else {"chunk": []}

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)

    result = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)

    assert result == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "before", "msgtype": "m.text",
        "thread_id": None, "timestamp": None, "sender_authorized": True,
    }], "errors": []}


@pytest.mark.asyncio
async def test_encrypted_replacement_uses_owning_crypto_and_reports_missing_edit_key():
    class SessionNotFound(Exception):
        pass

    original = {
        "room_id": ROOM, "event_id": "$child", "sender": SENDER,
        "type": "m.room.encrypted",
        "content": {"ciphertext": "original", "m.relates_to": {"rel_type": "m.thread", "event_id": "$root"}},
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.encrypted",
            "content": {"ciphertext": "replacement", "m.relates_to": {
                "rel_type": "m.replace", "event_id": "$child",
            }},
        }}},
    }
    original_text = SimpleNamespace(content={"msgtype": "m.text", "body": "before"})
    replacement_text = SimpleNamespace(content=SimpleNamespace(serialize=lambda: {
        "msgtype": "m.text", "body": "* after",
        "m.new_content": {"msgtype": "m.text", "body": "after"},
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
    }))
    decrypt = AsyncMock(side_effect=[original_text, replacement_text, original_text, SessionNotFound()])

    async def request(_method, path, **_kwargs):
        return original if "/event/" in path else {"chunk": []}

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt),
    )
    mautrix = SimpleNamespace(types=SimpleNamespace(Event=SimpleNamespace(deserialize=lambda raw: raw)))
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix.types}):
        visible = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)
        missing = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)

    assert visible == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "after", "msgtype": "m.text",
        "thread_id": "$root", "timestamp": None, "sender_authorized": True, "edited": True,
    }], "errors": []}
    assert missing == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "before", "msgtype": "m.text",
        "thread_id": "$root", "timestamp": None, "sender_authorized": True,
    }], "errors": [{"event_id": "$edit", "error": "missing decryption keys"}]}
    assert [call.args[0] for call in decrypt.await_args_list] == [
        original, original["unsigned"]["m.relations"]["m.replace"],
        original, original["unsigned"]["m.relations"]["m.replace"],
    ]


@pytest.mark.asyncio
async def test_pinned_mautrix_typed_edit_keeps_new_content():
    mautrix_types = pytest.importorskip("mautrix.types")
    original = {
        "room_id": ROOM, "event_id": "$child", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "original"},
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.encrypted", "content": {
                "ciphertext": "replacement",
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
            },
        }}},
    }
    decrypted_original = mautrix_types.Event.deserialize({
        **_original("$child", "before"), "origin_server_ts": 1,
    })
    decrypted_edit = mautrix_types.Event.deserialize({
        "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
        "type": "m.room.message", "origin_server_ts": 2,
        "content": {
            "msgtype": "m.text", "body": "* after",
            "m.new_content": {"msgtype": "m.text", "body": "after"},
            "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
        },
    })
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = [(decrypted_original, None), (decrypted_edit, None)]
        state = await effective_event(SimpleNamespace(), original)

    assert decrypted_edit.content.serialize() == {
        "msgtype": "m.text", "body": "* after",
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
        "m.new_content": {"msgtype": "m.text", "body": "after"},
    }
    assert state == MatrixEffectiveEvent(
        {"msgtype": "m.text", "body": "after"}, original["content"], edited=True,
    )


@pytest.mark.asyncio
async def test_encrypted_edit_is_visible_in_room_catch_up():
    raw = {
        "room_id": ROOM, "event_id": "$original", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "original"},
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.encrypted", "content": {
                "ciphertext": "replacement",
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$original"},
            },
        }}},
    }
    decrypted = [
        SimpleNamespace(content={"msgtype": "m.text", "body": "before"}),
        SimpleNamespace(content=SimpleNamespace(serialize=lambda: {
            "msgtype": "m.text", "body": "* after",
            "m.new_content": {"msgtype": "m.text", "body": "after"},
            "m.relates_to": {"rel_type": "m.replace", "event_id": "$original"},
        })),
    ]

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [raw]}
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    crypto = SimpleNamespace(decrypt_megolm_event=AsyncMock(side_effect=decrypted))
    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=crypto)
    mautrix = SimpleNamespace(types=SimpleNamespace(Event=SimpleNamespace(deserialize=lambda event: event)))
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix.types}):
        entries = await fetch_room_entries(client, MatrixEventContextCache(), ROOM, "$current", limit=1)

    assert entries == [MatrixEventContext(SENDER, "after")]
    assert [call.args[0] for call in crypto.decrypt_megolm_event.await_args_list] == [
        raw, raw["unsigned"]["m.relations"]["m.replace"],
    ]


@pytest.mark.asyncio
async def test_catch_up_renders_edited_room_and_thread_messages():
    room_event = _edited(_original("$room", "before room"), "after room")
    thread_event = _edited(_original("$child", "before thread", root="$root"), "after thread")

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"start": "boundary", "chunk": [room_event, thread_event]}
        if "/m.thread" in path:
            return {"chunk": [thread_event]}
        if "/m.annotation" in path:
            return {"chunk": []}
        if "/event/" in path:
            return _edited(_original("$root", "root before"), "root after")
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)
    cache = MatrixEventContextCache()
    cache.store(ROOM, "$root", MatrixEventContext(SENDER, "root before"))

    room = await fetch_room_entries(client, cache, ROOM, "$current", limit=2)
    thread = await fetch_thread_entries(client, cache, ROOM, "$root", limit=2, before_event_id="$current")

    assert room == [MatrixEventContext(SENDER, "after room")]
    assert thread == [
        MatrixEventContext(SENDER, "root after"), MatrixEventContext(SENDER, "after thread"),
    ]


@pytest.mark.asyncio
async def test_room_catch_up_keeps_redaction_even_when_original_body_is_stale():
    original = _edited(_original("$child", "before"), "after")
    original["unsigned"]["redacted_because"] = {"event_id": "$redaction"}

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [original]}
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)

    entries = await fetch_room_entries(client, MatrixEventContextCache(), ROOM, "$current", limit=1)

    assert entries == [MatrixEventContext(SENDER, "[redacted]", redacted=True)]


@pytest.mark.asyncio
async def test_reply_target_uses_raw_aggregated_state_and_blocks_redacted_original():
    original = _edited(_original("$child", "before"), "after")
    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value=original)),
        get_event=AsyncMock(side_effect=AssertionError("typed get_event loses bundled edits")),
        crypto=None,
    )
    cache = MatrixEventContextCache()

    edited = await cache.resolve(client, ROOM, "$child")
    cache = MatrixEventContextCache()
    original["unsigned"]["redacted_because"] = {"event_id": "$redaction"}
    redacted = await cache.resolve(client, ROOM, "$child")

    assert (edited, redacted) == (MatrixEventContext(SENDER, "after"), None)
    client.get_event.assert_not_awaited()
