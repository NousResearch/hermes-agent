"""Matrix history exposes the current event state without changing event identity."""

import asyncio
from contextlib import ExitStack
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.effective_event import MatrixEffectiveEvent, effective_event
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache
from plugins.platforms.matrix.room_context import MatrixHistoryContext, fetch_room_entries
from plugins.platforms.matrix.thread_context import fetch_thread_entries
from tests.gateway.test_matrix import _rendered


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
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = SimpleNamespace(
        _client=client, _joined_rooms={ROOM}, _user_id="@bot:example.org",
        _event_context_cache=MatrixEventContextCache(),
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )
    return adapter


def _edit_store(content: dict) -> SimpleNamespace:
    plaintext = json.dumps({"room_id": ROOM, "type": "m.room.message", "content": content})
    session = SimpleNamespace(decrypt=lambda _ciphertext: (plaintext, 0))
    return SimpleNamespace(get_group_session=AsyncMock(return_value=session))


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
    }], "errors": [], "skipped": 0}


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
    }], "errors": [], "skipped": 0}


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
    }], "errors": [], "skipped": 0}


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
            "content": {"ciphertext": "replacement", "session_id": "sess", "m.relates_to": {
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
        crypto=SimpleNamespace(
            decrypt_megolm_event=decrypt,
            crypto_store=_edit_store({
                "msgtype": "m.text", "body": "* after",
                "m.new_content": {"msgtype": "m.text", "body": "after"},
            }),
        ),
    )
    mautrix = SimpleNamespace(types=SimpleNamespace(EncryptedEvent=SimpleNamespace(deserialize=lambda raw: raw), JSON=lambda raw: raw))
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix.types}):
        visible = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)
        missing = await read_matrix_context(_adapter(client), "event", ROOM, "$child", 5, requester=SENDER)

    assert visible == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "after", "msgtype": "m.text",
        "thread_id": "$root", "timestamp": None, "sender_authorized": True, "edited": True,
    }], "errors": [], "skipped": 0}
    assert missing == {"events": [{
        "event_id": "$child", "sender": SENDER, "body": "before", "msgtype": "m.text",
        "thread_id": "$root", "timestamp": None, "sender_authorized": True,
    }], "errors": [{"event_id": "$edit", "error": "missing decryption keys"}], "skipped": 0}
    assert [call.args[0] for call in decrypt.await_args_list] == [
        original, original["unsigned"]["m.relations"]["m.replace"],
        original, original["unsigned"]["m.relations"]["m.replace"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_new_content", [False, True])
async def test_pinned_mautrix_typed_edit_keeps_new_content(include_new_content: bool):
    mautrix_types = pytest.importorskip("mautrix.types")
    original = {
        "room_id": ROOM, "event_id": "$child", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "original"},
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.encrypted", "content": {
                "ciphertext": "replacement", "session_id": "sess",
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
            },
        }}},
    }
    decrypted_original = mautrix_types.Event.deserialize({
        **_original("$child", "before"), "origin_server_ts": 1,
    })
    edit_content = {
        "msgtype": "m.text", "body": "* after",
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"},
    }
    if include_new_content:
        edit_content["m.new_content"] = {"msgtype": "m.text", "body": "after"}
    store = _edit_store(json.loads(json.dumps(edit_content)))
    decrypted_edit = mautrix_types.Event.deserialize({
        "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
        "type": "m.room.message", "origin_server_ts": 2,
        "content": edit_content,
    })
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = [(decrypted_original, None), (decrypted_edit, None)]
        state = await effective_event(SimpleNamespace(crypto=SimpleNamespace(crypto_store=store)), original)

    assert isinstance(decrypted_edit.content.serialize().get("m.new_content"), dict)
    assert state == (
        MatrixEffectiveEvent({"msgtype": "m.text", "body": "after"}, original["content"], edited=True, replacement_id="$edit",
                             event_type="m.room.message")
        if include_new_content else
        MatrixEffectiveEvent({"msgtype": "m.text", "body": "before"}, original["content"], event_type="m.room.message")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("relation_kind", ["matching", "outer-only", "wrong-target", "wrong-type"])
async def test_encrypted_replacement_requires_consistent_plaintext_relation(relation_kind: str):
    mautrix_types = pytest.importorskip("mautrix.types")
    original = {
        **_original("$original", "before"), "type": "m.room.encrypted",
        "origin_server_ts": 1,
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "original",
            "session_id": "session", "sender_key": "key", "device_id": "device",
        },
    }
    outer_relation = {"rel_type": "m.replace", "event_id": "$original"}
    replacement = {
        **original, "event_id": "$edit",
        "content": {**original["content"], "ciphertext": "replacement", "m.relates_to": outer_relation},
    }
    original["unsigned"] = {"m.relations": {"m.replace": replacement}}
    plaintext_content = {
        "msgtype": "m.text", "body": "* after",
        "m.new_content": {"msgtype": "m.text", "body": "after"},
    }
    relations = {
        "matching": outer_relation,
        "wrong-target": {"rel_type": "m.replace", "event_id": "$other"},
        "wrong-type": {"rel_type": "m.thread", "event_id": "$original"},
    }
    if relation_kind != "outer-only":
        plaintext_content["m.relates_to"] = relations[relation_kind]
    store = _edit_store(json.loads(json.dumps(plaintext_content)))
    clear_content = json.loads(json.dumps(plaintext_content))
    if relation_kind == "outer-only":
        clear_content["m.relates_to"] = outer_relation

    async def decrypt(event):
        content = clear_content if event.event_id == "$edit" else {"msgtype": "m.text", "body": "before"}
        return mautrix_types.Event.deserialize({
            **_original(str(event.event_id), ""), "origin_server_ts": 1, "content": content,
        })

    client = SimpleNamespace(crypto=SimpleNamespace(decrypt_megolm_event=decrypt, crypto_store=store))
    state = await effective_event(client, original, cache=MatrixEventContextCache(), room_id=ROOM)

    assert state == (
        MatrixEffectiveEvent({"msgtype": "m.text", "body": "after"}, original["content"],
                             edited=True, replacement_id="$edit", event_type="m.room.message")
        if relation_kind in {"matching", "outer-only"} else
        MatrixEffectiveEvent({"msgtype": "m.text", "body": "before"}, original["content"], event_type="m.room.message")
    )


@pytest.mark.asyncio
async def test_encrypted_edit_requires_actual_new_content_in_typed_payload():
    original = {
        **_original("$child", "before"), "type": "m.room.encrypted",
        "unsigned": {"m.relations": {"m.replace": {
            **_original("$edit", "* after"), "type": "m.room.encrypted",
            "content": {"ciphertext": "replacement", "session_id": "sess",
                        "m.relates_to": {"rel_type": "m.replace", "event_id": "$child"}},
        }}},
    }
    typed_edit = SimpleNamespace(content=SimpleNamespace(
        unrecognized_={},
        serialize=lambda: {
            "msgtype": "m.text", "body": "* after",
            "m.new_content": {"msgtype": "m.text", "body": "after"},
        },
    ))
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = [
            (SimpleNamespace(content={"msgtype": "m.text", "body": "before"}), None),
            (typed_edit, None),
        ]
        store = _edit_store({"msgtype": "m.text", "body": "* after"})
        state = await effective_event(SimpleNamespace(crypto=SimpleNamespace(crypto_store=store)), original)

    assert state == MatrixEffectiveEvent(
        {"msgtype": "m.text", "body": "before"}, original["content"],
    )


@pytest.mark.asyncio
async def test_encrypted_edit_is_visible_in_room_catch_up():
    raw = {
        "room_id": ROOM, "event_id": "$original", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "original"},
        "unsigned": {"m.relations": {"m.replace": {
            "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
            "type": "m.room.encrypted", "content": {
                "ciphertext": "replacement", "session_id": "sess",
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

    crypto = SimpleNamespace(
        decrypt_megolm_event=AsyncMock(side_effect=decrypted),
        crypto_store=_edit_store({
            "msgtype": "m.text", "body": "* after",
            "m.new_content": {"msgtype": "m.text", "body": "after"},
        }),
    )
    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=crypto)
    mautrix = SimpleNamespace(types=SimpleNamespace(EncryptedEvent=SimpleNamespace(deserialize=lambda event: event), JSON=lambda event: event))
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix.types}):
        entries = await fetch_room_entries(client, MatrixEventContextCache(), ROOM, "$current", limit=1)

    assert entries == [MatrixEventContext(SENDER, "after", event_id="$original", replacement_id="$edit")]
    assert [call.args[0] for call in crypto.decrypt_megolm_event.await_args_list] == [
        raw, raw["unsigned"]["m.relations"]["m.replace"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["original", "edit"])
async def test_reply_retries_catch_up_event_after_decryption_keys_arrive(missing: str):
    raw = {
        "room_id": ROOM, "event_id": "$target", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "original"},
    }
    edit = {
        "room_id": ROOM, "event_id": "$edit", "sender": SENDER,
        "type": "m.room.encrypted", "content": {
            "ciphertext": "replacement", "session_id": "sess",
            "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
        },
    }
    if missing == "edit":
        raw["unsigned"] = {"m.relations": {"m.replace": edit}}
    decrypted_original = SimpleNamespace(content={"msgtype": "m.text", "body": "before"})
    decrypted_edit = SimpleNamespace(content={
        "msgtype": "m.text", "body": "* after",
        "m.new_content": {"msgtype": "m.text", "body": "after"},
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
    })
    missing_key = (None, {"event_id": "$target" if missing == "original" else "$edit",
                          "error": "missing decryption keys"})
    decryptions = (
        [missing_key, (decrypted_original, None)] if missing == "original" else
        [(decrypted_original, None), missing_key,
         (decrypted_original, None), (decrypted_edit, None)]
    )

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [raw]}
        if "/event/" in path:
            return raw
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        crypto=SimpleNamespace(crypto_store=_edit_store({
            "msgtype": "m.text", "body": "* after",
            "m.new_content": {"msgtype": "m.text", "body": "after"},
        })),
    )
    cache = MatrixEventContextCache()
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = decryptions
        earlier = await fetch_room_entries(client, cache, ROOM, "$current", limit=1)
        reply = await cache.resolve(client, ROOM, "$target")

    expected_earlier = (
        "[encrypted message could not be decrypted]" if missing == "original" else "before"
    )
    assert earlier == [MatrixEventContext(SENDER, expected_earlier,
                                          state_error="missing decryption keys", event_id="$target")]
    assert reply == MatrixEventContext(
        SENDER, "before" if missing == "original" else "after", event_id="$target",
        replacement_id="$edit" if missing == "edit" else None,
    )
    assert [call.args[1] for call in client.api.request.await_args_list if "/event/" in call.args[1]] == [
        f"/_matrix/client/v3/rooms/{quote(ROOM, safe='')}/event/%24target"
    ]
    assert decrypt.await_count == len(decryptions)


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

    assert room == [MatrixEventContext(SENDER, "after room", event_id="$room", replacement_id="$edit")]
    assert thread == [
        MatrixEventContext(SENDER, "root after", event_id="$root", replacement_id="$edit"),
        MatrixEventContext(SENDER, "after thread", event_id="$child", replacement_id="$edit"),
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

    assert entries == [MatrixEventContext(SENDER, "[redacted]", redacted=True, event_id="$child")]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope, target", [
    ("room", "$room"), ("thread", "$root"), ("thread", "$child"),
])
async def test_catch_up_keeps_redaction_received_during_reaction_lookup(scope: str, target: str):
    started = asyncio.Event()
    release = asyncio.Event()
    cache = MatrixEventContextCache()
    event_ids = ["$room"] if scope == "room" else ["$root", "$child"]
    events = {
        event_id: _original(
            event_id, "withdrawn secret" if event_id == target else "retained text",
            root="$root" if event_id == "$child" else None,
        )
        for event_id in event_ids
    }

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path or "/m.thread" in path:
            return {"start": "boundary", "chunk": [events[event_ids[-1]]]}
        if "/event/" in path:
            return events["$root"]
        if path.endswith(f"/relations/{quote(target, safe='')}/m.annotation"):
            started.set()
            await release.wait()
            return {"chunk": [{
                "type": "m.reaction", "event_id": "$reaction", "sender": SENDER,
                "content": {"m.relates_to": {
                    "rel_type": "m.annotation", "event_id": target, "key": "👍",
                }},
            }], "next_batch": "more-reactions"}
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None)
    catch_up = (
        fetch_room_entries(client, cache, ROOM, "$current", limit=2) if scope == "room"
        else fetch_thread_entries(client, cache, ROOM, "$root", limit=2, before_event_id="$current")
    )
    pending = asyncio.create_task(catch_up)
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        cache.redact(ROOM, target)
    finally:
        release.set()
        entries = await pending

    assert entries == [
        MatrixEventContext(SENDER, "", redacted=True, event_id=event_id) if event_id == target
        else MatrixEventContext(SENDER, "retained text", event_id=event_id)
        for event_id in event_ids
    ]


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

    assert (edited, redacted) == (MatrixEventContext(SENDER, "after", event_id="$child", replacement_id="$edit"), None)
    client.get_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_refetch_does_not_restore_text_redacted_during_decryption():
    started = asyncio.Event()
    release = asyncio.Event()
    cache = MatrixEventContextCache()
    cache.store(ROOM, "$image", MatrixEventContext(SENDER, "[Image]", is_image=True))

    async def decrypt(_client, _raw):
        started.set()
        await release.wait()
        return None, {"event_id": "$image", "error": "missing decryption keys"}

    encrypted = {
        "room_id": ROOM, "event_id": "$image", "sender": SENDER,
        "type": "m.room.encrypted", "content": {"ciphertext": "encrypted"},
    }
    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(return_value=encrypted)), crypto=None)
    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        pending = asyncio.create_task(cache.resolve(client, ROOM, "$image", AsyncMock(return_value=None)))
        await started.wait()
        cache.redact(ROOM, "$image")
        release.set()
        result = await pending

    assert result is None
    assert cache.history_entry(ROOM, "$image") == MatrixEventContext(SENDER, "", redacted=True, event_id="$image")


@pytest.mark.asyncio
async def test_thread_root_fetch_failure_uses_redaction_received_during_request():
    started = asyncio.Event()
    release = asyncio.Event()
    cache = MatrixEventContextCache()
    cache.store(ROOM, "$root", MatrixEventContext(SENDER, "Before redaction"))

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"events_before": []}
        if "/event/" in path:
            started.set()
            await release.wait()
            raise RuntimeError("root fetch failed")
        raise AssertionError(path)

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    pending = asyncio.create_task(fetch_thread_entries(
        client, cache, ROOM, "$root", limit=1, before_event_id="$current",
    ))
    await started.wait()
    cache.redact(ROOM, "$root")
    release.set()

    assert await pending == [MatrixEventContext(SENDER, "", redacted=True, event_id="$root")]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event", "room", "thread"])
@pytest.mark.parametrize("replacement", [False, True])
async def test_bounded_read_rechecks_redaction_after_reactions(kind: str, replacement: bool):
    from tests.gateway.test_matrix import _make_adapter

    started, release = asyncio.Event(), asyncio.Event()
    raw = _original("$target", "before" if replacement else "withdrawn secret",
                    root="$root" if kind == "thread" else None)
    if replacement:
        raw = _edited(raw, "withdrawn secret")

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return raw
        if "/messages" in path or "/m.thread" in path:
            return {"chunk": [raw]}
        if "/m.annotation" in path:
            started.set()
            await release.wait()
            return {"chunk": [{
                "type": "m.reaction", "event_id": "$reaction", "sender": SENDER,
                "content": {"m.relates_to": {
                    "rel_type": "m.annotation", "event_id": "$target", "key": "👍",
                }},
            }], "next_batch": "more"}
        raise AssertionError(path)

    adapter = _make_adapter()
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
    )
    adapter._joined_rooms = {ROOM}
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    target = "$root" if kind == "thread" else "$target"
    pending = asyncio.create_task(adapter.read_matrix_context(kind, ROOM, target, 2, requester=SENDER))
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$edit" if replacement else "$target"))
    finally:
        release.set()
        result = await pending

    expected = {
        "event_id": "$target", "sender": SENDER, "body": "[event content unavailable]" if replacement else "[redacted]",
        "msgtype": None,
        "thread_id": "$root" if kind == "thread" else None,
        "timestamp": None, "sender_authorized": True,
    }
    if not replacement:
        expected["redacted"] = True
    assert result == {"events": [expected], "errors": [
        {"event_id": "$target", "error": "replacement was redacted"},
    ] if replacement else [], "skipped": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["reply", "reply-inline", "room", "thread"])
@pytest.mark.parametrize("replacement", [False, True])
async def test_formatting_rechecks_redaction_after_last_display_name_lookup(scope: str, replacement: bool):
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    started, release = asyncio.Event(), asyncio.Event()
    target = "$root" if scope == "thread" else "$target"
    raws = [_original(target, "withdrawn secret"), _original("$later", "retained text")]
    if replacement:
        raws[0] = _edited(_original(target, "before"), "withdrawn secret")

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/event/" in path:
            return raws[0]
        if "/messages" in path or "/m.thread" in path:
            return {"chunk": list(reversed(raws))}
        if "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    calls = 0

    async def display_name(_room, _sender):
        nonlocal calls
        calls += 1
        if calls == (2 if scope == "room" else 1):
            started.set()
            await release.wait()
        return "Alice"

    adapter = _make_adapter()
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    adapter._get_display_name = display_name
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    if scope == "reply-inline":
        await adapter._event_context_cache.resolve(adapter._client, ROOM, target)
    if scope in {"reply", "reply-inline"}:
        body = f"> <{SENDER}> withdrawn secret\n\nquestion" if scope == "reply-inline" else "question"
        context = adapter._extract_reply_context(
            ROOM, body, {"body": body}, {"m.in_reply_to": {"event_id": target}}, sender=SENDER,
            chat_type="group",
        )
    elif scope == "room":
        context = _rendered(adapter.fetch_room_history(ROOM, "$current"))
    else:
        context = _rendered(adapter.fetch_thread_history(ROOM, "$root", before_event_id="$current"))
    pending = asyncio.create_task(context)
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$edit" if replacement else target))
    finally:
        release.set()
        result = await pending

    withdrawn = ("[event content unavailable]\n[Matrix event state unavailable: replacement was redacted.]"
                 if replacement else "[redacted]")
    expected = {
        "reply": MatrixReplyContext("question", target, None, SENDER, "Alice", False, True),
        "reply-inline": MatrixReplyContext("question", target, None, SENDER, "Alice", False, replacement),
        "room": f"[Recent room messages]\n[Alice] {withdrawn}\n[Alice] retained text",
        "thread": f"[Earlier messages in this thread]\n[Alice] {withdrawn}",
    }
    assert result == expected[scope]


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_live_typed_edit_resolves_actual_payload_before_caching(encrypted: bool, valid: bool):
    from tests.gateway.test_matrix import _make_adapter

    mautrix_types = pytest.importorskip("mautrix.types")
    raw = _edited(_original("$target", "before"), "after")
    replacement = raw["unsigned"]["m.relations"]["m.replace"]
    content = json.loads(json.dumps(replacement["content"]))
    if not valid:
        content.pop("m.new_content")
    replacement["content"] = json.loads(json.dumps(content))
    typed = mautrix_types.Event.deserialize({**replacement, "content": json.loads(json.dumps(content)),
                                           "origin_server_ts": 1})
    if encrypted:
        typed["mautrix"] = {"was_encrypted": True}
        raw = {**raw, "type": "m.room.encrypted", "content": {"ciphertext": "original"}}
        raw["unsigned"]["m.relations"]["m.replace"] = {
            **replacement, "type": "m.room.encrypted",
            "content": {"ciphertext": "replacement", "session_id": "sess", "m.relates_to": {
                "rel_type": "m.replace", "event_id": "$target",
            }},
        }
    adapter = _make_adapter()
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._event_context_cache.store(ROOM, "$target", MatrixEventContext(SENDER, "before"))
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value=raw)),
        crypto=SimpleNamespace(crypto_store=_edit_store(content)),
    )
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = [(SimpleNamespace(content={"msgtype": "m.text", "body": "before"}), None),
                               (typed, None)]
        await adapter._on_room_message(typed)
        result = await adapter._event_context_cache.resolve(adapter._client, ROOM, "$target")

    assert result == MatrixEventContext(
        SENDER, "after" if valid else "before", event_id="$target",
        replacement_id="$edit" if valid else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["reply", "room", "thread", "live"])
@pytest.mark.parametrize("remaining_edit", [False, True])
async def test_redacting_effective_replacement_invalidates_original_and_recovers_content(source: str, remaining_edit: bool):
    from tests.gateway.test_matrix import _make_adapter

    target = "$root" if source == "thread" else "$target"
    raw = _edited(_original(target, "before"), "withdrawn secret")

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [raw]}
        if "/event/" in path:
            return raw
        if "/m.thread" in path or "/m.annotation" in path:
            return {"chunk": []}
        raise AssertionError(path)

    adapter = _make_adapter()
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    cache = adapter._event_context_cache
    if source == "room":
        await fetch_room_entries(adapter._client, cache, ROOM, "$current", limit=2)
    elif source == "thread":
        await fetch_thread_entries(adapter._client, cache, ROOM, target, limit=2, before_event_id="$current")
    elif source == "live":
        cache.store(ROOM, target, MatrixEventContext(SENDER, "before"))
        replacement = raw["unsigned"]["m.relations"]["m.replace"]
        await adapter._on_room_message(SimpleNamespace(**replacement))
    before = await cache.resolve(adapter._client, ROOM, target)
    if remaining_edit:
        earlier = _edited(_original(target, "before"), "remaining valid edit")
        earlier["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$earlier"
        raw.clear()
        raw.update(earlier)
    else:
        raw.pop("unsigned")
    await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$edit"))
    after = await cache.resolve(adapter._client, ROOM, target)

    assert (before, after) == (
        MatrixEventContext(SENDER, "withdrawn secret", event_id=target, replacement_id="$edit"),
        MatrixEventContext(SENDER, "remaining valid edit" if remaining_edit else "before", event_id=target,
                           replacement_id="$earlier" if remaining_edit else None),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event", "room", "thread"])
@pytest.mark.parametrize("recovery", ["earlier", "stale", "failure", "keys", "inspection"])
async def test_replacement_redaction_during_read_requires_current_bundle(kind: str, recovery: str):
    started, release = asyncio.Event(), asyncio.Event()
    raw = _edited(_original("$target", "initial draft", root="$root" if kind == "thread" else None), "withdrawn secret")
    earlier = _edited(_original("$target", "initial draft", root="$root" if kind == "thread" else None), "surviving edit")
    earlier["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$earlier"
    if recovery == "inspection":
        raw["type"] = "m.room.encrypted"
        replacement = raw["unsigned"]["m.relations"]["m.replace"]
        replacement["type"] = "m.room.encrypted"
        replacement["content"].update(session_id="latest", ciphertext="fake")
    if recovery == "keys":
        earlier["type"] = "m.room.encrypted"
        earlier["unsigned"]["m.relations"]["m.replace"]["type"] = "m.room.encrypted"
    invalidated = False

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            if kind == "thread" and path.endswith(quote("$root", safe="")):
                return _original("$root", "thread root")
            if invalidated and recovery == "failure":
                raise RuntimeError("recovery unavailable")
            if invalidated and recovery == "stale":
                replacement = raw["unsigned"]["m.relations"]["m.replace"]
                return {**raw, "unsigned": {"m.relations": {"m.replace": {
                    **replacement, "unsigned": {"redacted_because": {"event_id": "$redaction"}},
                }}}}
            return earlier if invalidated and recovery in {"earlier", "keys", "inspection"} else raw
        if "/messages" in path or "/m.thread" in path:
            return {"chunk": [raw]}
        if path.endswith(f"/relations/{quote('$target', safe='')}/m.annotation"):
            started.set()
            await release.wait()
        return {"chunk": []}

    async def missing_session(_room, _session):
        started.set()
        await release.wait()
        raise RuntimeError("decryption keys unavailable")

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
        crypto=SimpleNamespace(crypto_store=SimpleNamespace(get_group_session=missing_session)),
    )
    adapter = _adapter(client)
    async def decrypt(_client, event):
        if event["event_id"] == "$earlier":
            return None, {"event_id": "$earlier", "error": "missing decryption keys"}
        return SimpleNamespace(content=event["content"]), None

    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        pending = asyncio.create_task(read_matrix_context(
            adapter, kind, ROOM, "$root" if kind == "thread" else "$target", 2, requester=SENDER,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            invalidated = True
            adapter._event_context_cache.redact(ROOM, "$edit")
        finally:
            release.set()
            result = await pending

    recovered = recovery in {"earlier", "inspection"}
    target = {
        "event_id": "$target", "sender": SENDER,
        "body": "surviving edit" if recovered else "[event content unavailable]",
        "msgtype": "m.text" if recovered else None,
        "thread_id": "$root" if kind == "thread" else None,
        "timestamp": None, "sender_authorized": True,
    }
    if recovered:
        target["edited"] = True
    events = [target]
    if kind == "thread":
        events.insert(0, {
            "event_id": "$root", "sender": SENDER, "body": "thread root", "msgtype": "m.text",
            "thread_id": None, "timestamp": None, "sender_authorized": True,
        })
    assert result == {
        "events": events,
        "errors": ([] if recovered else [{"event_id": "$earlier", "error": "missing decryption keys"}]
                   if recovery == "keys" else [{"event_id": "$target", "error": "replacement was redacted"}]),
        "skipped": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("quote_kind", ["plain", "html"])
@pytest.mark.parametrize("failure", ["request", "keys", "invalidate"])
async def test_inline_reply_after_replacement_redaction_validates_and_retries(quote_kind: str, failure: str):
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    adapter = _make_adapter()
    cache = adapter._event_context_cache
    cache.store(ROOM, "$target", MatrixEventContext(SENDER, "withdrawn secret", replacement_id="$edit"))
    cache.redact(ROOM, "$edit")
    if failure == "invalidate":
        cache.invalidate(ROOM, "$target")
    earlier = _edited(_original("$target", "initial draft"), "surviving edit")
    earlier["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$earlier"
    if failure == "keys":
        earlier["type"] = "m.room.encrypted"
        replacement = earlier["unsigned"]["m.relations"]["m.replace"]
        replacement["type"] = "m.room.encrypted"
        replacement["content"].update(session_id="earlier", ciphertext="fake")
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=[
            RuntimeError("recovery unavailable") if failure != "keys" else earlier, earlier,
        ])),
        crypto=SimpleNamespace(crypto_store=_edit_store({"m.new_content": {
            "msgtype": "m.text", "body": "surviving edit",
        }})),
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    body = f"> <{SENDER}> withdrawn secret\n\nquestion" if quote_kind == "plain" else "question"
    formatted_body = "<mx-reply><blockquote>withdrawn secret</blockquote></mx-reply>question" if quote_kind == "html" else None
    keys_available = False

    async def decrypt(_client, event):
        if event["event_id"] == "$earlier" and not keys_available:
            return None, {"event_id": "$earlier", "error": "missing decryption keys"}
        return SimpleNamespace(content=event["content"]), None

    replies = []
    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        for _ in range(3):
            replies.append(await adapter._extract_reply_context(
                ROOM, body, {"body": body, **({"format": "org.matrix.custom.html",
                                               "formatted_body": formatted_body} if formatted_body else {})},
                {"m.in_reply_to": {"event_id": "$target"}}, sender=SENDER, chat_type="group",
            ))
            if not keys_available:
                assert cache.history_entry(ROOM, "$target") == MatrixEventContext(
                    SENDER, "[event content unavailable]", event_id="$target",
                    state_error="replacement was redacted", replacement_id="$edit",
                )
            keys_available = True

    assert replies == [
        MatrixReplyContext("question", "$target", None, SENDER, "Alice", False, True),
        MatrixReplyContext("question", "$target", "surviving edit", SENDER, "Alice", False, True),
        MatrixReplyContext("question", "$target", "surviving edit", SENDER, "Alice", False, True),
    ]
    assert cache.history_entry(ROOM, "$target") == MatrixEventContext(
        SENDER, "surviving edit", event_id="$target", replacement_id="$earlier",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["event", "room", "thread"])
@pytest.mark.parametrize("boundary", ["read", "format"])
async def test_reaction_redaction_filters_its_own_event_id(scope: str, boundary: str):

    started, release = asyncio.Event(), asyncio.Event()
    target = "$root" if scope == "thread" else "$target"
    raw = _original(target, "retained text")

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/event/" in path:
            return raw
        if "/messages" in path:
            return {"chunk": [raw]}
        if "/m.thread" in path:
            return {"chunk": []}
        if "/m.annotation" in path:
            if boundary == "read":
                started.set()
                await release.wait()
            return {"chunk": [{
                "type": "m.reaction", "event_id": "$reaction", "sender": SENDER,
                "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": "withdrawn reaction"}},
            }]}
        raise AssertionError(path)

    async def display_name(_room, _sender):
        started.set()
        await release.wait()
        return "Alice"

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    adapter = _adapter(client)
    adapter._get_display_name = display_name
    cache = adapter._event_context_cache

    async def read_and_format():
        if scope == "event":
            return await read_matrix_context(adapter, scope, ROOM, target, 1, requester=SENDER)
        entries = (await fetch_room_entries(client, cache, ROOM, "$current", limit=1) if scope == "room"
                   else await fetch_thread_entries(client, cache, ROOM, target, limit=1, before_event_id="$current"))
        if boundary == "format":
            return await _rendered(MatrixHistoryContext.prepare(adapter, ROOM, entries, "History"))
        return entries

    if scope == "event" and boundary == "format":
        cache.redact(ROOM, "$reaction")
        result = await read_and_format()
    else:
        pending = asyncio.create_task(read_and_format())
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            cache.redact(ROOM, "$reaction")
        finally:
            release.set()
            result = await pending

    if scope == "event":
        assert result == {"events": [{
            "event_id": target, "sender": SENDER, "body": "retained text", "msgtype": "m.text",
            "thread_id": None, "timestamp": None, "sender_authorized": True,
        }], "errors": [], "skipped": 0}
    elif boundary == "format":
        assert result == "[History]\n[Alice] retained text"
    else:
        assert result == [MatrixEventContext(SENDER, "retained text", event_id=target)]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event", "room", "thread"])
async def test_bounded_read_observed_original_redaction_invalidates_shared_reply(kind: str):
    target = "$root" if kind == "thread" else "$target"
    raw = _edited(_original(target, "withdrawn secret"), "withdrawn edit")
    raw["unsigned"]["redacted_because"] = {"event_id": "$redaction"}

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return raw
        return {"chunk": [raw] if "/messages" in path else []}

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
    )
    adapter = _adapter(client)
    cache = adapter._event_context_cache
    cache.store(ROOM, target, MatrixEventContext(SENDER, "withdrawn edit", replacement_id="$edit"))
    result = await read_matrix_context(adapter, kind, ROOM, target, 1, requester=SENDER)
    reply = await cache.resolve(client, ROOM, target)

    assert (result, reply) == ({"events": [{
        "event_id": target, "sender": SENDER, "body": "[redacted]", "msgtype": None,
        "thread_id": None, "timestamp": None, "sender_authorized": True, "redacted": True,
    }], "errors": [], "skipped": 0}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("redact_replacement", [False, True])
async def test_typed_invalidation_retains_dependency_until_successful_retry(redact_replacement: bool):
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    mautrix_types = pytest.importorskip("mautrix.types")
    adapter = _make_adapter()
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    cache = adapter._event_context_cache
    cache.store(ROOM, "$target", MatrixEventContext(SENDER, "withdrawn secret", replacement_id="$edit"))
    new = _edited(_original("$target", "initial draft"), "surviving edit")
    typed_raw = new["unsigned"]["m.relations"]["m.replace"]
    typed_raw["event_id"] = "$next"
    await adapter._on_room_message(mautrix_types.Event.deserialize({**typed_raw, "origin_server_ts": 1}))
    if redact_replacement:
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$edit"))
    invalidated = cache.history_entry(ROOM, "$target")
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=[
        RuntimeError("recovery unavailable"), new,
    ])))
    replies = [await adapter._extract_reply_context(
        ROOM, f"> <{SENDER}> withdrawn secret\n\nquestion", {"body": f"> <{SENDER}> withdrawn secret\n\nquestion"},
        {"m.in_reply_to": {"event_id": "$target"}}, sender=SENDER, chat_type="group",
    ) for _ in range(3)]

    assert (invalidated, replies, cache.history_entry(ROOM, "$target")) == (
        MatrixEventContext(SENDER, "[event content unavailable]", event_id="$target", replacement_id="$edit",
                           state_error="replacement was redacted" if redact_replacement else "event content changed"),
        [MatrixReplyContext("question", "$target", None, SENDER, "Alice", False, True)]
        + [MatrixReplyContext("question", "$target", "surviving edit", SENDER, "Alice", False, True)] * 2,
        MatrixEventContext(SENDER, "surviving edit", event_id="$target", replacement_id="$next"),
    )
    assert adapter._client.api.request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["event", "room-read", "thread-read", "reply", "reply-inline", "room", "thread", "room-decrypt", "thread-decrypt", "event-fetch", "room-fetch", "thread-fetch"])
@pytest.mark.parametrize("typed", [False, True])
async def test_context_refreshes_edit_at_last_asynchronous_boundary(scope: str, typed: bool):
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    mautrix_types = pytest.importorskip("mautrix.types")
    started, release = asyncio.Event(), asyncio.Event()
    target = "$root" if scope in {"thread", "thread-read", "thread-decrypt", "thread-fetch"} else "$target"
    original = _original(target, "before", root="$thread" if scope == "thread-read" else None)
    raw = original
    updated = _edited(original, "after")
    if scope.endswith("-decrypt"):
        original["type"] = updated["type"] = "m.room.encrypted"
        updated["unsigned"]["m.relations"]["m.replace"]["type"] = "m.room.encrypted"
        updated["unsigned"]["m.relations"]["m.replace"]["content"].update(session_id="sess", ciphertext="fake")

    async def decrypt(_client, event):
        if event is original:
            started.set()
            await release.wait()
        return SimpleNamespace(content=event["content"]), None

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            response = raw
            if scope in {"event-fetch", "thread-fetch"} and response is original:
                started.set()
                await release.wait()
            return response
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            response = {"chunk": [raw]}
            if scope == "room-fetch":
                started.set()
                await release.wait()
            return response
        if "/m.thread" in path:
            return {"chunk": []}
        if "/m.annotation" in path:
            if scope in {"event", "room-read", "thread-read"}:
                started.set()
                await release.wait()
            return {"chunk": []}
        raise AssertionError(path)

    async def display_name(_room, _sender):
        started.set()
        await release.wait()
        return "Alice"

    adapter = _make_adapter()
    adapter._joined_rooms = {ROOM}
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
        crypto=SimpleNamespace(crypto_store=_edit_store({"m.new_content": {"msgtype": "m.text", "body": "after"}})),
    )
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = display_name
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    if scope == "reply-inline":
        await adapter._event_context_cache.resolve(adapter._client, ROOM, target)
    if scope.startswith("reply"):
        body = f"> <{SENDER}> before\n\nquestion" if scope == "reply-inline" else "question"
        context = adapter._extract_reply_context(
            ROOM, body, {"body": body}, {"m.in_reply_to": {"event_id": target}}, sender=SENDER,
            chat_type="group",
        )
    elif scope in {"room", "room-decrypt", "room-fetch"}:
        context = _rendered(adapter.fetch_room_history(ROOM, "$current"))
    elif scope in {"thread", "thread-decrypt", "thread-fetch"}:
        context = _rendered(adapter.fetch_thread_history(ROOM, target, before_event_id="$current"))
    else:
        kind = {"event": "event", "event-fetch": "event", "room-read": "room", "thread-read": "thread"}[scope]
        context = adapter.read_matrix_context(kind, ROOM, target, 1, requester=SENDER)
    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        pending = asyncio.create_task(context)
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            raw = updated
            edit = {**updated["unsigned"]["m.relations"]["m.replace"], "type": "m.room.message"}
            event = mautrix_types.Event.deserialize({**edit, "origin_server_ts": 1}) if typed else SimpleNamespace(**edit)
            await adapter._on_room_message(event)
        finally:
            release.set()
            result = await pending

    if scope.startswith("reply"):
        expected = MatrixReplyContext("question", target, "after", SENDER, "Alice", False, True)
    elif scope in {"room", "thread", "room-decrypt", "thread-decrypt", "room-fetch", "thread-fetch"}:
        heading = "Recent room messages" if scope.startswith("room") else "Earlier messages in this thread"
        expected = f"[{heading}]\n[Alice] after"
    else:
        expected = {"events": [{
            "event_id": target, "sender": SENDER, "body": "after", "msgtype": "m.text",
            "thread_id": "$thread" if scope == "thread-read" else None,
            "timestamp": None, "sender_authorized": True, "edited": True,
        }], "errors": [], "skipped": 0}
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "thread"])
@pytest.mark.parametrize("retains_relation", [False, True])
async def test_catch_up_observed_replacement_redaction_invalidates_before_reply_fast_path(scope: str, retains_relation: bool):
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    cache = MatrixEventContextCache()
    cache.store(ROOM, "$target", MatrixEventContext(SENDER, "withdrawn secret", replacement_id="$edit"))
    redacted = {**_original("$edit", ""), "unsigned": {"redacted_because": {"event_id": "$redaction"}}}
    if retains_relation:
        redacted["content"]["m.relates_to"] = {"rel_type": "m.replace", "event_id": "$target"}
    recovered = _edited(_original("$target", "initial draft"), "surviving edit")
    recovered["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$earlier"

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [redacted]}
        if path.endswith(quote("$target", safe="")):
            return recovered
        if "/event/" in path:
            return redacted
        return {"chunk": []}

    adapter = _make_adapter()
    adapter._event_context_cache = cache
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    if scope == "room":
        await fetch_room_entries(adapter._client, cache, ROOM, "$current", limit=1)
    else:
        await fetch_thread_entries(adapter._client, cache, ROOM, "$edit", limit=1, before_event_id="$current")
    invalidated = cache.history_entry(ROOM, "$target")
    replies = [await adapter._extract_reply_context(
        ROOM, f"> <{SENDER}> withdrawn secret\n\nquestion", {"body": f"> <{SENDER}> withdrawn secret\n\nquestion"},
        {"m.in_reply_to": {"event_id": "$target"}}, sender=SENDER, chat_type="group",
    ) for _ in range(2)]

    assert (invalidated, replies) == (
        MatrixEventContext(SENDER, "[event content unavailable]", event_id="$target",
                           state_error="replacement was redacted", replacement_id="$edit"),
        [MatrixReplyContext("question", "$target", "surviving edit", SENDER, "Alice", False, True)] * 2,
    )
    assert sum(call.args[1].endswith(quote("$target", safe=""))
               for call in adapter._client.api.request.await_args_list) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event", "room", "thread"])
@pytest.mark.parametrize("error", ["missing decryption keys", "decryption failed"])
async def test_original_redaction_wins_over_failed_decryption_during_read(kind: str, error: str):
    started, release = asyncio.Event(), asyncio.Event()
    raw = {**_original("$target", "withdrawn secret", root="$root" if kind == "thread" else None),
           "type": "m.room.encrypted"}

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return raw
        return {"chunk": [raw]}

    async def decrypt(_client, _raw):
        started.set()
        await release.wait()
        return None, {"event_id": "$target", "error": error}

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
    )
    adapter = _adapter(client)
    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        pending = asyncio.create_task(read_matrix_context(
            adapter, kind, ROOM, "$target", 1, requester=SENDER,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            adapter._event_context_cache.redact(ROOM, "$target")
        finally:
            release.set()
            result = await pending

    assert result == {"events": [{
        "event_id": "$target", "sender": SENDER, "body": "[redacted]", "msgtype": None,
        "thread_id": "$root" if kind == "thread" else None,
        "timestamp": None, "sender_authorized": True, "redacted": True,
    }], "errors": [], "skipped": 0}


@pytest.mark.asyncio
async def test_validated_bounded_read_updates_reply_and_existing_formatting_snapshot():
    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    adapter = _make_adapter()
    adapter._joined_rooms = {ROOM}
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    before = adapter._event_context_cache.store(ROOM, "$target", MatrixEventContext(SENDER, "before"))
    raw = _edited(_original("$target", "before"), "after")

    async def request(_method, path, **_kwargs):
        return raw if "/event/" in path else {"chunk": []}

    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    result = await adapter.read_matrix_context("event", ROOM, "$target", 1, requester=SENDER)
    reply = await adapter._extract_reply_context(
        ROOM, f"> <{SENDER}> before\n\nquestion", {"body": f"> <{SENDER}> before\n\nquestion"},
        {"m.in_reply_to": {"event_id": "$target"}}, sender=SENDER, chat_type="group",
    )
    formatted = await _rendered(MatrixHistoryContext.prepare(adapter, ROOM, [before], "History"))

    assert (result, reply, formatted) == (
        {"events": [{
            "event_id": "$target", "sender": SENDER, "body": "after", "msgtype": "m.text",
            "thread_id": None, "timestamp": None, "sender_authorized": True, "edited": True,
        }], "errors": [], "skipped": 0},
        MatrixReplyContext("question", "$target", "after", SENDER, "Alice", False, True),
        "[History]\n[Alice] after",
    )
    assert adapter._client.api.request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "thread"])
async def test_failed_catch_up_recovery_keeps_invalidated_replacement(scope: str):
    cache = MatrixEventContextCache()
    target = "$root" if scope == "thread" else "$target"
    cache.store(ROOM, target, MatrixEventContext(SENDER, "withdrawn edit", replacement_id="$latest"))
    cache.redact(ROOM, "$latest")
    invalidated = cache.history_entry(ROOM, target)
    raw = _edited(_original(target, "original draft"), "surviving older edit")
    raw["type"] = "m.room.encrypted"
    replacement = raw["unsigned"]["m.relations"]["m.replace"]
    replacement.update(event_id="$earlier", type="m.room.encrypted")
    replacement["content"].update(session_id="earlier", ciphertext="fake")
    keys_available = False

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/event/" in path:
            return raw
        return {"chunk": [raw] if "/messages" in path else []}

    async def decrypt(_client, event):
        if event["event_id"] == "$earlier" and not keys_available:
            return None, {"event_id": "$earlier", "error": "missing decryption keys"}
        return SimpleNamespace(content=event["content"]), None

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        crypto=SimpleNamespace(crypto_store=_edit_store({"m.new_content": {
            "msgtype": "m.text", "body": "surviving older edit",
        }})),
    )
    with patch("plugins.platforms.matrix.effective_event._decrypt", side_effect=decrypt):
        failed = (await fetch_room_entries(client, cache, ROOM, "$current", limit=1) if scope == "room"
                  else await fetch_thread_entries(client, cache, ROOM, target, limit=1, before_event_id="$current"))
        assert (failed, cache.history_entry(ROOM, target)) == ([invalidated], invalidated)
        keys_available = True
        recovered = await cache.resolve(client, ROOM, target)
    assert recovered == MatrixEventContext(SENDER, "surviving older edit", event_id=target, replacement_id="$earlier")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["room", "thread"])
async def test_failed_read_remains_subject_to_redaction_after_later_reaction_await(kind: str):
    started, release = asyncio.Event(), asyncio.Event()
    failed = {**_original("$failed", "unreadable", root="$root" if kind == "thread" else None),
              "type": "m.room.encrypted"}
    later = _original("$later", "readable", root="$root" if kind == "thread" else None)

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return _original("$root", "root")
        if "/m.annotation" in path:
            started.set()
            await release.wait()
            return {"chunk": []}
        return {"chunk": [later, failed]}

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
    )
    adapter = _adapter(client)
    with patch("plugins.platforms.matrix.effective_event._decrypt", new=AsyncMock(
        return_value=(None, {"event_id": "$failed", "error": "missing decryption keys"}),
    )):
        pending = asyncio.create_task(read_matrix_context(adapter, kind, ROOM, "$root", 3, requester=SENDER))
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
            adapter._event_context_cache.redact(ROOM, "$failed")
        finally:
            release.set()
            result = await pending
    root = [{"event_id": "$root", "sender": SENDER, "body": "root", "msgtype": "m.text",
             "thread_id": None, "timestamp": None, "sender_authorized": True}] if kind == "thread" else []
    assert result == {"events": root + [
        {"event_id": "$failed", "sender": SENDER, "body": "[redacted]", "msgtype": None,
         "thread_id": "$root" if kind == "thread" else None, "timestamp": None,
         "sender_authorized": True, "redacted": True},
        {"event_id": "$later", "sender": SENDER, "body": "readable", "msgtype": "m.text",
         "thread_id": "$root" if kind == "thread" else None, "timestamp": None, "sender_authorized": True},
    ], "errors": [], "skipped": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "thread"])
@pytest.mark.parametrize("change", ["unchanged", "url", "encrypted-key", "body", "redaction", "text"])
async def test_catch_up_retains_only_identical_quoted_attachment(scope: str, change: str, tmp_path):
    import copy

    image = tmp_path / "quoted.png"
    image.write_bytes(b"quoted image")
    original = _original("$image", "image.png")
    original["content"].update(
        msgtype="m.image", file={"url": "mxc://example.org/image", "key": {"k": "old"}},
        info={"mimetype": "image/png"},
    )
    raw = copy.deepcopy(original)
    if change == "url":
        raw["content"]["file"]["url"] = "mxc://example.org/replaced"
    elif change == "encrypted-key":
        raw["content"]["file"]["key"]["k"] = "new"
    elif change == "body":
        raw["content"]["body"] = "new caption"
    elif change == "redaction":
        raw["content"] = {}
        raw["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
    elif change == "text":
        raw["content"] = {"msgtype": "m.text", "body": "new text"}
    catch_up = False

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return raw if catch_up else original
        if "/context/" in path:
            return {"start": "boundary"}
        return {"chunk": [raw] if "/messages" in path else []}

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    cache = MatrixEventContextCache()
    loaded = await cache.resolve(client, ROOM, "$image", AsyncMock(return_value=(str(image), "image/png")))
    assert loaded.media_path == str(image)
    catch_up = True
    if scope == "room":
        await fetch_room_entries(client, cache, ROOM, "$current", limit=1)
    else:
        await fetch_thread_entries(client, cache, ROOM, "$image", limit=1, before_event_id="$current")
    current = cache.history_entry(ROOM, "$image")
    assert (current.media_path, current.media_type) == (
        (str(image), "image/png") if change == "unchanged" else (None, None)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["reply", "room", "thread", "event", "room-read", "thread-read"])
@pytest.mark.parametrize("change", ["unchanged", "original", "replacement", "edit"])
async def test_active_context_keeps_effective_state_after_eviction(scope: str, change: str, tmp_path):
    import copy
    import gc
    import weakref

    from tests.gateway.test_matrix import _make_adapter
    from plugins.platforms.matrix.reply_context import MatrixReplyContext

    started, release = asyncio.Event(), asyncio.Event()
    image = tmp_path / "quoted.png"
    image.write_bytes(b"image")
    original = _original("$target", "quoted.png")
    original["content"].update(msgtype="m.image", url="mxc://example.org/image")
    raw = _edited(original, "quoted.png")
    replacement = raw["unsigned"]["m.relations"]["m.replace"]
    replacement["content"]["m.new_content"] = dict(original["content"])

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return copy.deepcopy(raw)
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [copy.deepcopy(raw)]}
        if "/m.annotation" in path and scope in {"event", "room-read", "thread-read"}:
            started.set()
            await release.wait()
        return {"chunk": []}

    async def display_name(_room, _sender):
        started.set()
        await release.wait()
        return "Alice"

    adapter = _make_adapter()
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)),
                                     sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")))
    adapter._joined_rooms = {ROOM}
    adapter._event_context_cache = cache = MatrixEventContextCache(max_entries=3)
    adapter._get_display_name = display_name
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    loader = AsyncMock(return_value=(str(image), "image/png"))
    adapter._cache_quoted_image = loader
    parent = await cache.resolve(adapter._client, ROOM, "$target", loader)
    parent_ref = weakref.ref(parent)
    independent = MatrixEventContextCache(max_entries=3)
    independent.store(ROOM, "$target", replace(parent))

    if scope == "reply":
        context = adapter._extract_reply_context(ROOM, "question", {"body": "question"},
                                                {"m.in_reply_to": {"event_id": "$target"}},
                                                sender=SENDER, chat_type="group")
    elif scope == "room":
        context = _rendered(adapter.fetch_room_history(ROOM, "$current"))
    elif scope == "thread":
        context = _rendered(adapter.fetch_thread_history(ROOM, "$target", before_event_id="$current"))
    else:
        context = adapter.read_matrix_context({"event": "event", "room-read": "room", "thread-read": "thread"}[scope],
                                             ROOM, "$target", 1, requester=SENDER)
    pending = asyncio.create_task(context)
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        for index in range(3):
            cache.store(ROOM, f"$before{index}", MatrixEventContext(SENDER, "unrelated"))
        if change in {"original", "replacement"}:
            cache.redact(ROOM, "$target" if change == "original" else "$edit")
        elif change == "edit":
            raw = _edited(original, "current text")
            raw["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$next"
            cache.apply_edit(ROOM, SENDER, raw["unsigned"]["m.relations"]["m.replace"]["content"], replacement_id="$next")
        for index in range(3):
            cache.store(ROOM, f"$after{index}", MatrixEventContext(SENDER, "unrelated"))
        gc.collect()
    finally:
        release.set()
        result = await pending

    text = {"unchanged": "[image]", "original": "[redacted]", "replacement": "[event content unavailable]", "edit": "current text"}[change]
    if scope == "reply":
        assert result == MatrixReplyContext("question", "$target", text if change in {"unchanged", "edit"} else None,
                                           SENDER, "Alice", False, True,
                                           media_path=str(image) if change == "unchanged" else None,
                                           media_type="image/png" if change == "unchanged" else None,
                                           media_content_id=parent.attachment_identity if change == "unchanged" else None)
    elif scope in {"room", "thread"}:
        heading = "Recent room messages" if scope == "room" else "Earlier messages in this thread"
        suffix = "\n[Matrix event state unavailable: replacement was redacted.]" if change == "replacement" else ""
        assert result == f"[{heading}]\n[Alice] {text}{suffix}"
    else:
        event = {"event_id": "$target", "sender": SENDER, "body": text,
                 "msgtype": "m.image" if change == "unchanged" else "m.text" if change == "edit" else None,
                 "thread_id": None, "timestamp": None, "sender_authorized": True}
        if change in {"unchanged", "edit"}:
            event["edited"] = True
        elif change == "original":
            event["redacted"] = True
        assert result == {"events": [event], "errors": [{"event_id": "$target", "error": "replacement was redacted"}]
                          if change == "replacement" else [], "skipped": 0}
    assert independent.recheck(ROOM, independent.history_entry(ROOM, "$target")).media_path == str(image)
    assert len(cache._entries) <= cache.max_entries
    current = cache.recheck(ROOM, parent)
    current_ref = weakref.ref(current)
    del parent, current, pending, result, context
    await asyncio.sleep(0)
    independent._entries.clear()
    cache._entries.clear()
    gc.collect()
    assert (parent_ref(), current_ref()) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consumer",
    [
        "queued",
        "queued-copy",
        "queued-merge",
        "queued-command",
        "queued-unresolved-edit",
        "queued-room-intake",
        "queued-media-intake",
        "reaction-room",
        "reaction-thread",
        "error-room",
        "error-thread",
    ],
)
@pytest.mark.parametrize("withdrawn", [False, True])
async def test_active_consumers_retain_dependencies_from_intake(
    consumer, withdrawn, tmp_path, monkeypatch
):
    with ExitStack() as stores:
        import gc
        import weakref

        from gateway.config import GatewayConfig, Platform
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource, SessionStore
        from tests.gateway.test_matrix import _make_adapter

        adapter = _make_adapter()
        adapter._joined_rooms = {ROOM}
        adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        adapter._is_dm_room = AsyncMock(return_value=False)
        adapter._is_sender_authorized = lambda *_args, **_kwargs: True
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._event_context_cache = cache = MatrixEventContextCache(max_entries=2)
        target = "$target"
        raw = _original(target, "withdrawn quote")
        started, release = asyncio.Event(), asyncio.Event()
        reaction = {
            "type": "m.reaction",
            "event_id": "$reaction",
            "sender": SENDER,
            "content": {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": target,
                    "key": "withdrawn reaction",
                }
            },
        }
        failed = {
            **_original("$failed", ""),
            "type": "m.room.encrypted",
            "content": {"ciphertext": "unavailable"},
        }
        if consumer == "error-thread":
            failed["content"]["m.relates_to"] = {"rel_type": "m.thread", "event_id": target}

        async def request(_method, path, **_kwargs):
            if "/event/" in path:
                return json.loads(json.dumps(raw))
            if "/context/" in path:
                return {"start": "boundary"}
            if path.endswith("/m.thread") and consumer.startswith("error"):
                return {"chunk": [failed]}
            if "/messages" in path or path.endswith("/m.thread"):
                return {"chunk": [failed, raw] if consumer.startswith("error") else [raw]}
            if path.endswith("/m.annotation"):
                if consumer.startswith("error"):
                    started.set()
                    await release.wait()
                return {"chunk": [reaction] if consumer.startswith("reaction") else []}
            raise AssertionError(path)

        adapter._client = SimpleNamespace(
            api=SimpleNamespace(request=AsyncMock(side_effect=request)),
            crypto=None,
            sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
        )
        independent = MatrixEventContextCache()
        independent.store(ROOM, target, MatrixEventContext(SENDER, "independent owner"))

        def evict():
            for index in range(cache.max_entries):
                cache.store(ROOM, f"$other{index}", MatrixEventContext(SENDER, "other"))
            gc.collect()

        if consumer.startswith("queued"):
            source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
            body = f"> <{SENDER}> withdrawn quote\n\nquestion"
            relation = {"m.in_reply_to": {"event_id": target}}
            intake = consumer in {"queued-room-intake", "queued-media-intake"}
            if intake:
                from gateway.platforms.event import MessageEvent

                adapter._is_dm_room = AsyncMock(return_value=True)
                adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
                    display_name="Room", room_topic=None, server_name=None, room_state=None,
                ))
                adapter.set_message_handler(AsyncMock())
                adapter._text_batch_delay_seconds = 0
                adapter._busy_text_debounce_seconds = 0
                session_key = adapter._event_session_key(MessageEvent("question", source=source))
                adapter._active_sessions[session_key] = asyncio.Event()

                async def pause(*_args):
                    started.set()
                    await release.wait()
                    return True

                content = {"msgtype": "m.text", "body": body, "m.relates_to": relation}
                if consumer == "queued-room-intake":
                    adapter._is_allowed_matrix_room_event = pause
                else:
                    import base64

                    content.update(msgtype="m.image", url="mxc://example.org/image")

                    async def download(_uri):
                        await pause()
                        return base64.b64decode(
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                        )

                    adapter._client.download_media = download
                pending = asyncio.create_task(adapter._on_room_message(SimpleNamespace(
                    room_id=ROOM, sender=SENDER, event_id="$current", timestamp=0, content=content,
                )))
                try:
                    await asyncio.wait_for(started.wait(), timeout=2)
                    if withdrawn:
                        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts=target))
                    evict()
                finally:
                    release.set()
                    await pending
                event = adapter._pending_messages.pop(session_key)
                adapter._pending_messages["session"] = event
                del pending
            else:
                event = await adapter._build_inbound_event(
                    ROOM,
                    SENDER,
                    "$current",
                    body,
                    {"body": body, "m.relates_to": relation},
                    relation,
                    ctx=(body, True, "dm", None, "Alice", False, source),
                )
            if consumer == "queued-copy":
                event = replace(event, text="copied question")
            if consumer == "queued-merge":
                from gateway.platforms.event import MessageEvent

                adapter._pending_messages["session"] = MessageEvent(
                    "first question", source=source
                )
            if consumer == "queued-command":
                runner = object.__new__(GatewayRunner)
                runner.config = GatewayConfig()
                runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
                stores.callback(runner.session_store.close_all_db_handles)
                runner.adapters = {Platform.MATRIX: adapter}
                event = replace(event, text="/queue question")
                monkeypatch.setattr(
                    runner,
                    "_enqueue_fifo",
                    lambda _key, item, _adapter: adapter._pending_messages.update(
                        session=item
                    ),
                )
                monkeypatch.setattr(runner, "_queue_depth", lambda *_args, **_kwargs: 1)
                await runner._busy_queue_command(event, "session", source)
            elif not intake:
                await adapter._handle_message_while_active(event, "session")
            del event
            if consumer == "queued-unresolved-edit":
                await adapter._on_room_message(
                    SimpleNamespace(
                        room_id=ROOM,
                        sender="@mallory:example.org",
                        event_id="$untrusted-edit",
                        timestamp=0,
                        content={
                            "msgtype": "m.text",
                            "body": "* untrusted replacement",
                            "m.new_content": {
                                "msgtype": "m.text",
                                "body": "untrusted replacement",
                            },
                            "m.relates_to": {"rel_type": "m.replace", "event_id": target},
                        },
                    )
                )
            if withdrawn and not intake:
                await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts=target))
            evict()
            event = adapter._pending_messages.pop("session")
            runner = object.__new__(GatewayRunner)
            runner.config = GatewayConfig()
            runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
            stores.callback(runner.session_store.close_all_db_handles)
            runner.adapters = {Platform.MATRIX: adapter}
            if intake:
                monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: "native")
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[{}], session_key="session"
            )
            assert ("withdrawn quote" in result) is (not withdrawn)
            assert event.reply_to_message_id == target
            requests = adapter._client.api.request.await_args_list
            assert [call.args[1] for call in requests] == (
                []
                if withdrawn
                else [
                    f"/_matrix/client/v3/rooms/{quote(ROOM, safe='')}/event/{quote(target, safe='')}"
                ]
            )
            dependency = cache.history_entry(ROOM, target)
            reference = weakref.ref(dependency) if dependency else lambda: None
            del event, dependency, result
        elif consumer.startswith("reaction"):
            entries = (
                await fetch_room_entries(adapter._client, cache, ROOM, "$current", limit=1)
                if consumer == "reaction-room"
                else await fetch_thread_entries(
                    adapter._client,
                    cache,
                    ROOM,
                    target,
                    limit=1,
                    before_event_id="$current",
                )
            )
            if withdrawn:
                await adapter._on_redaction(
                    SimpleNamespace(room_id=ROOM, redacts="$reaction")
                )
            evict()
            result = await _rendered(MatrixHistoryContext.prepare(adapter, ROOM, entries, "History"))
            assert result == "[History]\n[Alice] withdrawn quote" + (
                ""
                if withdrawn
                else f"\n[reaction by {SENDER} to {target}] withdrawn reaction"
            )
            dependency = cache.history_entry(ROOM, "$reaction")
            reference = weakref.ref(dependency) if dependency else lambda: None
            del entries, result, dependency
        else:
            kind = "room" if consumer == "error-room" else "thread"
            pending = asyncio.create_task(
                read_matrix_context(adapter, kind, ROOM, target, 3, requester=SENDER)
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                if withdrawn:
                    await adapter._on_redaction(
                        SimpleNamespace(room_id=ROOM, redacts="$failed")
                    )
                evict()
            finally:
                release.set()
                result = await pending
            failed_event = {
                "event_id": "$failed",
                "sender": SENDER,
                "body": "[redacted]",
                "msgtype": None,
                "thread_id": target if consumer == "error-thread" else None,
                "timestamp": None,
                "sender_authorized": True,
                "redacted": True,
            }
            target_event = {
                "event_id": target,
                "sender": SENDER,
                "body": "withdrawn quote",
                "msgtype": "m.text",
                "thread_id": None,
                "timestamp": None,
                "sender_authorized": True,
            }
            events = [target_event, failed_event] if withdrawn else [target_event]
            assert result == {
                "events": events,
                "errors": []
                if withdrawn
                else [{"event_id": "$failed", "error": "missing decryption keys"}],
                "skipped": 0,
            }
            reference = lambda: None
            del pending, result
        assert independent.history_entry(ROOM, target).text == "independent owner"
        cache._entries.clear()
        gc.collect()
        assert reference() is None
        assert len(cache._active_states) == 0
