"""Matrix reactions appear in bounded reads without starting a turn."""

import asyncio
import gc
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.reaction_context import (
    MatrixReaction,
    ReactionSnapshot,
    UndecryptableReaction,
    fetch_event_reactions,
)
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache
from plugins.platforms.matrix.room_context import MatrixHistoryContext
from tests.gateway.test_matrix import _rendered
from plugins.platforms.matrix.thread_context import NON_CONVERSATIONAL_KEY


@pytest.mark.asyncio
async def test_event_read_includes_current_reactions_once_per_sender_and_emoji():
    room_id = "!room:example.org"
    target_id = "$message"
    reactions = [
        {"type": "m.reaction", "event_id": "$first", "sender": "@alice:example.org",
         "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target_id, "key": "👍"}}},
        {"type": "m.reaction", "event_id": "$duplicate", "sender": "@alice:example.org",
         "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target_id, "key": "👍"}}},
        {"type": "m.reaction", "event_id": "$long", "sender": "@bob:example.org",
         "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target_id, "key": "x" * 200}}},
        {"type": "m.reaction", "event_id": "$removed", "sender": "@bob:example.org",
         "content": {}, "unsigned": {"redacted_because": {"event_id": "$redaction"}}},
        {"type": "m.reaction", "event_id": "$wrong", "sender": "@bob:example.org",
         "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$other", "key": "🔥"}}},
    ]

    async def request(_method, path, **_kwargs):
        if path.endswith("/event/%24message"):
            return {
                "type": "m.room.message", "event_id": target_id, "sender": "@alice:example.org",
                "content": {"msgtype": "m.text", "body": "hello"},
            }
        assert path.endswith("/relations/%24message/m.annotation")
        return {"chunk": reactions}

    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
    )
    adapter = SimpleNamespace(
        _event_context_cache=MatrixEventContextCache(),
        _client=client, _joined_rooms={room_id}, _user_id="@bot:example.org",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )

    result = await read_matrix_context(adapter, "event", room_id, target_id, 5,
                                       requester="@alice:example.org")

    assert result == {
        "events": [{
            "event_id": target_id, "sender": "@alice:example.org", "body": "hello",
            "msgtype": "m.text", "thread_id": None, "timestamp": None,
            "sender_authorized": True,
            "reactions": [{
                "event_id": "$first", "sender": "@alice:example.org",
                "emoji": "👍", "target_event_id": target_id, "sender_authorized": True,
            }, {
                "event_id": "$long", "sender": "@bob:example.org",
                "emoji": "x" * 40, "emoji_truncated": True,
                "target_event_id": target_id, "sender_authorized": True,
            }],
        }],
        "errors": [],
        "skipped": 0,
    }


@pytest.mark.asyncio
async def test_encrypted_reaction_requires_decryption_and_uses_outer_cleartext_relation():
    room_id = "!room:example.org"
    target_id = "$message"
    raw = {
        "type": "m.room.encrypted", "event_id": "$encrypted", "sender": "@alice:example.org",
        "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target_id, "key": "❤️"}},
    }
    decrypt = AsyncMock(return_value={
        "type": "m.reaction", "content": {
            "m.relates_to": {"rel_type": "m.annotation", "event_id": "$wrong", "key": "👎"},
        },
    })
    client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value={"chunk": [raw]})),
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt),
    )

    mautrix = types.ModuleType("mautrix")
    mautrix_types = types.ModuleType("mautrix.types")
    mautrix_types.EncryptedEvent = SimpleNamespace(deserialize=lambda event: event)
    mautrix_types.JSON = lambda event: event
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix_types}):
        visible = await fetch_event_reactions(client, room_id, target_id)

    client.crypto = None
    missing = await fetch_event_reactions(client, room_id, target_id)

    assert (visible, missing) == (
        ReactionSnapshot((MatrixReaction("$encrypted", "@alice:example.org", "❤️", target_id),)),
        ReactionSnapshot(undecryptable=(UndecryptableReaction("$encrypted", "missing decryption keys"),)),
    )
    decrypt.assert_awaited_once_with(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize(("relations", "expected"), [
    pytest.param(
        RuntimeError("reaction lookup failed"),
        "[Recent room messages]\n[Some reactions could not be read.]\n[Alice] Earlier",
        id="lookup-failed",
    ),
    pytest.param(
        {"chunk": [], "next_batch": "more"},
        "[Recent room messages]\n[Alice] Earlier\n[More reactions were omitted from this bounded context.]",
        id="truncated",
    ),
    pytest.param(
        {"chunk": [{
            "type": "m.room.encrypted", "event_id": "$encrypted-reaction", "sender": "@bob:example.org",
            "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$earlier", "key": "👍"}},
        }]},
        "[Recent room messages]\n[Alice] Earlier\n[Some reactions could not be decrypted.]",
        id="undecryptable",
    ),
])
async def test_catch_up_reports_incomplete_reactions(relations, expected):
    from tests.gateway.test_matrix import _make_adapter

    room_id = "!room:example.org"
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.crypto = None

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [{
                "type": "m.room.message", "event_id": "$earlier", "sender": "@alice:example.org",
                "content": {"msgtype": "m.text", "body": "Earlier"},
            }]}
        assert path.endswith("/relations/%24earlier/m.annotation")
        if isinstance(relations, Exception):
            raise relations
        return relations

    adapter._client.api.request = AsyncMock(side_effect=request)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True

    context = await _rendered(adapter.fetch_room_history(room_id, "$current"))

    assert context == expected


@pytest.mark.parametrize(("scope", "previous_turn", "targets"), [
    ("room", True, ["$gated-1", "$gated-2"]),
    ("thread", True, ["$gated-1", "$gated-2"]),
    ("thread", False, ["$root", "$gated-1", "$gated-2"]),
])
@pytest.mark.asyncio
async def test_catch_up_reactions_belong_to_the_messages_that_the_scan_kept(scope, previous_turn, targets):
    """A scan that stops at the previous turn leaves out the thread root and the older
    messages, and every scan passes over the bot's status notices. Their reactions are not
    looked up, and no other message receives them."""
    from urllib.parse import unquote

    from tests.gateway.test_matrix import (
        _CATCH_UP_THREAD, _catch_up_adapter, _catch_up_message, _catch_up_trigger,
    )

    relates_to = _CATCH_UP_THREAD if scope == "thread" else {}
    older = [
        _catch_up_message("$reply", "@bot:example.org", "Previous answer", relates_to),
        _catch_up_message("$older", "@bob:example.org", "Older", relates_to),
    ]
    status = _catch_up_message("$status", "@bot:example.org", "Still working", relates_to)
    status["content"] |= {"msgtype": "m.notice", NON_CONVERSATIONAL_KEY: True}
    adapter = _catch_up_adapter([
        _catch_up_message("$gated-2", "@bob:example.org", "Gated two", relates_to),
        status,
        _catch_up_message("$gated-1", "@bob:example.org", "Gated one", relates_to),
        *(older if previous_turn else []),
    ], thread=scope == "thread")
    history = adapter._client.api.request.side_effect
    looked_up = []

    async def request(method, path, **kwargs):
        if not path.endswith("/m.annotation"):
            return await history(method, path, **kwargs)
        target = unquote(path.split("/relations/")[1].removesuffix("/m.annotation"))
        looked_up.append(target)
        return {"chunk": [{
            "type": "m.reaction", "event_id": f"$reaction-to-{target[1:]}", "sender": "@carol:example.org",
            "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": "👍"}},
        }]}

    adapter._client.api.request = AsyncMock(side_effect=request)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    if scope == "thread":
        adapter._thread_require_mention = True
        await adapter._threads.mark_async("$root")
    event = await _catch_up_trigger(adapter, relates_to)

    context = await _rendered(adapter.fetch_mention_history(event))

    heading = "Earlier messages in this thread" if scope == "thread" else "Recent room messages"
    bodies = {"$root": "[alice] Thread root", "$gated-1": "[bob] Gated one", "$gated-2": "[bob] Gated two"}
    assert (context, sorted(looked_up)) == (
        "\n".join([f"[{heading}]", *(
            f"{bodies[target]}\n[reaction by @carol:example.org to {target}] 👍" for target in targets
        )]),
        sorted(targets),
    )


@pytest.mark.asyncio
async def test_catch_up_marks_a_clipped_reaction_key():
    from tests.gateway.test_matrix import _make_adapter

    adapter = _make_adapter()
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    entry = MatrixEventContext("@alice:example.org", "Earlier", reactions=(
        MatrixReaction("$reaction", "@alice:example.org", "x" * 40, "$earlier", emoji_truncated=True),
    ))

    context = await _rendered(MatrixHistoryContext.prepare(
        adapter, "!room:example.org", [entry], "Recent room messages",
    ))

    assert context == (
        "[Recent room messages]\n[Alice] Earlier\n"
        "[reaction by @alice:example.org to $earlier] " + "x" * 40 + " [key truncated]"
    )


@pytest.mark.asyncio
async def test_reaction_batch_keeps_fast_results_when_another_lookup_stalls():
    from plugins.platforms.matrix import reaction_context

    waiting = asyncio.Event()

    async def request(_method, path, **_kwargs):
        if "/%24slow/" in path:
            await waiting.wait()
        return {"chunk": []}

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    with patch.object(reaction_context, "_REACTION_BATCH_TIMEOUT_SECONDS", 0.05):
        snapshots = await reaction_context.fetch_reactions_for_events(
            client, "!room:example.org", ["$fast", "$slow"],
        )

    assert snapshots == [
        reaction_context.ReactionSnapshot(),
        reaction_context.ReactionSnapshot(error="reactions unavailable: timeout"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("eviction", [False, True])
async def test_reaction_page_retains_later_redaction_during_first_decrypt(eviction: bool):
    mautrix_types = pytest.importorskip("mautrix.types")
    room_id = "!room:example.org"
    sender = "@alice:example.org"
    target_id = "$original"
    first_relation = {"rel_type": "m.annotation", "event_id": target_id, "key": "retained reaction"}
    later_relation = {"rel_type": "m.annotation", "event_id": target_id, "key": "withdrawn reaction"}
    first = {
        "room_id": room_id, "event_id": "$first", "sender": sender,
        "type": "m.room.encrypted", "origin_server_ts": 1,
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "first",
            "session_id": "session", "sender_key": "key", "device_id": "device",
            "m.relates_to": first_relation,
        },
    }
    later = {
        "room_id": room_id, "event_id": "$later", "sender": sender,
        "type": "m.reaction", "origin_server_ts": 2,
        "content": {"m.relates_to": later_relation},
    }
    original = {
        "room_id": room_id, "event_id": target_id, "sender": sender,
        "type": "m.room.message", "origin_server_ts": 0,
        "content": {"msgtype": "m.text", "body": "retained text"},
    }
    started, release = asyncio.Event(), asyncio.Event()

    async def request(_method, path, **_kwargs):
        return {"chunk": [first, later]} if "/m.annotation" in path else original

    async def decrypt(_event):
        started.set()
        await release.wait()
        return mautrix_types.Event.deserialize({
            **first, "type": "m.reaction", "content": {"m.relates_to": first_relation},
        })

    cache = MatrixEventContextCache(max_entries=1)
    adapter = SimpleNamespace(
        _client=SimpleNamespace(api=SimpleNamespace(request=request),
                                crypto=SimpleNamespace(decrypt_megolm_event=decrypt)),
        _joined_rooms={room_id}, _user_id="@bot:example.org", _event_context_cache=cache,
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )
    pending = asyncio.create_task(read_matrix_context(
        adapter, "event", room_id, target_id, 1, requester=sender,
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        cache.redact(room_id, "$later")
        if eviction:
            cache.store(room_id, "$pressure", MatrixEventContext(sender, "pressure"))
            gc.collect()
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)

    assert result == {"events": [{
        "event_id": target_id, "sender": sender, "body": "retained text", "msgtype": "m.text",
        "thread_id": None, "timestamp": 0, "sender_authorized": True,
        "reactions": [{
            "event_id": "$first", "sender": sender, "emoji": "retained reaction",
            "target_event_id": target_id, "sender_authorized": True,
        }],
    }], "errors": [], "skipped": 0}


class SessionNotFound(Exception):
    pass


@pytest.mark.asyncio
async def test_event_read_reports_incomplete_reactions_against_the_target():
    room_id = "!room:example.org"
    target_id = "$message"

    def encrypted_reaction(event_id):
        return {
            "type": "m.room.encrypted", "event_id": event_id, "sender": "@bob:example.org",
            "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": target_id, "key": "👍"}},
        }

    target = {
        "type": "m.room.message", "event_id": target_id, "sender": "@alice:example.org",
        "content": {"msgtype": "m.text", "body": "hello"},
    }

    async def request(_method, path, **_kwargs):
        if path.endswith("/event/%24message"):
            return target
        assert path.endswith("/relations/%24message/m.annotation")
        return {"chunk": [encrypted_reaction("$no-keys"), encrypted_reaction("$broken")], "next_batch": "more"}

    async def decrypt(raw):
        if raw["event_id"] == "$no-keys":
            raise SessionNotFound()
        raise ValueError("bad ratchet index")

    client = SimpleNamespace(
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
    )
    adapter = SimpleNamespace(
        _event_context_cache=MatrixEventContextCache(),
        _client=client, _joined_rooms={room_id}, _user_id="@bot:example.org",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room,
        adapter,
    )

    mautrix = types.ModuleType("mautrix")
    mautrix_types = types.ModuleType("mautrix.types")
    mautrix_types.EncryptedEvent = SimpleNamespace(deserialize=lambda event: event)
    mautrix_types.JSON = lambda event: event
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": mautrix_types}):
        result = await read_matrix_context(adapter, "event", room_id, target_id, 1,
                                           requester="@alice:example.org")

    assert result == {
        "events": [{
            "event_id": target_id, "sender": "@alice:example.org", "body": "hello",
            "msgtype": "m.text", "thread_id": None, "timestamp": None,
            "sender_authorized": True, "reactions_truncated": True,
        }],
        "errors": [
            {"event_id": target_id, "reaction_event_id": "$no-keys", "error": "reaction missing decryption keys"},
            {"event_id": target_id, "reaction_event_id": "$broken", "error": "reaction decryption failed"},
        ],
        "skipped": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed_by", [
    None, "_handle_approval_reaction", "_handle_model_picker_reaction", "_handle_choice_picker_reaction",
])
async def test_reaction_reaches_prompt_handlers_without_starting_a_turn(claimed_by):
    from tests.gateway.test_matrix import _make_adapter

    adapter = _make_adapter()
    adapter._user_id = "@bot:example.org"
    adapter.handle_message = AsyncMock()
    handlers = ("_handle_approval_reaction", "_handle_model_picker_reaction", "_handle_choice_picker_reaction")
    for name in handlers:
        setattr(adapter, name, AsyncMock(return_value=name == claimed_by))
    event = SimpleNamespace(
        sender="@alice:example.org", event_id="$reaction", room_id="!room:example.org",
        content={"m.relates_to": {"rel_type": "m.annotation", "event_id": "$message", "key": "👍"}},
    )

    await adapter._on_reaction(event)

    reached = handlers if claimed_by is None else handlers[:handlers.index(claimed_by) + 1]
    assert {name: getattr(adapter, name).await_args_list for name in handlers} == {
        name: [call("!room:example.org", "$message", "👍", "@alice:example.org")] if name in reached else []
        for name in handlers
    }
    adapter.handle_message.assert_not_awaited()
