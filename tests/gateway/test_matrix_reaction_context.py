"""Matrix reactions appear in bounded reads without starting a turn."""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.reaction_context import (
    MatrixReaction,
    ReactionSnapshot,
    UndecryptableReaction,
    fetch_event_reactions,
)
from plugins.platforms.matrix.reply_context import MatrixEventContext


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
        assert path.endswith("/relations/%24message/m.annotation")
        return {"chunk": reactions}

    client = SimpleNamespace(
        get_event=AsyncMock(return_value={
            "type": "m.room.message", "event_id": target_id, "sender": "@alice:example.org",
            "content": {"msgtype": "m.text", "body": "hello"},
        }),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
    )
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={room_id}, _user_id="@bot:example.org",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
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
    mautrix_types.Event = SimpleNamespace(deserialize=lambda event: event)
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
async def test_catch_up_reports_failed_reaction_lookup():
    from tests.gateway.test_matrix import _make_adapter

    room_id = "!room:example.org"
    adapter = _make_adapter()
    adapter._client = MagicMock()

    async def request(_method, path, **_kwargs):
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path:
            return {"chunk": [{
                "type": "m.room.message", "event_id": "$earlier", "sender": "@alice:example.org",
                "content": {"msgtype": "m.text", "body": "Earlier"},
            }]}
        raise RuntimeError("reaction lookup failed")

    adapter._client.api.request = AsyncMock(side_effect=request)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True

    context = await adapter.fetch_room_context(room_id, "$current")

    assert context == (
        "[Recent room messages]\n[Some reactions could not be read.]\n[Alice] Earlier"
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

    context = await adapter._format_history_context("!room:example.org", [entry], "Recent room messages")

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

    async def request(_method, path, **_kwargs):
        assert path.endswith("/relations/%24message/m.annotation")
        return {"chunk": [encrypted_reaction("$no-keys"), encrypted_reaction("$broken")], "next_batch": "more"}

    async def decrypt(raw):
        if raw["event_id"] == "$no-keys":
            raise SessionNotFound()
        raise ValueError("bad ratchet index")

    client = SimpleNamespace(
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt),
        get_event=AsyncMock(return_value={
            "type": "m.room.message", "event_id": target_id, "sender": "@alice:example.org",
            "content": {"msgtype": "m.text", "body": "hello"},
        }),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
    )
    adapter = SimpleNamespace(
        _client=client, _joined_rooms={room_id}, _user_id="@bot:example.org",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )

    mautrix = types.ModuleType("mautrix")
    mautrix_types = types.ModuleType("mautrix.types")
    mautrix_types.Event = SimpleNamespace(deserialize=lambda event: event)
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
    }

