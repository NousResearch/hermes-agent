"""Reply and quote context resolution for Matrix inbound messages."""
import asyncio
import types

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.platforms.event import MessageType
from tests.gateway.matrix_helpers import _make_adapter, _make_room_adapter


@pytest.mark.asyncio
async def test_reply_context_from_later_matrix_chunk_survives_text_batch():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org"]
    )
    adapter._get_display_name = AsyncMock(return_value="Bob")
    adapter._background_read_receipt = MagicMock()
    adapter._text_batch_delay_seconds = 60

    try:
        await adapter._handle_text_message(
            "!room:example.org", "@alice:example.org", "$first", 0,
            {"msgtype": "m.text", "body": "first"}, {},
        )
        await adapter._handle_text_message(
            "!room:example.org", "@alice:example.org", "$second", 0,
            {"msgtype": "m.text", "body": "> <@bob:example.org> earlier\n\nsecond"},
            {"m.in_reply_to": {"event_id": "$parent"}},
        )

        queued = list(adapter._pending_text_batches.values())
        assert [(
            event.text, event.reply_to_message_id, event.reply_to_text,
            event.reply_to_author_id, event.reply_to_author_name,
            event.reply_to_author_authorized,
        ) for event in queued] == [(
            "first\nsecond", "$parent", "earlier", "@bob:example.org", "Bob", False,
        )]
    finally:
        for task in adapter._pending_text_batch_tasks.values():
            task.cancel()


@pytest.mark.asyncio
async def test_thread_fallback_is_not_an_explicit_reply():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org", "@bob:example.org"]
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$message", "hello",
        {"msgtype": "m.text", "body": "hello"},
        {"rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
         "m.in_reply_to": {"event_id": "$root"}},
    )

    assert (event.source.thread_id, event.reply_to_message_id, event.reply_to_text) == (
        "$root", None, None,
    )


@pytest.mark.parametrize(
    "content,expected",
    [
        ({"rel_type": "m.thread", "event_id": "$root", "m.in_reply_to": {"event_id": "$reply"}},
         ("$root", "$reply", None, False)),
        ({"rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
          "m.in_reply_to": {"event_id": "$root"}},
         ("$root", None, "$root", False)),
        ({"rel_type": "m.replace", "event_id": "$old"}, (None, None, None, True)),
        (["malformed"], (None, None, None, False)),
    ],
)
def test_matrix_relation_distinguishes_reply_from_thread_fallback(content, expected):
    from plugins.platforms.matrix.relations import MatrixRelation

    relation = MatrixRelation.from_content(content)

    assert (relation.thread_root, relation.reply_target, relation.thread_fallback_target, relation.is_edit) == expected


@pytest.mark.asyncio
async def test_legacy_thread_fallback_quote_is_not_current_message():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(return_value=[
        "@bot:example.org", "@alice:example.org",
    ])
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = True
    body = "> <@alice:example.org> quoted @file:/tmp/private\n\n!model"

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$current", body,
        {"msgtype": "m.text", "body": body},
        {"rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
         "m.in_reply_to": {"event_id": "$root"}},
    )

    assert (
        event.text, event.message_type, event.source.thread_id,
        event.reply_to_message_id, event.reply_to_text,
    ) == ("/model", MessageType.COMMAND, "$root", None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("content,expected_text", [
    ({"body": "> quoted from elsewhere\n\nwhat does this mean?"}, "> quoted from elsewhere\n\nwhat does this mean?"),
    ({"body": "> <@alice:example.org> root\n\n> my own quote\n\nquestion"}, "> my own quote\n\nquestion"),
    ({"body": "> * <@alice:example.org> waves\n\nhello"}, "hello"),
    ({"body": "> <@bob:example.org> said it failed\nI disagree"}, "> <@bob:example.org> said it failed\nI disagree"),
    ({"body": "> latest in thread\n\nmy answer", "format": "org.matrix.custom.html",
      "formatted_body": "<mx-reply><blockquote>latest in thread</blockquote></mx-reply>my answer"}, "my answer"),
])
async def test_thread_message_strips_only_the_reply_fallback(content, expected_text):
    adapter = _make_room_adapter()

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$message", content["body"],
        {"msgtype": "m.text", **content},
        {"rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
         "m.in_reply_to": {"event_id": "$latest"}},
    )

    assert (event.text, event.reply_to_message_id, event.reply_to_text) == (expected_text, None, None)


@pytest.mark.asyncio
async def test_reply_without_inline_quote_fetches_parent_with_author_trust():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org", "@bob:example.org"]
    )
    adapter._client.get_event = AsyncMock(return_value=types.SimpleNamespace(
        sender="@stranger:example.org",
        content={"msgtype": "m.text", "body": "prior message"},
    ))
    adapter._get_display_name = AsyncMock(side_effect=lambda room, user: user.split(":")[0][1:])
    adapter._is_sender_authorized = MagicMock(side_effect=lambda user, **kwargs: user != "@stranger:example.org")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$reply", "new message",
        {"msgtype": "m.text", "body": "new message"},
        {"m.in_reply_to": {"event_id": "$parent"}},
    )

    assert (
        event.reply_to_message_id, event.reply_to_text, event.reply_to_author_id,
        event.reply_to_author_name, event.reply_to_author_authorized,
    ) == ("$parent", "prior message", "@stranger:example.org", "stranger", False)


@pytest.mark.asyncio
async def test_media_reply_without_inline_quote_fetches_parent_and_survives_failure():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org"]
    )
    adapter._client.get_event = AsyncMock(side_effect=[
        types.SimpleNamespace(sender="@alice:example.org", content={"msgtype": "m.text", "body": "earlier"}),
        RuntimeError("homeserver unavailable"),
    ])
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/photo.png")
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    content = {"msgtype": "m.image", "body": "photo.png", "url": "mxc://example.org/photo"}
    for event_id, parent_id in (("$first", "$first-parent"), ("$second", "$second-parent")):
        await adapter._handle_media_message(
            "!room:example.org", "@alice:example.org", event_id, 0.0, content,
            {"m.in_reply_to": {"event_id": parent_id}}, "m.image",
        )

    assert [
        (event.message_type, event.reply_to_message_id, event.reply_to_text, event.media_urls)
        for event in captured
    ] == [
        (MessageType.PHOTO, "$first-parent", "earlier", ["/tmp/photo.png"]),
        (MessageType.PHOTO, "$second-parent", None, ["/tmp/photo.png"]),
    ]
    assert adapter._client.get_event.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed_author", ["@stranger:example.org", "@bot:example.org"])
async def test_inline_reply_fallback_does_not_verify_claimed_author(claimed_author):
    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Claimed author")
    adapter._is_sender_authorized = MagicMock(return_value=True)

    reply = await adapter._extract_reply_context(
        "!room:example.org", f"> <{claimed_author}> earlier\n\nContinue", {},
        {"m.in_reply_to": {"event_id": "$parent"}},
        sender="@alice:example.org", chat_type="group",
    )

    assert (reply.body, reply.author_id, reply.is_own_message, reply.author_authorized) == (
        "Continue", claimed_author, False, False,
    )


@pytest.mark.asyncio
async def test_reply_context_uses_edit_and_never_resurfaces_redacted_text():
    adapter = _make_adapter()
    room_id = "!room:example.org"
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org", "@bob:example.org"]
    )
    adapter._client.get_event = AsyncMock()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()

    await adapter._on_room_message(types.SimpleNamespace(
        room_id=room_id, sender="@alice:example.org", event_id="$parent", timestamp=0,
        content={"msgtype": "m.text", "body": "before"},
    ))
    await adapter._on_room_message(types.SimpleNamespace(
        room_id=room_id, sender="@alice:example.org", event_id="$edit", timestamp=0,
        content={"msgtype": "m.text", "body": "* after",
                 "m.relates_to": {"rel_type": "m.replace", "event_id": "$parent"},
                 "m.new_content": {"msgtype": "m.text", "body": "after"}},
    ))
    relation = {"m.in_reply_to": {"event_id": "$parent"}}
    edited_reply = await adapter._build_inbound_event(
        room_id, "@bob:example.org", "$reply1", "question",
        {"msgtype": "m.text", "body": "question"}, relation,
    )
    await adapter._on_redaction(types.SimpleNamespace(room_id=room_id, redacts="$parent"))
    await adapter._on_room_message(types.SimpleNamespace(
        room_id=room_id, sender="@alice:example.org", event_id="$late-edit", timestamp=0,
        content={"msgtype": "m.text", "body": "* revived",
                 "m.relates_to": {"rel_type": "m.replace", "event_id": "$parent"},
                 "m.new_content": {"msgtype": "m.text", "body": "revived"}},
    ))
    redacted_reply = await adapter._build_inbound_event(
        room_id, "@bob:example.org", "$reply2", "another question",
        {"msgtype": "m.text", "body": "another question"}, relation,
    )

    assert (edited_reply.reply_to_text, redacted_reply.reply_to_text) == ("after", None)
    adapter._client.get_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_from_another_sender_does_not_replace_uncached_reply_target():
    adapter = _make_room_adapter()
    room_id = "!room:example.org"
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._client.get_event = AsyncMock(return_value=types.SimpleNamespace(
        sender="@alice:example.org", content={"msgtype": "m.text", "body": "real text"},
    ))

    await adapter._on_room_message(types.SimpleNamespace(
        room_id=room_id, sender="@mallory:example.org", event_id="$edit", timestamp=0,
        content={"msgtype": "m.text", "body": "* forged",
                 "m.relates_to": {"rel_type": "m.replace", "event_id": "$alice-msg"},
                 "m.new_content": {"msgtype": "m.text", "body": "forged"}},
    ))
    reply = await adapter._build_inbound_event(
        room_id, "@bob:example.org", "$reply", "is this right?",
        {"msgtype": "m.text", "body": "is this right?"},
        {"m.in_reply_to": {"event_id": "$alice-msg"}},
    )

    assert (reply.reply_to_text, reply.reply_to_author_id) == ("real text", "@alice:example.org")


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["get_event", "image_loader"])
async def test_redaction_during_reply_image_resolution_returns_no_context(tmp_path, pause_at):
    from plugins.platforms.matrix.reply_context import MatrixEventContextCache

    room_id = "!room:example.org"
    event_id = "$image"
    started = asyncio.Event()
    release = asyncio.Event()
    image = tmp_path / "photo.png"
    image.write_bytes(b"png")
    event = types.SimpleNamespace(
        sender="@alice:example.org",
        content={"msgtype": "m.image", "body": "photo.png", "url": "mxc://example.org/photo"},
    )

    async def get_event(_room_id, _event_id):
        if pause_at == "get_event":
            started.set()
            await release.wait()
        return event

    async def load_image(_content, _event_id):
        if pause_at == "image_loader":
            started.set()
            await release.wait()
        return str(image), "image/png"

    client = MagicMock()
    client.get_event = AsyncMock(side_effect=get_event)
    cache = MatrixEventContextCache()
    resolving = asyncio.create_task(cache.resolve(client, room_id, event_id, load_image))
    await asyncio.wait_for(started.wait(), timeout=2)
    cache.redact(room_id, event_id)
    release.set()

    assert (await resolving, await cache.resolve(None, room_id, event_id)) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["reply-fetch", "cached-edit", "thread-fetch"])
@pytest.mark.parametrize("fallback", ["user-quote", "legacy-pill", "formatted-reply"])
async def test_context_preserves_user_quotes_and_removes_reply_fallbacks(
    path, fallback
):
    from plugins.platforms.matrix.reply_context import (
        MatrixEventContext,
        MatrixEventContextCache,
    )
    from plugins.platforms.matrix.thread_context import fetch_thread_entries

    quote = "> a user-authored quote"
    content = {"msgtype": "m.text", "body": f"{quote}\n\nrest of the message"}
    expected = content["body"]
    if fallback == "legacy-pill":
        content["body"] = "> <@alice:example.org> earlier\n\nrest of the message"
        expected = "rest of the message"
    if fallback == "formatted-reply":
        content.update(
            format="org.matrix.custom.html",
            formatted_body=(
                "<mx-reply><blockquote>earlier</blockquote></mx-reply>rest of the message"
            ),
        )
        expected = "rest of the message"
    cache = MatrixEventContextCache()
    client = MagicMock()
    client.get_event = AsyncMock(
        return_value={"sender": "@alice:example.org", "content": content}
    )
    if path == "reply-fetch":
        entry = await cache.resolve(client, "!room", "$parent")
    elif path == "cached-edit":
        cache.store(
            "!room", "$parent", MatrixEventContext("@alice:example.org", "previous")
        )
        cache.apply_edit(
            "!room",
            "@alice:example.org",
            {
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$parent"},
                "m.new_content": content,
            },
        )
        entry = await cache.resolve(client, "!room", "$parent")
    else:
        client.get_event = AsyncMock(side_effect=RuntimeError("root unavailable"))
        content["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$root"}
        client.api.request = AsyncMock(
            side_effect=[
                {"start": "trigger-boundary"},
                {"start": "messages-boundary", "chunk": []},
                {
                    "chunk": [
                        {
                            "event_id": "$parent",
                            "sender": "@alice:example.org",
                            "content": content,
                        }
                    ]
                },
            ]
        )
        [entry] = await fetch_thread_entries(
            client,
            cache,
            "!room",
            "$root",
            limit=10,
            before_event_id="$trigger",
            exclude_event_ids=["$root"],
        )
    assert entry == MatrixEventContext("@alice:example.org", expected)


@pytest.mark.asyncio
async def test_sent_matrix_message_is_available_as_reply_context():
    from plugins.platforms.matrix.reply_context import MatrixEventContext

    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$sent")

    result = await adapter.send("!room:example.org", "hello from the bot")
    cached = await adapter._event_context_cache.resolve(None, "!room:example.org", "$sent")

    assert (result.success, result.message_id, cached) == (
        True, "$sent", MatrixEventContext("@bot:example.org", "hello from the bot"),
    )


@pytest.mark.asyncio
async def test_successful_matrix_edit_updates_cached_reply_target():
    from plugins.platforms.matrix.reply_context import MatrixEventContext

    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(side_effect=[
        "$sent", "$edit", RuntimeError("send failed"),
    ])
    room_id = "!room:example.org"

    sent = await adapter.send(room_id, "first fragment")
    edited = await adapter.edit_message(room_id, sent.message_id, "final answer")
    after_success = await adapter._event_context_cache.resolve(None, room_id, sent.message_id)
    failed = await adapter.edit_message(room_id, sent.message_id, "unpublished")
    after_failure = await adapter._event_context_cache.resolve(None, room_id, sent.message_id)

    assert (
        sent.success, edited.success, failed.success, after_success, after_failure,
    ) == (
        True, True, False,
        MatrixEventContext("@bot:example.org", "final answer"),
        MatrixEventContext("@bot:example.org", "final answer"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("parent_redacted", [False, True])
async def test_matrix_edit_budget_and_cached_parent(transport_outcome, parent_redacted):
    import json

    from gateway.platforms.base import SendResult
    from plugins.platforms.matrix.reply_context import MatrixEventContext

    adapter = _make_adapter()
    room_id = "!room:example.org"
    user_ids = [f"@u{i}:a.b" for i in range(1300)]
    payload = " ".join(user_ids)
    prior = MatrixEventContext("@bot:example.org", "before", media_type="application/test")
    adapter._event_context_cache.store(room_id, "$parent", prior)
    if parent_redacted:
        adapter._event_context_cache.redact(room_id, "$parent")

    cancellation = asyncio.CancelledError()
    effects = {"success": None, "failure": RuntimeError("send failed"), "cancelled": cancellation}
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(
        return_value="$edit", side_effect=effects[transport_outcome],
    )

    result = None
    cancelled = False
    if transport_outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await adapter.edit_message(room_id, "$parent", payload)
        cancelled = raised.value is cancellation
    else:
        result = await adapter.edit_message(room_id, "$parent", payload)

    awaited_call = adapter._client.send_message_event.await_args
    assert awaited_call is not None
    sent_content = awaited_call.args[2]
    sent_bytes = len(json.dumps(sent_content, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    after_edit = await adapter._event_context_cache.resolve(None, room_id, "$parent")
    rejected_store = None
    after_store = after_edit
    if parent_redacted:
        rejected_store = adapter._event_context_cache.store(room_id, "$parent", prior)
        after_store = await adapter._event_context_cache.resolve(None, room_id, "$parent")

    expected_results = {
        "success": SendResult(success=True, message_id="$edit"),
        "failure": SendResult(success=False, error="send failed"),
        "cancelled": None,
    }
    expected_context = prior
    if transport_outcome == "success":
        expected_context = MatrixEventContext("@bot:example.org", payload, media_type="application/test")
    if parent_redacted:
        expected_context = None

    assert {
        "input_within_advertised_budget": len(payload.encode("utf-8")) <= adapter.max_message_length_for_chat(room_id),
        "event_fits_homeserver": sent_bytes * 4 // 3 + 4096 <= 65_536,
        "replacement": sent_content["m.new_content"],
        "relation": sent_content["m.relates_to"],
        "fallback_shortened": sent_content["body"] != f"* {payload}",
        "result": result,
        "same_cancellation": cancelled,
        "cached_parent": after_edit,
        "rejected_store": rejected_store,
        "cached_parent_after_store": after_store,
    } == {
        "input_within_advertised_budget": True,
        "event_fits_homeserver": True,
        "replacement": {"msgtype": "m.text", "body": payload, "m.mentions": {"user_ids": user_ids}},
        "relation": {"rel_type": "m.replace", "event_id": "$parent"},
        "fallback_shortened": True,
        "result": expected_results[transport_outcome],
        "same_cancellation": transport_outcome == "cancelled",
        "cached_parent": expected_context,
        "rejected_store": None,
        "cached_parent_after_store": expected_context,
    }


@pytest.mark.asyncio
async def test_text_reply_to_image_attaches_the_quoted_image():
    from pathlib import Path

    from hermes_constants import get_hermes_home

    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org"]
    )
    adapter._client.get_event = AsyncMock(return_value={
        "sender": "@alice:example.org",
        "content": {"msgtype": "m.image", "body": "photo.png", "url": "mxc://example.org/photo",
                    "info": {"mimetype": "image/png", "size": 12}},
    })
    image_bytes = b"\x89PNG\r\n\x1a\nDATA"
    adapter._client.download_media = AsyncMock(return_value=image_bytes)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$reply", "What is this?",
        {"msgtype": "m.text", "body": "What is this?"},
        {"m.in_reply_to": {"event_id": "$photo"}},
    )

    image = Path(event.media_urls[0])
    assert (event.reply_to_text, event.media_types, image.read_bytes(), image.is_relative_to(get_hermes_home())) == (
        "[image]", ["image/png"], image_bytes, True
    )
    adapter._client.download_media.assert_awaited_once_with("mxc://example.org/photo")


@pytest.mark.asyncio
async def test_image_reply_with_plain_fallback_still_attaches_image(tmp_path):
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org"]
    )
    adapter._client.get_event = AsyncMock(return_value={
        "sender": "@alice:example.org",
        "content": {"msgtype": "m.image", "body": "photo.jpg", "url": "mxc://example.org/photo",
                    "info": {"mimetype": "image/jpeg", "size": 4}},
    })
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg")
    adapter._download_and_cache_media = AsyncMock(return_value=str(image))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$reply",
        "> <@alice:example.org> photo.jpg\n\nWhat is this?",
        {"msgtype": "m.text", "body": "> <@alice:example.org> photo.jpg\n\nWhat is this?"},
        {"m.in_reply_to": {"event_id": "$photo"}},
    )

    assert (event.text, event.reply_to_text, event.media_urls) == (
        "What is this?", "[image]", [str(image)]
    )


@pytest.mark.asyncio
async def test_formatted_reply_fallback_supplies_quote_without_parent_fetch():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(side_effect=Exception("no room state"))
    adapter._client.state_store.has_full_member_list = AsyncMock(return_value=True)
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:example.org", "@alice:example.org"]
    )
    adapter._client.get_event = AsyncMock()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()

    event = await adapter._build_inbound_event(
        "!room:example.org", "@alice:example.org", "$reply", "Continue",
        {"msgtype": "m.text", "body": "Continue", "format": "org.matrix.custom.html", "formatted_body":
         '<mx-reply><blockquote><a>In reply to</a> <a>@alice</a><br/>Earlier '
         '<b>text</b></blockquote></mx-reply>Continue'},
        {"m.in_reply_to": {"event_id": "$parent"}},
    )

    assert (event.text, event.reply_to_text, event.reply_to_author_authorized) == (
        "Continue", "Earlier text", False,
    )
    adapter._client.get_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_check", ["second-resolve", "redaction"])
async def test_cached_reply_target_stays_consistent_while_its_media_file_is_checked(monkeypatch, during_check):
    """The cached media file is checked off the event loop, so another resolve or a redaction can
    land meanwhile: neither may raise, and a redaction always wins."""
    import threading

    from plugins.platforms.matrix import reply_context
    from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache

    checking, release = threading.Event(), threading.Event()

    class _SlowPath:
        def __init__(self, path):
            self.path = path

        def is_file(self):
            checking.set()
            release.wait(timeout=5)
            return during_check == "redaction"  # gone in the race case, so both resolves drop the entry

    monkeypatch.setattr(reply_context, "Path", _SlowPath)
    room_id, event_id = "!room:example.org", "$image"
    cache = MatrixEventContextCache()
    cache.store(room_id, event_id, MatrixEventContext(
        "@alice:example.org", "photo", media_path="/cache/photo.png", is_image=True))

    first = asyncio.create_task(cache.resolve(None, room_id, event_id))
    await asyncio.to_thread(checking.wait, 5)
    if during_check == "second-resolve":
        second = asyncio.create_task(cache.resolve(None, room_id, event_id))
        await asyncio.sleep(0)
    elif during_check == "redaction":
        cache.redact(room_id, event_id)
    release.set()
    results = [await first] + ([await second] if during_check == "second-resolve" else [])

    assert results == [None] * len(results)
