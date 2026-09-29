"""Inbound location contracts, including the source adaptation of #66236."""

import logging
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter.time.time", lambda: 1_700_000_000.0
    )
    instance = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "user_id": "@bot:example.org",
                "require_mention": False,
                "auto_thread": False,
            },
        )
    )
    instance._text_batch_delay_seconds = 0
    instance._startup_ts = 1_700_000_000.0
    instance.handle_message = AsyncMock()
    instance._resolve_room_identity = AsyncMock(
        return_value=SimpleNamespace(
            display_name="Test Room",
            room_topic=None,
            server_name="example.org",
            chat_type="group",
        )
    )
    return instance


def location_event(content):
    return SimpleNamespace(
        sender="@alice:example.org",
        event_id="$location",
        room_id="!room:example.org",
        timestamp=1_700_000_000_000,
        content={"msgtype": "m.location", "body": "Location", **content},
    )


DISPATCH_TIME = datetime(2026, 1, 1)


def dispatched_messages(adapter):
    return [
        replace(call.args[0], timestamp=DISPATCH_TIME)
        for call in adapter.handle_message.await_args_list
    ]


def room_message(event, text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        metadata={"matrix_requires_mention": True},
        raw_message=event.content,
        message_id=event.event_id,
        user_id=event.sender,
        user_name="alice",
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id=event.room_id,
            chat_name="Test Room",
            chat_type="group",
            user_id=event.sender,
            user_name="alice",
            guild_id="example.org",
            message_id=event.event_id,
            source_permalink=f"https://matrix.to/#/{event.room_id}/{event.event_id}?via=example.org",
        ),
        timestamp=DISPATCH_TIME,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "text"),
    [
        ({"geo_uri": "geo:40.5694,9.7845"}, "📍 Location: 40.5694, 9.7845"),
        ({"geo_uri": "geo:-90,180"}, "📍 Location: -90.0, 180.0"),
        (
            {"geo_uri": "GEO:90,-180;CRS=WGS84;U=0"},
            "📍 Location: 90.0, -180.0; uncertainty: 0.0 m",
        ),
        (
            {"geo_uri": "geo:51.5,-0.1,-12.5;u=35.5;foo=bar%20baz;flag"},
            "📍 Location: 51.5, -0.1; altitude: -12.5 m; uncertainty: 35.5 m",
        ),
        ({"geo_uri": "geo:0,-0,0"}, "📍 Location: 0.0, -0.0; altitude: 0.0 m"),
        ({"geo_uri": "geo:001,2"}, "📍 Location: 1.0, 2.0"),
        ({"geo_uri": "geo:-090,0000180.0"}, "📍 Location: -90.0, 180.0"),
        (
            {"geo_uri": "geo:40.5694,9.7845", "body": "Posizione"},
            "📍 Location: 40.5694, 9.7845",
        ),
        (
            {"geo_uri": "geo:40.5694,9.7845", "body": "Meeting point"},
            "📍 Location: 40.5694, 9.7845 (Meeting point)",
        ),
        (
            {
                "geo_uri": "geo:1,2;u=3",
                "body": "Location geo:1,2;u=3 at 2026-09-29T10:00:00.000Z",
                "org.matrix.msc3488.location": {"uri": "geo:1,2;u=3"},
            },
            "📍 Location: 1.0, 2.0; uncertainty: 3.0 m",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "body": "User Location geo:1,2 at 2026-09-29T10:00:00.000Z",
            },
            "📍 Location: 1.0, 2.0",
        ),
        (
            {
                "geo_uri": "geo:1,2;u=3",
                "body": "Location was shared at geo:1,2;u=3",
            },
            "📍 Location: 1.0, 2.0; uncertainty: 3.0 m",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "body": "geo:1,2",
                "org.matrix.msc3488.location": {
                    "uri": "geo:1,2",
                    "description": "geo:1,2",
                },
            },
            "📍 Location: 1.0, 2.0",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "body": "Meeting point",
                "org.matrix.msc3488.location": {"description": "geo:1,2"},
            },
            "📍 Location: 1.0, 2.0 (Meeting point)",
        ),
        (
            {
                "geo_uri": "geo:40.5694,9.7845",
                "org.matrix.msc3488.location": {"description": "Current position"},
            },
            "📍 Location: 40.5694, 9.7845 (Current position)",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "org.matrix.msc3488.location": {
                    "uri": "geo:40.5694,9.7845",
                    "description": "Current position",
                },
            },
            "📍 Location: 40.5694, 9.7845 (Current position)",
        ),
        (
            {
                "org.matrix.msc3488.location": {"uri": "geo:40.5694,9.7845"},
                "body": None,
            },
            "📍 Location: 40.5694, 9.7845",
        ),
        (
            {"geo_uri": "geo:1,2", "org.matrix.msc3488.location": None, "body": 42},
            "📍 Location: 1.0, 2.0",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "org.matrix.msc3488.location": {"description": ["invalid"]},
                "body": "Meeting point",
            },
            "📍 Location: 1.0, 2.0 (Meeting point)",
        ),
        (
            {
                "geo_uri": "geo:1,2",
                "org.matrix.msc3488.location": {"description": ""},
                "body": "Meeting point",
            },
            "📍 Location: 1.0, 2.0 (Meeting point)",
        ),
    ],
)
async def test_location_reaches_text_path_with_original_identity(
    adapter, content, text
):
    event = location_event(content)
    event.content["m.relates_to"] = {
        "rel_type": "m.thread",
        "event_id": "$root",
        "m.in_reply_to": {"event_id": "$reply"},
    }
    original = deepcopy(event.content)

    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    message = adapter.handle_message.await_args.args[0]
    assert message == MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        raw_message=original,
        message_id=event.event_id,
        user_id=event.sender,
        user_name="alice",
        reply_to_message_id="$reply",
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id=event.room_id,
            chat_name="Test Room",
            chat_type="group",
            user_id=event.sender,
            user_name="alice",
            thread_id="$root",
            guild_id="example.org",
            parent_chat_id=event.room_id,
            message_id=event.event_id,
            source_permalink=f"https://matrix.to/#/{event.room_id}/{event.event_id}?via=example.org",
        ),
        timestamp=message.timestamp,
    )
    assert event.content == original


@pytest.mark.asyncio
@pytest.mark.parametrize("html_fallback", [False, True])
@pytest.mark.parametrize(
    ("label", "separator"),
    [
        ("", ""),
        ("", "\n"),
        ("", "\n\n"),
        ("Location", "\n\n"),
        ("Meeting point @file:current.txt", "\n\n"),
    ],
)
@pytest.mark.parametrize(
    "description", [None, "", ["invalid"], "> MSC label @file:current.txt"]
)
async def test_location_reply_keeps_quoted_references_out_of_current_text(
    adapter, html_fallback, label, separator, description
):
    adapter._require_mention = True
    quoted_text = "@file:notes.txt\n@skill:previous-context"
    event = location_event({
        "geo_uri": "geo:1,2",
        "body": (
            "> <@bot:example.org> @file:notes.txt\n"
            f"> @skill:previous-context{separator}{label}"
        ),
        "org.matrix.msc3488.location": {"description": description},
        "m.relates_to": {
            "rel_type": "m.thread",
            "event_id": "$root",
            "m.in_reply_to": {"event_id": "$reply"},
        },
    })
    if html_fallback:
        event.content.update({
            "format": "org.matrix.custom.html",
            "formatted_body": (
                '<mx-reply><blockquote><a href="https://matrix.to/#/!room:example.org/$reply">'
                'In reply to</a> <a href="https://matrix.to/#/@bot:example.org">Bot</a>'
                f"<br>@file:notes.txt<br>@skill:previous-context</blockquote></mx-reply>{label}"
            ),
        })
    original = deepcopy(event.content)
    current_label = (
        description if isinstance(description, str) and description else label
    )
    text = "📍 Location: 1.0, 2.0"
    if current_label and current_label != "Location":
        text += f" ({current_label})"

    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    message = adapter.handle_message.await_args.args[0]
    assert message == MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        raw_message=original,
        message_id=event.event_id,
        user_id=event.sender,
        user_name="alice",
        reply_to_message_id="$reply",
        reply_to_text=quoted_text,
        reply_to_author_id="@bot:example.org",
        reply_to_author_name="bot",
        reply_to_author_authorized=False,
        metadata={"matrix_requires_mention": True},
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id=event.room_id,
            chat_name="Test Room",
            chat_type="group",
            user_id=event.sender,
            user_name="alice",
            thread_id="$root",
            guild_id="example.org",
            parent_chat_id=event.room_id,
            message_id=event.event_id,
            source_permalink=f"https://matrix.to/#/{event.room_id}/{event.event_id}?via=example.org",
        ),
        timestamp=message.timestamp,
    )
    assert event.content == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        None,
        42,
        {},
        "",
        "not-a-geo-uri",
        "geo:bad,coords",
        "geo:nan,1",
        "geo:1,inf",
        "geo:91,0",
        "geo:-90.1,0",
        "geo:0,181",
        "geo:0,-180.1",
        "geo:1",
        "geo:1,2,3,4",
        "geo:1,2,",
        "geo:1,2,nan",
        "geo:1,2," + "9" * 400,
        "geo:1,2;crs=other",
        "geo:1,2;u=-1",
        "geo:1,2;u=nan",
        "geo:1,2;u=",
        "geo:1,2;u=" + "9" * 400,
        "geo:1,2;u=1;u=2",
        "geo:1,2;crs=wgs84;CRS=wgs84",
        "geo:1,2;u=1;crs=wgs84",
        "geo:1,2;foo=bar;u=1",
        "geo:1,2;foo=bar;crs=wgs84",
        "geo:+1,2",
        "geo:1e1,2",
        "geo:.1,2",
        "geo:1.,2",
        "geo:1, 2",
        "geo:1,2\n",
        "geo:0091,0",
        "geo:0,-000180.1",
        "geo:1,2;foo=",
        "geo:1,2;foo=%xy",
        "geo:1,2;",
        "geo:1,2?z=10",
        "geo:1,2#map",
    ],
)
async def test_invalid_location_is_logged_and_not_dispatched(adapter, caplog, uri):
    modern = location_event({
        "geo_uri": "geo:1,2",
        "org.matrix.msc3488.location": {"uri": uri},
    })
    modern.event_id = "$msc-location"

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.matrix.adapter"):
        await adapter._on_room_message(location_event({"geo_uri": uri}))
        await adapter._on_room_message(modern)

    adapter.handle_message.assert_not_awaited()
    assert [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if "invalid location" in record.getMessage()
    ] == [
        ("DEBUG", "Matrix: ignoring invalid location $location in !room:example.org"),
        (
            "DEBUG",
            "Matrix: ignoring invalid location $msc-location in !room:example.org",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "accepted"),
    [
        ({}, False),
        ({"body": "/stop"}, False),
        (
            {"org.matrix.msc3488.location": {"uri": "geo:1,2", "description": "@bot"}},
            False,
        ),
        ({"m.mentions": {"user_ids": ["@other:example.org"]}}, False),
        ({"m.mentions": {"user_ids": ["@bot:example.org"]}}, True),
        ({"body": "@bot:example.org Location"}, True),
        (
            {
                "formatted_body": '<a href="https://matrix.to/#/@bot:example.org">Bot</a>'
            },
            True,
        ),
        ({"m.relates_to": {"rel_type": "m.replace", "event_id": "$old"}}, False),
        ({"org.matrix.msc3488.location": {"uri": "geo:91,0"}}, False),
        ({"org.matrix.msc3488.location": {"uri": None}}, False),
        ({"org.matrix.msc3488.location": {"uri": 42}}, False),
    ],
)
async def test_location_keeps_mention_and_relation_gates(adapter, changes, accepted):
    adapter._require_mention = True
    event = location_event({"geo_uri": "geo:1,2", **changes})

    await adapter._on_room_message(event)

    assert adapter.handle_message.await_count == int(accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "text"),
    [
        ("@bot:example.org Meeting point", "📍 Location: 1.0, 2.0 (Meeting point)"),
        ("Meeting point @bot", "📍 Location: 1.0, 2.0 (Meeting point)"),
        ("@bot:example.org Location", "📍 Location: 1.0, 2.0"),
        ("@bot:example.org", "📍 Location: 1.0, 2.0"),
    ],
)
async def test_location_label_drops_bot_mention_like_text(adapter, body, text):
    adapter._require_mention = True
    event = location_event({"geo_uri": "geo:1,2", "body": body})

    await adapter._on_room_message(event)

    assert dispatched_messages(adapter) == [room_message(event, text)]


@pytest.mark.asyncio
async def test_location_bare_mention_does_not_claim_parked_voice(adapter):
    adapter._require_mention = True
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/voice.ogg")
    adapter._background_read_receipt = MagicMock()
    voice = SimpleNamespace(
        sender="@alice:example.org",
        event_id="$voice",
        room_id="!room:example.org",
        timestamp=1_700_000_000_000,
        content={
            "msgtype": "m.audio",
            "body": "voice message",
            "url": "mxc://example.org/voice",
            "info": {"mimetype": "audio/ogg"},
            "org.matrix.msc3245.voice": {},
            "m.mentions": {},
        },
    )
    location = location_event({
        "geo_uri": "geo:1,2",
        "body": "@bot:example.org",
        "m.mentions": {"user_ids": ["@bot:example.org"]},
    })

    await adapter._on_room_message(voice)
    await adapter._on_room_message(location)

    assert dispatched_messages(adapter) == [
        room_message(location, "📍 Location: 1.0, 2.0")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate", ["self", "bridge", "ignored_sender", "room", "old", "duplicate"]
)
async def test_location_keeps_intake_gates(adapter, gate):
    import re

    event = location_event({"geo_uri": "geo:1,2"})
    if gate == "self":
        event.sender = "@Bot:Example.ORG"
    if gate == "bridge":
        event.sender = "@_bridge:example.org"
    if gate == "ignored_sender":
        adapter._ignored_user_patterns = [re.compile("alice")]
    if gate == "room":
        adapter._allowed_room_ids = {"!other:example.org"}
    if gate == "old":
        event.timestamp -= 100_000
    if gate == "duplicate":
        adapter._is_duplicate_event(event.event_id)

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
