from datetime import datetime
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest


def _load_messaging_module():
    class FakeIntents:
        @staticmethod
        def default():
            return SimpleNamespace()

    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = FakeIntents
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        danger=3,
        green=1,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
        Decoder=MagicMock,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    sys.modules["discord"] = discord_mod
    ext_mod = ModuleType("discord.ext")
    commands_mod = ModuleType("discord.ext.commands")
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod
    importlib.reload(importlib.import_module("gateway.platforms.discord_impl.delivery"))
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.messaging"))


messaging = _load_messaging_module()


def _make_client(channel):
    return SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=None if channel is None else channel),
    )


class _AsyncItemsIterator:
    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def test_normalize_edit_content_truncates_to_limit():
    result = messaging.normalize_edit_content(
        "a" * 25,
        format_message=lambda value: value.upper(),
        max_message_length=10,
    )

    assert result == "AAAAAAA..."


@pytest.mark.asyncio
async def test_edit_message_returns_success():
    message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    client = _make_client(channel)

    result = await messaging.edit_message(
        client,
        "123",
        "456",
        "hello",
        format_message=lambda value: f"**{value}**",
        max_message_length=2000,
    )

    assert result.success is True
    assert result.message_id == "456"
    channel.fetch_message.assert_awaited_once_with(456)
    message.edit.assert_awaited_once_with(content="**hello**")


@pytest.mark.asyncio
async def test_edit_message_returns_missing_channel_error():
    result = await messaging.edit_message(
        _make_client(None),
        "123",
        "456",
        "hello",
    )

    assert result.success is False
    assert result.error == "Channel 123 not found"


@pytest.mark.asyncio
async def test_edit_message_returns_fetch_error():
    channel = SimpleNamespace(fetch_message=AsyncMock(side_effect=RuntimeError("missing message")))
    client = _make_client(channel)

    result = await messaging.edit_message(client, "123", "456", "hello")

    assert result.success is False
    assert result.error == "missing message"


@pytest.mark.asyncio
async def test_edit_message_truncates_oversized_content():
    message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    client = _make_client(channel)

    result = await messaging.edit_message(
        client,
        "123",
        "456",
        "a" * 20,
        max_message_length=10,
    )

    assert result.success is True
    message.edit.assert_awaited_once_with(content="aaaaaaa...")


@pytest.mark.asyncio
async def test_list_threads_returns_active_threads():
    active_thread = SimpleNamespace(
        id=10,
        name="alpha",
        parent=SimpleNamespace(id=5, name="general"),
        guild=SimpleNamespace(id=1, name="Hermes"),
        archived=False,
        locked=False,
        message_count=3,
        member_count=2,
    )
    channel = SimpleNamespace(threads=[active_thread])

    result = await messaging.list_threads(_make_client(channel), "123")

    assert result == [
        {
            "id": "10",
            "name": "alpha",
            "parent_id": "5",
            "parent_name": "general",
            "guild_id": "1",
            "guild_name": "Hermes",
            "archived": False,
            "locked": False,
            "message_count": 3,
            "member_count": 2,
        }
    ]


@pytest.mark.asyncio
async def test_list_threads_includes_archived_threads_when_requested():
    archived_thread = SimpleNamespace(
        id=11,
        name="archive",
        parent=SimpleNamespace(id=5, name="general"),
        guild=SimpleNamespace(id=1, name="Hermes"),
        archived=True,
        locked=True,
        message_count=9,
        member_count=4,
    )
    channel = SimpleNamespace(
        threads=[],
        archived_threads=MagicMock(return_value=_AsyncItemsIterator([archived_thread])),
    )

    result = await messaging.list_threads(_make_client(channel), "123", include_archived=True, limit=25)

    assert result[0]["id"] == "11"
    channel.archived_threads.assert_called_once_with(
        private=False,
        joined=False,
        limit=25,
        before=None,
    )


@pytest.mark.asyncio
async def test_reply_in_thread_rejects_non_thread_channel():
    channel = SimpleNamespace(parent=None)

    result = await messaging.reply_in_thread(_make_client(channel), "123", "hello")

    assert result.success is False
    assert result.error == "Channel 123 is not a thread"


@pytest.mark.asyncio
async def test_reply_in_thread_sends_chunks():
    sent_message = SimpleNamespace(id=55)
    thread = SimpleNamespace(
        parent=SimpleNamespace(id=5),
        fetch_message=AsyncMock(return_value=SimpleNamespace(id=99)),
    )
    send_text_message = AsyncMock(return_value=sent_message)

    result = await messaging.reply_in_thread(
        _make_client(thread),
        "123",
        "hello",
        reply_to="99",
        format_message=lambda value: value.upper(),
        truncate_message=lambda value, _max_len: [value[:3], value[3:]],
        send_text_message=send_text_message,
    )

    assert result.success is True
    assert result.message_id == "55"
    assert send_text_message.await_count == 2
    first_call = send_text_message.await_args_list[0]
    second_call = send_text_message.await_args_list[1]
    assert first_call.args[0] is thread
    assert first_call.args[1] == "HEL"
    assert first_call.kwargs["reference"].id == 99
    assert second_call.args[1] == "LO"
    assert second_call.kwargs["reference"] is None


@pytest.mark.asyncio
async def test_list_pins_serializes_messages():
    author = SimpleNamespace(id=42, name="Jezza", display_name="Jezza", bot=False)
    pinned = SimpleNamespace(
        id=7,
        author=author,
        content="important",
        created_at=datetime(2026, 3, 18, 12, 0, 0),
        attachments=[],
        reference=None,
    )
    channel = SimpleNamespace(pins=MagicMock(return_value=_AsyncItemsIterator([pinned])))

    result = await messaging.list_pins(_make_client(channel), "123")

    assert result == [
        {
            "id": "7",
            "author_id": "42",
            "author_name": "Jezza",
            "content": "important",
            "timestamp": "2026-03-18T12:00:00",
            "is_bot": False,
            "attachments": [],
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_pin_and_unpin_message_call_message_methods():
    message = SimpleNamespace(pin=AsyncMock(), unpin=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    client = _make_client(channel)

    pin_result = await messaging.pin_message(client, "123", "456", reason="keep")
    unpin_result = await messaging.unpin_message(client, "123", "456", reason="drop")

    assert pin_result.success is True
    assert unpin_result.success is True
    message.pin.assert_awaited_once_with(reason="keep")
    message.unpin.assert_awaited_once_with(reason="drop")
