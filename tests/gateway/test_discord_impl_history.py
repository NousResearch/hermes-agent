from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest

def _install_discord_mock():
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


def _load_history_module():
    _install_discord_mock()
    importlib.reload(importlib.import_module("gateway.platforms.discord_impl.delivery"))
    importlib.reload(importlib.import_module("gateway.platforms.discord_impl.permissions"))
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.history"))


history = _load_history_module()


class FakeHistoryIterator:
    def __init__(self, messages):
        self._messages = list(messages)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        message = self._messages[self._index]
        self._index += 1
        return message


class FakeHistoryChannel:
    def __init__(self, messages, readable=True):
        self.id = 123
        self.name = "general"
        self.guild = SimpleNamespace(me=SimpleNamespace(id=999), name="Hermes")
        self._messages = list(messages)
        self.last_history_kwargs = None
        self.history_calls = 0
        self.permissions_for = MagicMock(
            return_value=SimpleNamespace(
                view_channel=readable,
                read_message_history=readable,
                send_messages=True,
                attach_files=True,
                embed_links=True,
                add_reactions=True,
                manage_threads=False,
                create_public_threads=False,
                create_private_threads=False,
            )
        )

    def history(self, *, limit, before=None, after=None):
        self.history_calls += 1
        self.last_history_kwargs = {
            "limit": limit,
            "before": before,
            "after": after,
        }

        filtered = list(self._messages)
        if before is not None:
            filtered = [message for message in filtered if int(message.id) < int(before.id)]
        if after is not None:
            filtered = [message for message in filtered if int(message.id) > int(after.id)]
        return FakeHistoryIterator(filtered[:limit])


def _make_message(
    message_id,
    *,
    content,
    author_id="42",
    author_name="Jezza",
    is_bot=False,
    timestamp=None,
    attachments=None,
    reply_to=None,
):
    if timestamp is None:
        timestamp = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
    reference = None if reply_to is None else SimpleNamespace(message_id=reply_to)
    return SimpleNamespace(
        id=message_id,
        author=SimpleNamespace(id=author_id, name=author_name, display_name=author_name, bot=is_bot),
        content=content,
        created_at=timestamp,
        attachments=[SimpleNamespace(url=url) for url in (attachments or [])],
        reference=reference,
    )


def _make_client(channel):
    return SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=None if channel is None else channel),
        user=SimpleNamespace(id=999),
    )


@pytest.mark.asyncio
async def test_fetch_history_returns_messages_and_clamps_limits():
    messages = [
        _make_message(message_id, content=f"message {message_id}")
        for message_id in range(600, 0, -1)
    ]
    channel = FakeHistoryChannel(messages)
    client = _make_client(channel)

    result_min = await history.fetch_history(client, "123", limit=0)

    assert len(result_min) == 1
    assert channel.last_history_kwargs["limit"] == 1

    result_max = await history.fetch_history(client, "123", limit=999)

    assert len(result_max) == 500
    assert channel.last_history_kwargs["limit"] == 500
    assert result_max[0].id == "600"
    assert result_max[-1].id == "101"


@pytest.mark.asyncio
async def test_fetch_history_applies_before_and_after_filters():
    messages = [
        _make_message(message_id, content=f"message {message_id}")
        for message_id in (105, 104, 103, 102, 101, 100)
    ]
    channel = FakeHistoryChannel(messages)
    client = _make_client(channel)

    result = await history.fetch_history(client, "123", limit=10, before="104", after="101")

    assert [message.id for message in result] == ["103", "102"]
    assert channel.last_history_kwargs["before"].id == 104
    assert channel.last_history_kwargs["after"].id == 101


@pytest.mark.asyncio
async def test_fetch_history_returns_empty_for_empty_channel():
    channel = FakeHistoryChannel([])
    client = _make_client(channel)

    result = await history.fetch_history(client, "123")

    assert result == []


@pytest.mark.asyncio
async def test_fetch_history_returns_empty_without_read_access():
    channel = FakeHistoryChannel([_make_message(1, content="hidden")], readable=False)
    client = _make_client(channel)

    result = await history.fetch_history(client, "123")

    assert result == []
    assert channel.history_calls == 0


@pytest.mark.asyncio
async def test_fetch_history_returns_empty_for_missing_channel():
    client = _make_client(None)

    result = await history.fetch_history(client, "123")

    assert result == []


def test_history_message_dataclass_fields():
    message = history.HistoryMessage(
        id="1",
        author_id="42",
        author_name="Jezza",
        content="hello",
        timestamp="2026-03-18T12:00:00+00:00",
        is_bot=False,
        attachments=["https://example.com/file.png"],
        reply_to="0",
    )

    assert message.id == "1"
    assert message.author_id == "42"
    assert message.author_name == "Jezza"
    assert message.content == "hello"
    assert message.timestamp == "2026-03-18T12:00:00+00:00"
    assert message.is_bot is False
    assert message.attachments == ["https://example.com/file.png"]
    assert message.reply_to == "0"


@pytest.mark.asyncio
async def test_search_history_filters_by_case_insensitive_query():
    channel = FakeHistoryChannel(
        [
            _make_message(10, content="Hermes status update"),
            _make_message(9, content="unrelated"),
            _make_message(8, content="another hermes note"),
        ]
    )
    client = _make_client(channel)

    result = await history.search_history(client, "123", "HeRmEs", limit=5)

    assert [message.id for message in result] == ["10", "8"]


@pytest.mark.asyncio
async def test_search_history_filters_by_author_id():
    channel = FakeHistoryChannel(
        [
            _make_message(10, content="deploy update", author_id="1", author_name="Ada"),
            _make_message(9, content="deploy update", author_id="2", author_name="Linus"),
            _make_message(8, content="deploy update", author_id="1", author_name="Ada"),
        ]
    )
    client = _make_client(channel)

    result = await history.search_history(client, "123", "deploy", limit=5, author_id="1")

    assert [message.id for message in result] == ["10", "8"]
    assert all(message.author_id == "1" for message in result)


@pytest.mark.asyncio
async def test_search_history_returns_empty_when_no_matches():
    channel = FakeHistoryChannel(
        [
            _make_message(10, content="hello world"),
            _make_message(9, content="still nothing"),
        ]
    )
    client = _make_client(channel)

    result = await history.search_history(client, "123", "needle")

    assert result == []


@pytest.mark.asyncio
async def test_search_history_returns_empty_for_missing_channel():
    client = _make_client(None)

    result = await history.search_history(client, "123", "needle")

    assert result == []


@pytest.mark.asyncio
async def test_search_history_finds_match_beyond_old_limit_multiplier_window():
    channel = FakeHistoryChannel(
        [
            _make_message(message_id, content="needle" if message_id == 130 else f"message {message_id}")
            for message_id in range(250, 0, -1)
        ]
    )
    client = _make_client(channel)

    result = await history.search_history(client, "123", "needle", limit=1)

    assert [message.id for message in result] == ["130"]


@pytest.mark.asyncio
async def test_search_history_finds_author_filtered_match_beyond_old_limit_multiplier_window():
    messages = []
    for message_id in range(250, 0, -1):
        if 250 >= message_id >= 150:
            messages.append(_make_message(message_id, content="deploy update", author_id="2"))
        elif message_id == 130:
            messages.append(_make_message(message_id, content="deploy update", author_id="1"))
        else:
            messages.append(_make_message(message_id, content=f"message {message_id}", author_id="2"))

    channel = FakeHistoryChannel(messages)
    client = _make_client(channel)

    result = await history.search_history(client, "123", "deploy", limit=1, author_id="1")

    assert [message.id for message in result] == ["130"]
