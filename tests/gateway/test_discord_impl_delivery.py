from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest


def _load_delivery_module():
    class FakeIntents:
        @staticmethod
        def default():
            return SimpleNamespace()

    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = FakeIntents
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock()
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
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.delivery"))


delivery = _load_delivery_module()


@pytest.mark.asyncio
async def test_resolve_channel_returns_cached_channel():
    channel = SimpleNamespace(id=123)
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await delivery.resolve_channel(client, "123")

    assert result is channel
    client.fetch_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_channel_fetches_when_not_cached():
    channel = SimpleNamespace(id=456)
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=channel),
    )

    result = await delivery.resolve_channel(client, "456")

    assert result is channel
    client.fetch_channel.assert_awaited_once_with(456)


@pytest.mark.asyncio
async def test_resolve_channel_returns_none_when_missing():
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=None),
    )

    result = await delivery.resolve_channel(client, "789")

    assert result is None


@pytest.mark.asyncio
async def test_send_text_message_sends_normally():
    message = SimpleNamespace(id=1)
    channel = SimpleNamespace(send=AsyncMock(return_value=message))

    result = await delivery.send_text_message(channel, "hello")

    assert result is message
    channel.send.assert_awaited_once_with(content="hello", reference=None)


@pytest.mark.asyncio
async def test_send_text_message_retries_without_reference_for_system_message():
    message = SimpleNamespace(id=2)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return message

    reference = SimpleNamespace(id=99)
    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))

    result = await delivery.send_text_message(channel, "hello", reference=reference)

    assert result is message
    assert send_calls == [
        {"content": "hello", "reference": reference},
        {"content": "hello", "reference": None},
    ]


@pytest.mark.asyncio
async def test_send_file_attachment_succeeds(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    sent_message = SimpleNamespace(id=42)
    channel = SimpleNamespace(send=AsyncMock(return_value=sent_message))
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )
    file_cls = MagicMock()
    monkeypatch.setattr(delivery.discord, "File", file_cls)

    result = await delivery.send_file_attachment(client, "123", str(file_path), caption="cap")

    assert result.success is True
    assert result.message_id == "42"
    channel.send.assert_awaited_once()
    assert file_cls.call_args.kwargs["filename"] == "sample.txt"


@pytest.mark.asyncio
async def test_send_file_attachment_returns_not_connected_error():
    result = await delivery.send_file_attachment(None, "123", "/tmp/missing.txt")

    assert result.success is False
    assert result.error == "Not connected"


@pytest.mark.asyncio
async def test_send_file_attachment_returns_channel_not_found_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=None),
    )

    result = await delivery.send_file_attachment(client, "123", str(file_path))

    assert result.success is False
    assert result.error == "Channel 123 not found"
