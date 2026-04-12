import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.MessageType = SimpleNamespace(default="default", reply="reply")
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


def _make_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter.handle_message = AsyncMock()
    adapter._client = MagicMock()
    adapter._client.user = object()
    return adapter


def _base_message():
    guild = SimpleNamespace(name="Guild")
    channel = SimpleNamespace(id=123, name="general", guild=guild, topic=None)
    author = SimpleNamespace(id=1, display_name="yake", name="yake", bot=False)
    return SimpleNamespace(
        id=999,
        channel=channel,
        author=author,
        content="current reply",
        attachments=[],
        created_at=None,
        mentions=[],
        type="reply",
    )


@pytest.mark.asyncio
async def test_handle_message_uses_resolved_reply_text():
    adapter = _make_adapter()
    message = _base_message()
    referenced = SimpleNamespace(
        content="quoted text from resolved message",
        attachments=[],
        author=SimpleNamespace(display_name="Alice"),
    )
    message.reference = SimpleNamespace(message_id=111, resolved=referenced)

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "111"
    assert event.reply_to_text == "quoted text from resolved message"


@pytest.mark.asyncio
async def test_handle_message_fetches_reply_text_when_resolved_missing():
    adapter = _make_adapter()
    message = _base_message()
    fetched = SimpleNamespace(
        content="fetched quoted text",
        attachments=[],
        author=SimpleNamespace(display_name="Bob"),
    )
    async def _fetch_message(message_id):
        assert message_id == 222
        return fetched
    message.channel.fetch_message = _fetch_message
    message.reference = SimpleNamespace(message_id=222, resolved=None, channel_id=123)

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "222"
    assert event.reply_to_text == "fetched quoted text"


@pytest.mark.asyncio
async def test_handle_message_builds_reply_text_from_attachments_when_no_text():
    adapter = _make_adapter()
    message = _base_message()
    attachment = SimpleNamespace(filename="design.png", content_type="image/png")
    referenced = SimpleNamespace(
        content="   ",
        attachments=[attachment],
        author=SimpleNamespace(display_name="Carol"),
    )
    message.reference = SimpleNamespace(message_id=333, resolved=referenced)

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "333"
    assert "design.png" in (event.reply_to_text or "")
    assert "attachment" in (event.reply_to_text or "")
