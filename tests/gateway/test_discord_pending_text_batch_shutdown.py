"""Regression guard for Discord text-batch flush during gateway shutdown."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
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

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_cancel_background_tasks_awaits_pending_text_batch_before_clearing():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    flushed = asyncio.Event()

    async def pending_flush():
        await asyncio.sleep(0)
        flushed.set()

    task = asyncio.create_task(pending_flush())
    adapter._pending_text_batch_tasks["chat"] = task
    adapter._pending_text_batches["chat"] = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="group"),
    )

    await adapter.cancel_background_tasks()

    assert flushed.is_set()
    assert task.done()
    assert adapter._pending_text_batch_tasks == {}
    assert adapter._pending_text_batches == {}
