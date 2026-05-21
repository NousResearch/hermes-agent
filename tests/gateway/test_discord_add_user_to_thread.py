from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


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

from gateway.platforms.discord import DiscordAdapter  # noqa: E402

@pytest.mark.asyncio
async def test_create_thread_adds_user():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    # mock thread
    thread = SimpleNamespace(
        id=123,
        name="test-thread",
        add_user=AsyncMock(),
        send=AsyncMock(),
    )

    # mock parent channel
    parent_channel = SimpleNamespace(
        create_thread=AsyncMock(return_value=thread)
    )

    # mock interaction
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"),
    )

    # patch methods
    adapter._resolve_interaction_channel = AsyncMock(return_value="dummy_channel")
    adapter._thread_parent_channel = lambda _: parent_channel

    result = await adapter._create_thread(
        interaction,
        name="test-thread"
    )

    assert result["success"] is True
    thread.add_user.assert_awaited_once_with(interaction.user)
	
@pytest.mark.asyncio
async def test_create_thread_fallback_adds_user():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread = SimpleNamespace(
        id=456,
        name="fallback-thread",
        add_user=AsyncMock(),
    )

    seed_msg = SimpleNamespace(
        create_thread=AsyncMock(return_value=thread)
    )

    parent_channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=Exception("fail")),
        send=AsyncMock(return_value=seed_msg),
    )

    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"),
    )

    adapter._resolve_interaction_channel = AsyncMock(return_value="dummy_channel")
    adapter._thread_parent_channel = lambda _: parent_channel

    result = await adapter._create_thread(
        interaction,
        name="fallback-thread"
    )

    assert result["success"] is True
    thread.add_user.assert_awaited_once_with(interaction.user)
