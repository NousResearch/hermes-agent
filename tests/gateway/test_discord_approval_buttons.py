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
    discord_mod.ui = SimpleNamespace(
        View=type("View", (), {"__init__": lambda self, *a, **k: None}),
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        green=1,
        grey=2,
        blurple=3,
        red=4,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
    )
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

from gateway.platforms import discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_send_exec_approval_includes_plain_text_command_preview(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    sent_msg = SimpleNamespace(id=1234)
    channel = SimpleNamespace(send=AsyncMock(return_value=sent_msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    view = object()
    monkeypatch.setattr(discord_platform, "ExecApprovalView", MagicMock(return_value=view))

    result = await adapter.send_exec_approval(
        chat_id="555",
        command="rm -rf /important",
        session_key="agent:main:discord:group:555:999",
        description="dangerous deletion",
    )

    assert result.success is True
    assert result.message_id == "1234"
    channel.send.assert_awaited_once()
    kwargs = channel.send.call_args.kwargs
    assert kwargs["view"] is view
    assert "embed" not in kwargs
    assert "rm -rf /important" in kwargs["content"]
    assert "dangerous deletion" in kwargs["content"]
