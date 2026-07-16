"""Discord slash-command scope defaults through real discord.py serialization."""

import subprocess
import sys
import textwrap


def test_bot_defaults_serialize_guild_and_user_command_scopes():
    """A real Bot/tree applies Hermes scope defaults to registered commands."""
    probe = textwrap.dedent(
        """
        import discord
        from discord.ext import commands

        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import (
            DiscordAdapter,
            _discord_app_command_scope_kwargs,
        )

        bot = commands.Bot(
            command_prefix="!",
            intents=discord.Intents.default(),
            **_discord_app_command_scope_kwargs(),
        )
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
        adapter._client = bot
        adapter._register_slash_commands()

        payloads = [command.to_dict(bot.tree) for command in bot.tree.get_commands()]
        assert payloads
        assert {tuple(p["integration_types"]) for p in payloads} == {(0, 1)}
        assert {tuple(p["contexts"]) for p in payloads} == {(0, 1, 2)}
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
