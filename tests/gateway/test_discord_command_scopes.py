"""Discord slash-command scopes through real discord.py serialization."""

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import discord
    from discord.ext import commands

    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    guild, user = json.loads(sys.argv[1])
    app_info = SimpleNamespace(
        guild_integration_config=object() if guild else None,
        user_integration_config=object() if user else None,
    )
    bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
    bot._application = app_info
    bot.tree.sync = AsyncMock(return_value=[])
    os.environ["DISCORD_COMMAND_SYNC_POLICY"] = "bulk"
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = bot
    adapter._register_slash_commands()
    asyncio.run(adapter._run_post_connect_initialization())
    assert bot.tree.sync.await_count == 1
    payloads = [command.to_dict(bot.tree) for command in bot.tree.get_commands()]
    assert payloads
    print(json.dumps(sorted({json.dumps([p["integration_types"], p["contexts"]]) for p in payloads})))
    """
)


@pytest.mark.parametrize(
    ("guild", "user", "expected"),
    [
        # The reported app: Guild Install only. Every command must carry guild install;
        # private_channel is refused by Discord without user install.
        (True, False, [[0], [0, 1]]),
        (True, True, [[0, 1], [0, 1, 2]]),
        (False, True, [[1], [0, 1, 2]]),
        (False, False, None),
    ],
)
def test_registered_commands_follow_app_installation_settings(guild, user, expected):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, f"[{str(guild).lower()}, {str(user).lower()}]"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named 'discord'" in result.stderr:
        pytest.skip("discord.py optional dependency is not installed")
    assert result.returncode == 0, result.stdout + result.stderr
    import json

    rows = [json.loads(row) for row in json.loads(result.stdout.strip().splitlines()[-1])]
    assert rows == [expected if expected is not None else [None, None]]
