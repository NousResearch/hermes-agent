import json
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

    guild, user, refresh_fails = json.loads(sys.argv[1])
    app_info = SimpleNamespace(
        guild_integration_config=object() if guild else None,
        user_integration_config=object() if user else None,
    )
    bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
    # Login-time snapshot predates a settings change; the sync must use the fresh one.
    bot._application = SimpleNamespace(guild_integration_config=None, user_integration_config=object())
    bot.application_info = AsyncMock(
        side_effect=RuntimeError("503") if refresh_fails else None, return_value=app_info
    )
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
    ("guild", "user", "refresh_fails", "expected"),
    [
        # Discord refuses private_channel without user install.
        (True, False, False, [[0], [0, 1]]),
        (True, True, False, [[0, 1], [0, 1, 2]]),
        (False, True, False, [[1], [0, 1, 2]]),
        (False, False, False, None),
        # A failed refresh falls back to the login-time snapshot (user install only here).
        (True, False, True, [[1], [0, 1, 2]]),
    ],
)
def test_registered_commands_follow_app_installation_settings(guild, user, refresh_fails, expected):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps([guild, user, refresh_fails])],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named 'discord'" in result.stderr:
        pytest.skip("discord.py optional dependency is not installed")
    assert result.returncode == 0, result.stdout + result.stderr

    rows = [json.loads(row) for row in json.loads(result.stdout.strip().splitlines()[-1])]
    assert rows == [expected if expected is not None else [None, None]]
