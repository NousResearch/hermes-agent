import json
import subprocess
import sys
import textwrap

import pytest

# Real discord.py in a subprocess: other gateway tests install a fake ``discord`` module.
_PROBE = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import discord
    from discord.ext import commands

    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    os.environ["DISCORD_COMMAND_SYNC_POLICY"] = "bulk"
    USER_ONLY = SimpleNamespace(guild_integration_config=None, user_integration_config=object())

    def settings(guild, user):
        return SimpleNamespace(
            guild_integration_config=object() if guild else None,
            user_integration_config=object() if user else None,
        )

    def scopes(bot):
        return sorted({
            json.dumps([p["integration_types"], p["contexts"]])
            for p in (c.to_dict(bot.tree) for c in bot.tree.get_commands())
        })

    async def run(refreshes):
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
        bot._application = USER_ONLY  # login-time snapshot
        bot.tree.sync = AsyncMock(return_value=[])
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
        adapter._client = bot
        adapter._register_slash_commands()
        seen = []
        for refresh in refreshes:
            bot.application_info = AsyncMock(
                side_effect=None if refresh else RuntimeError("503"),
                return_value=settings(*refresh) if refresh else None,
            )
            await adapter._run_post_connect_initialization()
            seen.append(scopes(bot))
        assert bot.tree.sync.await_count == len(refreshes)
        return seen

    print(json.dumps([asyncio.run(run(case)) for case in json.loads(sys.argv[1])]))
    """
)

GUILD = [[0], [0, 1]]
BOTH = [[0, 1], [0, 1, 2]]
USER = [[1], [0, 1, 2]]
UNSET = [None, None]

# Each case is a run of post-connect calls on one client; None means application_info() fails.
_CASES = [
    ([[True, False]], [GUILD]),
    ([[True, True]], [BOTH]),
    ([[False, True]], [USER]),
    ([[False, False]], [UNSET]),
    # A failed first refresh uses the login-time snapshot (user install only).
    ([None], [USER]),
    # A failed later refresh keeps the last refreshed settings, not the older snapshot.
    ([[True, False], None], [GUILD, GUILD]),
    # Settings changed between READY events on the same client.
    ([[True, True], [True, False]], [BOTH, GUILD]),
]


def test_registered_commands_follow_app_installation_settings():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps([refreshes for refreshes, _ in _CASES])],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named 'discord'" in result.stderr:
        pytest.skip("discord.py optional dependency is not installed")
    assert result.returncode == 0, result.stdout + result.stderr

    observed = json.loads(result.stdout.strip().splitlines()[-1])
    expected = [[[json.dumps(scope)] for scope in scopes] for _, scopes in _CASES]
    assert observed == expected
