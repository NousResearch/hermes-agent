import json
import subprocess
import sys
import textwrap

import pytest

# Runs in a subprocess so it gets the real discord.py, not another test file's fake module.
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

    def new_bot(adapter, login_settings):
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
        bot._application = settings(*login_settings)
        bot.tree.sync = AsyncMock(return_value=[])
        adapter._app_info = None  # as connect() does when it builds a client
        adapter._client = bot
        adapter._register_slash_commands()
        return bot

    async def run(steps):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
        seen = []
        for step in steps:
            kind = step[0]
            if kind == "connect":
                bot = new_bot(adapter, step[1])
                continue
            if kind == "refresh_then_replaced":
                stale = bot
                async def refresh():
                    new_bot(adapter, step[2])
                    return settings(*step[1])
                stale.application_info = refresh
                await adapter._run_post_connect_initialization()
                seen.append([stale.tree.sync.await_count, scopes(adapter._client)])
                bot = adapter._client
                continue
            bot.application_info = AsyncMock(
                side_effect=RuntimeError("503") if kind == "refresh_fails" else None,
                return_value=settings(*step[1]) if kind == "refresh" else None,
            )
            await adapter._run_post_connect_initialization()
            seen.append(scopes(bot))
        return seen

    print(json.dumps({name: asyncio.run(run(steps)) for name, steps in json.loads(sys.argv[1]).items()}))
    """
)

GUILD = json.dumps([[0], [0, 1]])
BOTH = json.dumps([[0, 1], [0, 1, 2]])
USER = json.dumps([[1], [0, 1, 2]])
UNSET = json.dumps([None, None])
USER_LOGIN = ["connect", [False, True]]

_SCENARIOS = {
    "guild_only": ([USER_LOGIN, ["refresh", [True, False]]], [[GUILD]]),
    "guild_and_user": ([USER_LOGIN, ["refresh", [True, True]]], [[BOTH]]),
    "user_only": ([USER_LOGIN, ["refresh", [False, True]]], [[USER]]),
    "settings_unavailable": ([USER_LOGIN, ["refresh", [False, False]]], [[UNSET]]),
    "first_refresh_fails_uses_login_snapshot": ([USER_LOGIN, ["refresh_fails"]], [[USER]]),
    "later_refresh_fails_keeps_last_refresh": (
        [USER_LOGIN, ["refresh", [True, False]], ["refresh_fails"]],
        [[GUILD], [GUILD]],
    ),
    "settings_change_between_ready_events": (
        [USER_LOGIN, ["refresh", [True, True]], ["refresh", [True, False]]],
        [[BOTH], [GUILD]],
    ),
    "new_client_ignores_old_client_refresh": (
        [["connect", [True, True]], ["refresh", [True, False]], USER_LOGIN, ["refresh_fails"]],
        [[GUILD], [USER]],
    ),
    "client_replaced_mid_refresh_skips_sync": (
        [["connect", [True, True]], ["refresh_then_replaced", [True, False], [False, True]], ["refresh_fails"]],
        [[0, [UNSET]], [USER]],
    ),
}


def test_registered_commands_follow_app_installation_settings():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps({name: steps for name, (steps, _) in _SCENARIOS.items()})],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named 'discord'" in result.stderr:
        pytest.skip("discord.py optional dependency is not installed")
    assert result.returncode == 0, result.stdout + result.stderr

    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {name: expected for name, (_, expected) in _SCENARIOS.items()}
