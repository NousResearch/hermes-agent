"""Real Discord READY teardown (outside gateway's automatic SDK mocks)."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

discord = pytest.importorskip("discord")
from discord.ext import commands

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_platform


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["disconnect", "liveness", "replacement"])
async def test_adapter_shutdown_drains_real_delayed_ready(monkeypatch, shutdown):
    bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
    await bot._async_setup_hook()
    ready_events = []

    @bot.event
    async def on_ready():
        ready_events.append("ready")

    state = bot._connection
    state.guild_ready_timeout = 60
    state._ready_state = asyncio.Queue()
    ready_task = state._ready_task = asyncio.create_task(state._delay_ready())
    await asyncio.sleep(0)

    adapter = discord_platform.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = bot
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: None)
    try:
        if shutdown == "disconnect":
            await adapter.disconnect()
        elif shutdown == "liveness":
            monkeypatch.setattr(adapter, "_notify_fatal_error", AsyncMock())
            await adapter._notify_liveness_fatal_error(bot)
        else:
            monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
            monkeypatch.setattr(discord.opus, "is_loaded", lambda: True)
            monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda **kwargs: None)

            def stop_before_network(**kwargs):
                raise RuntimeError("Replacement construction stopped before network access")

            monkeypatch.setattr(discord_platform.commands, "Bot", stop_before_network)
            assert await adapter.connect() is False

        assert bot.is_closed()
        assert ready_task.done(), "Discord READY task survived client shutdown"
        assert ready_task.exception() is None
        assert ready_events == []
    finally:
        if not ready_task.done():
            ready_task.cancel()
        await asyncio.gather(ready_task, return_exceptions=True)
        await bot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ready_state", ["absent", "mock", "current", "failed"])
async def test_ready_cleanup_preserves_close_and_unrelated_errors(ready_state):
    from plugins.platforms.discord.client_lifecycle import close_discord_client

    bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
    await bot._async_setup_hook()
    if ready_state == "mock":
        bot._connection._ready_task = Mock()
    elif ready_state == "current":
        bot._connection._ready_task = asyncio.current_task()
    elif ready_state == "failed":
        async def fail_ready():
            raise RuntimeError("unrelated READY failure")

        task = bot._connection._ready_task = asyncio.create_task(fail_ready())
        await asyncio.wait({task})

    if ready_state == "failed":
        with pytest.raises(RuntimeError, match="unrelated READY failure"):
            await close_discord_client(bot)
    else:
        await close_discord_client(bot)
    assert bot.is_closed()
