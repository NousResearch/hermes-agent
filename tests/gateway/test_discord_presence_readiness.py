"""Multiplex callback ownership and readiness across the presence carry."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.discord import adapter as platform
from tests.gateway.test_discord_presence import (
    Activity, ActivityType, CustomActivity, FakeBot, FakeDiscordException, Game,
)


class CallbackBot(FakeBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ready_requests = asyncio.Queue()

    async def start(self, token):
        await self.events["on_ready"]()
        while True:
            done = await self.ready_requests.get()
            if done is None:
                return
            await self.events["on_ready"]()
            done.set_result(None)

    async def close(self):
        await super().close()
        self.ready_requests.put_nowait(None)


@pytest.fixture
def make_adapter(monkeypatch):
    for error_name in ("HTTPException", "ClientException"):
        monkeypatch.setattr(platform.discord, error_name, FakeDiscordException)
    for name, value in {"ActivityType": ActivityType, "Activity": Activity,
                        "Game": Game, "CustomActivity": CustomActivity}.items():
        monkeypatch.setattr(platform.discord, name, value)
    monkeypatch.setattr(platform.commands, "Bot", CallbackBot)
    monkeypatch.setattr(platform.discord.opus, "is_loaded", lambda: True)
    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda **kw: None)

    def make():
        adapter = platform.DiscordAdapter(PlatformConfig(enabled=True, token="synthetic-token"))
        monkeypatch.setattr(adapter, "_slash_commands", False)
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", lambda *a: None)
        monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
        monkeypatch.setattr(adapter, "_run_post_connect_initialization", AsyncMock())
        monkeypatch.setattr(adapter, "_missed_message_backfill_enabled", lambda: False)
        return adapter
    return make


@contextmanager
def profile(home):
    home_token = set_hermes_home_override(home)
    scope_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_scoped_presence_settings_do_not_cross_profiles(tmp_path, monkeypatch):
    import os
    from tests.gateway.test_discord_presence import Activity, ActivityType, CustomActivity, Game

    for name, value in {"ActivityType": ActivityType, "Activity": Activity,
                        "Game": Game, "CustomActivity": CustomActivity}.items():
        monkeypatch.setattr(platform.discord, name, value)
    monkeypatch.setenv("DISCORD_ACTIVITY", "process-profile-activity")
    monkeypatch.setenv("DISCORD_ACTIVITY_TYPE", "listening")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope_token = secret_scope.set_secret_scope({
        "DISCORD_ACTIVITY": "own-profile-activity",
        "DISCORD_ACTIVITY_TYPE": "watching",
    })
    env_before = {name: os.environ[name] for name in ("DISCORD_ACTIVITY", "DISCORD_ACTIVITY_TYPE")}
    try:
        seeded = platform._apply_yaml_config({}, {"activity": "yaml-activity", "activity_type": "playing"})
        assert seeded == {"activity": "yaml-activity", "activity_type": "playing"}
        assert {name: os.environ[name] for name in env_before} == env_before
        from plugins.platforms.discord.config import build_presence

        activity = build_presence(SimpleNamespace(extra=seeded), platform.discord, platform._extra_or_secret)
        assert activity.name == "own-profile-activity"
        assert activity.type == ActivityType.watching
    finally:
        secret_scope.reset_secret_scope(scope_token)

@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, ValueError, TypeError])
async def test_failed_presence_does_not_bypass_initial_username_readiness(monkeypatch, make_adapter, failure):
    monkeypatch.setenv("DISCORD_ACTIVITY", "synthetic-activity")
    original_init = CallbackBot.__init__
    def failing_presence(self, **kwargs):
        original_init(self, **kwargs)
        self.change_presence.side_effect = failure("synthetic presence failure")
    monkeypatch.setattr(CallbackBot, "__init__", failing_presence)
    adapter = make_adapter()
    entered, release = asyncio.Event(), asyncio.Event()
    async def resolve():
        entered.set()
        await release.wait()
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", resolve)
    connecting = asyncio.create_task(adapter.connect())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        adapter._client.change_presence.assert_awaited_once()
        assert not adapter._ready_event.is_set()
        assert not connecting.done()
        release.set()
        assert await asyncio.wait_for(connecting, 5)
        assert adapter._ready_event.is_set()
        await adapter._post_connect_task
        adapter._run_post_connect_initialization.assert_awaited_once()
    finally:
        release.set()
        if not connecting.done():
            connecting.cancel()
        await asyncio.gather(connecting, return_exceptions=True)
        await adapter.disconnect()
