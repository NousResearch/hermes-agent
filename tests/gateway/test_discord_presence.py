"""Real adapter and profile scope with concrete activity/transport test doubles.

The gateway suite mocks discord.py at collection; activity classes below model
its constructor contract without relying on MagicMock attribute behavior.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from enum import IntEnum

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as platform


class FakeDiscordException(Exception):
    """Concrete SDK error type; MagicMock attributes cannot be caught."""


class ActivityType(IntEnum):
    playing = 0
    listening = 2
    watching = 3
    custom = 4


class Activity:
    def __init__(self, *, type, name):
        self.type = type
        self.name = name


class Game(Activity):
    def __init__(self, *, name):
        super().__init__(type=ActivityType.playing, name=name)


class CustomActivity(Activity):
    def __init__(self, *, name):
        super().__init__(type=ActivityType.custom, name=name)


class FakeBot:
    def __init__(self, **kwargs):
        self.user = SimpleNamespace(id=999, name="test-bot")
        self.events = {}
        self.change_presence = AsyncMock()
        self.closed = False
        self.stop = asyncio.Event()

    def event(self, fn):
        self.events[fn.__name__] = fn
        return fn

    async def start(self, token):
        await self.events["on_ready"]()
        await self.stop.wait()

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True
        self.stop.set()


@pytest.fixture
def adapter(monkeypatch):
    instance = platform.DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    for error_name in ("HTTPException", "ClientException"):
        monkeypatch.setattr(platform.discord, error_name, FakeDiscordException)
    for name, value in {"ActivityType": ActivityType, "Activity": Activity, "Game": Game, "CustomActivity": CustomActivity}.items():
        monkeypatch.setattr(platform.discord, name, value)
    monkeypatch.setattr(instance, "_slash_commands", False)
    monkeypatch.setattr(instance, "_acquire_platform_lock", lambda *a: True)
    monkeypatch.setattr(instance, "_release_platform_lock", lambda *a: None)
    monkeypatch.setattr(platform.commands, "Bot", FakeBot)
    monkeypatch.setattr(platform.discord.opus, "is_loaded", lambda: True)
    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda **kw: None)
    monkeypatch.setattr(instance, "_resolve_allowed_usernames", AsyncMock())
    monkeypatch.setattr(instance, "_run_post_connect_initialization", AsyncMock())
    monkeypatch.setattr(instance, "_missed_message_backfill_enabled", lambda: True)
    monkeypatch.setattr(instance, "_ensure_missed_message_backfill_task", Mock())
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("text,kind,expected", [
    ("  reading  ", None, ActivityType.playing),
    ("reading", "", ActivityType.playing),
    ("reading", "strange", ActivityType.playing),
    ("reading", " WATCHING ", ActivityType.watching),
    ("reading", "listening", ActivityType.listening),
    ("reading", "custom", ActivityType.custom),
    ("  ", "watching", None),
])
async def test_presence_reapplied_on_ready_without_blocking_backfill(adapter, monkeypatch, text, kind, expected):
    monkeypatch.setenv("DISCORD_ACTIVITY", text)
    monkeypatch.delenv("DISCORD_ACTIVITY_TYPE", raising=False)
    if kind is not None:
        monkeypatch.setenv("DISCORD_ACTIVITY_TYPE", kind)
    try:
        assert await adapter.connect() is True
        bot = adapter._client
        if expected is None:
            bot.change_presence.assert_not_awaited()
        else:
            bot.change_presence.assert_awaited_once()
            activity = bot.change_presence.await_args.kwargs["activity"]
            assert activity.name == text.strip()
            assert activity.type == expected
            assert isinstance(activity, CustomActivity if expected == ActivityType.custom else Game if expected == ActivityType.playing else Activity)
        bot.change_presence.side_effect = RuntimeError("Discord rejected presence")
        await bot.events["on_ready"]()
        assert bot.change_presence.await_count == (0 if expected is None else 2)
        assert adapter._resolve_allowed_usernames.await_count == 2
        assert adapter._ready_event.is_set()
        assert adapter._ensure_missed_message_backfill_task.call_count == 2
        await adapter._post_connect_task
        adapter._run_post_connect_initialization.assert_awaited()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,expected", [
    ({"DISCORD_ACTIVITY": "profile-only", "DISCORD_ACTIVITY_TYPE": "watching"}, ActivityType.watching),
    ({"DISCORD_ACTIVITY": "profile-only"}, ActivityType.playing),
    ({}, None),
])
async def test_multiplex_presence_never_borrows_process_environment(adapter, monkeypatch, scope, expected):
    monkeypatch.setenv("DISCORD_ACTIVITY", "wrong-profile")
    monkeypatch.setenv("DISCORD_ACTIVITY_TYPE", "listening")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope(scope)
    try:
        assert await adapter.connect() is True
    finally:
        secret_scope.reset_secret_scope(token)
    try:
        presence = adapter._client.change_presence
        if expected is None:
            presence.assert_not_awaited()
        else:
            presence.assert_awaited_once()
            activity = presence.await_args.kwargs["activity"]
            assert activity.name == "profile-only"
            assert activity.type == expected
    finally:
        await adapter.disconnect()
