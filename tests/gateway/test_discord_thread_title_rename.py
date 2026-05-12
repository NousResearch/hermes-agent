"""Tests for Discord thread renaming after auto-generated session titles."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _ensure_discord_mock() -> None:
    """Install a mock discord module when discord.py isn't available."""
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
from gateway.run import GatewayRunner  # noqa: E402


def _make_source(*, thread_id: str | None = "456") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="user-1",
        chat_id="channel-1",
        user_name="tester",
        chat_type="thread",
        thread_id=thread_id,
    )


def _make_runner(adapter: object) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._gateway_loop = MagicMock()
    runner._gateway_loop.is_closed.return_value = False
    return runner


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_edits_name_and_truncates_to_100_chars():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    thread = MagicMock()
    thread.edit = AsyncMock()

    adapter._client = MagicMock()
    adapter._client.get_channel.return_value = thread

    long_name = "x" * 140
    ok = await adapter.rename_thread(123456789, long_name)

    assert ok is True
    adapter._client.fetch_channel.assert_not_called()
    thread.edit.assert_awaited_once_with(name="x" * 100)


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_fetches_when_cache_misses():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    thread = MagicMock()
    thread.edit = AsyncMock()

    adapter._client = MagicMock()
    adapter._client.get_channel.return_value = None
    adapter._client.fetch_channel = AsyncMock(return_value=thread)

    ok = await adapter.rename_thread(777, "Readable Session")

    assert ok is True
    adapter._client.fetch_channel.assert_awaited_once_with(777)
    thread.edit.assert_awaited_once_with(name="Readable Session")


@pytest.mark.asyncio
async def test_schedule_discord_thread_title_rename_submits_coroutine(monkeypatch):
    adapter = MagicMock()
    adapter.rename_thread = AsyncMock(return_value=True)
    runner = _make_runner(adapter)

    captured: dict[str, object] = {}

    class _FakeFuture:
        def __init__(self, task: asyncio.Task):
            self._task = task

        def add_done_callback(self, cb):
            self._task.add_done_callback(cb)

        def result(self):
            return self._task.result()

    def _fake_run_coroutine_threadsafe(coro, loop):
        captured["loop"] = loop
        task = loop.create_task(coro)
        captured["task"] = task
        return _FakeFuture(task)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run_coroutine_threadsafe)

    runner._schedule_discord_thread_title_rename(_make_source(thread_id="456"), "Readable Session")
    await captured["task"]

    assert captured["loop"] is asyncio.get_running_loop()
    adapter.rename_thread.assert_awaited_once_with(456, "Readable Session")


def test_schedule_discord_thread_title_rename_ignores_invalid_thread_id(monkeypatch):
    adapter = MagicMock()
    adapter.rename_thread = AsyncMock(return_value=True)
    runner = _make_runner(adapter)

    called = False

    def _fake_run_coroutine_threadsafe(coro, loop):
        nonlocal called
        called = True
        raise AssertionError("should not schedule for invalid thread ids")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run_coroutine_threadsafe)

    runner._schedule_discord_thread_title_rename(_make_source(thread_id="not-a-number"), "Readable Session")

    assert called is False
    adapter.rename_thread.assert_not_called()
