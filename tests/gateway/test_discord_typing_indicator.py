"""Regression tests for Discord typing indicator cleanup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import asyncio
import sys

import pytest

from gateway.config import Platform, PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.TextChannel = type("TextChannel", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.http = SimpleNamespace(Route=lambda method, path, **kwargs: (method, path, kwargs))
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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.fixture
def adapter():
    inst = object.__new__(DiscordAdapter)
    inst.platform = Platform.DISCORD
    inst._typing_tasks = {}
    inst._background_tasks = set()
    inst._expected_cancelled_tasks = set()
    inst._session_tasks = {}
    inst._pending_messages = {}
    inst._active_sessions = {}
    inst._typing_paused = set()
    inst._voice_clients = {}
    inst._bot_task = None
    inst._post_connect_task = None
    inst._running = True
    inst._disconnecting = False
    inst._release_platform_lock = MagicMock()
    inst._ready_event = asyncio.Event()
    return inst


@pytest.mark.asyncio
async def test_discord_typing_uses_thread_metadata(adapter, monkeypatch):
    """Discord typing must target the thread when thread metadata is present."""
    monkeypatch.setattr(
        discord_platform.discord,
        "http",
        SimpleNamespace(Route=lambda method, path, **kwargs: (method, path, kwargs)),
        raising=False,
    )

    request = AsyncMock(return_value=None)
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))

    await adapter.send_typing("parent-channel", metadata={"thread_id": "456"})
    await asyncio.sleep(0)

    assert "456" in adapter._typing_tasks
    assert "parent-channel" not in adapter._typing_tasks
    route = request.await_args.args[0]
    assert route[2]["channel_id"] == "456"

    await adapter.stop_typing("parent-channel", metadata={"thread_id": "456"})
    assert adapter._typing_tasks == {}


@pytest.mark.asyncio
async def test_discord_typing_failed_loop_does_not_block_retry(adapter, monkeypatch):
    """A failed typing loop must clear its task so the next tick can retry."""
    monkeypatch.setattr(
        discord_platform.discord,
        "http",
        SimpleNamespace(Route=lambda method, path, **kwargs: (method, path, kwargs)),
        raising=False,
    )

    request = AsyncMock(side_effect=[RuntimeError("discord 429"), None])
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))

    await adapter.send_typing("123")
    await asyncio.sleep(0)
    assert "123" not in adapter._typing_tasks

    await adapter.send_typing("123")
    await asyncio.sleep(0)
    assert request.await_count == 2

    await adapter.stop_typing("123")


@pytest.mark.asyncio
async def test_discord_disconnect_clears_typing_tasks(adapter, monkeypatch):
    """Discord disconnect must not leave persistent typing loops behind."""
    monkeypatch.setattr(
        discord_platform.discord,
        "http",
        SimpleNamespace(Route=lambda method, path, **kwargs: (method, path, kwargs)),
        raising=False,
    )

    adapter._client = SimpleNamespace(
        http=SimpleNamespace(request=AsyncMock(return_value=None)),
        close=AsyncMock(return_value=None),
    )

    await adapter.send_typing("123")
    task = adapter._typing_tasks["123"]

    client = adapter._client
    await adapter.disconnect()

    assert adapter._typing_tasks == {}
    assert task.cancelled() or task.done()
    client.close.assert_awaited_once()
    adapter._release_platform_lock.assert_called_once()


@pytest.mark.asyncio
async def test_discord_cancel_background_tasks_clears_typing_tasks(adapter, monkeypatch):
    """Gateway shutdown cleanup must drain Discord-specific typing loops."""
    monkeypatch.setattr(
        discord_platform.discord,
        "http",
        SimpleNamespace(Route=lambda method, path, **kwargs: (method, path, kwargs)),
        raising=False,
    )

    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock(return_value=None)))

    await adapter.send_typing("123")
    task = adapter._typing_tasks["123"]

    await adapter.cancel_background_tasks()

    assert adapter._typing_tasks == {}
    assert task.cancelled() or task.done()
