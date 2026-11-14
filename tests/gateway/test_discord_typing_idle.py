"""The persistent Discord typing loop must not outlive the turn that started it.

``send_typing`` spawns a per-channel loop that re-POSTs every few seconds until
``stop_typing`` cancels it. A ``send_typing`` that lands just after the turn's final
``stop_typing`` (e.g. ``_progress_restore_typing`` waking up in that window) used to leave
that loop unreachable, so the channel showed "typing…" forever. The loop now retires
itself once its last refresh is older than ``_TYPING_LOOP_IDLE_SECONDS``.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
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

from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import adapter as discord_adapter


@pytest.fixture()
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    a._typing_tasks = {}
    a._typing_touch = {}
    return a


@pytest.fixture()
def fast_loop(monkeypatch):
    monkeypatch.setattr(discord_adapter, "_TYPING_POST_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(discord_adapter, "_TYPING_LOOP_IDLE_SECONDS", 0.2)


def test_orphaned_loop_self_terminates(adapter, fast_loop):
    async def scenario():
        await adapter.send_typing("chan-orphan")
        task = adapter._typing_tasks["chan-orphan"]
        # Nobody ever calls stop_typing again — the loop must give up by itself.
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        return adapter._client.http.request.call_count

    posts = asyncio.run(scenario())
    assert posts >= 1
    assert "chan-orphan" not in adapter._typing_tasks
    assert "chan-orphan" not in adapter._typing_touch


class _ClockShim:
    """``time`` for the adapter under test, with a monotonic clock the test drives."""

    def __init__(self, clock: "_FakeClock") -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(discord_adapter.time, name)


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_refreshed_loop_keeps_typing(adapter, fast_loop, monkeypatch):
    # The idle deadline is wall-clock behaviour, so drive the clock instead of sleeping on it.
    clock = _FakeClock()
    monkeypatch.setattr(discord_adapter, "time", _ClockShim(clock))
    monkeypatch.setattr(discord_adapter, "_TYPING_POST_INTERVAL_SECONDS", 0)  # sleep(0): bare yield
    monkeypatch.setattr(discord_adapter, "_TYPING_LOOP_IDLE_SECONDS", 10.0)

    async def scenario():
        await adapter.send_typing("chan-live")
        task = adapter._typing_tasks["chan-live"]
        # 100 simulated seconds — ten times the deadline — but every tick refreshes the touch.
        for _ in range(100):
            clock.advance(1.0)
            adapter._typing_touch["chan-live"] = clock.now
            await asyncio.sleep(0)  # let the loop run its idle check
        assert not task.done()
        await adapter.stop_typing("chan-live")
        return task

    task = asyncio.run(scenario())
    assert task.done()
    assert "chan-live" not in adapter._typing_tasks
    assert "chan-live" not in adapter._typing_touch


def test_stop_typing_cancels_and_clears(adapter, fast_loop):
    async def scenario():
        await adapter.send_typing("chan-stop")
        await adapter.stop_typing("chan-stop")
        return adapter._typing_tasks, adapter._typing_touch

    tasks, touch = asyncio.run(scenario())
    assert tasks == {}
    assert touch == {}
