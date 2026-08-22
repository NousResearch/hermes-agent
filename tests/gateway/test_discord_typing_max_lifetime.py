"""Tests for DiscordAdapter typing-loop max-lifetime deadline guard.

Issue #90151: the persistent typing loop in ``DiscordAdapter.send_typing``
has no natural exit condition other than ``stop_typing()`` or a non-429
error. If ``stop_typing`` never reaches the adapter (e.g. a crashed run,
or a thread-vs-parent-channel key mismatch), the loop runs forever and
Discord keeps showing the "…is typing" badge until the gateway restarts.

The fix adds a configurable max-lifetime deadline
(``discord.typing_loop_max_seconds``, default 600s, 0 disables it).
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
    discord_mod.ui = SimpleNamespace(
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5
    )
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

from types import SimpleNamespace  # noqa: E402

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_adapter(max_seconds: int = 600):
    """Build a DiscordAdapter with a mocked client ready to start typing."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._typing_tasks = {}
    adapter._typing_loop_max_seconds = max_seconds
    return adapter


class TestTypingLoopMaxLifetime:
    @pytest.mark.asyncio
    async def test_loop_expires_after_max_lifetime(self):
        """The typing loop must stop once the max-lifetime deadline elapses.

        Uses a very short deadline (1s) so the test finishes quickly, and a
        mocked client whose request resolves instantly so the loop spins at
        full speed — each iteration hits the deadline check first.
        """
        adapter = _make_adapter(max_seconds=1)
        adapter._client.http.request = AsyncMock()

        await adapter.send_typing("channel-1")
        assert "channel-1" in adapter._typing_tasks

        # Wait long enough for the deadline to elapse and the loop to exit.
        await asyncio.sleep(1.5)

        # The loop should have removed itself from the registry.
        assert "channel-1" not in adapter._typing_tasks

    @pytest.mark.asyncio
    async def test_stop_typing_still_works_with_deadline(self):
        """stop_typing must still cancel the loop cleanly before deadline."""
        adapter = _make_adapter(max_seconds=600)
        adapter._client.http.request = AsyncMock()

        await adapter.send_typing("channel-2")
        assert "channel-2" in adapter._typing_tasks

        await adapter.stop_typing("channel-2")
        assert "channel-2" not in adapter._typing_tasks

    @pytest.mark.asyncio
    async def test_zero_max_seconds_disables_deadline(self):
        """Setting max_seconds to 0 must disable the deadline guard.

        The loop runs indefinitely (until stop_typing) — verified by letting
        it spin well past where a 1s deadline would have fired and
        asserting it's still alive.
        """
        adapter = _make_adapter(max_seconds=0)
        adapter._client.http.request = AsyncMock()

        await adapter.send_typing("channel-3")
        # Give it enough time that a 1s deadline would have expired.
        await asyncio.sleep(1.5)
        assert "channel-3" in adapter._typing_tasks

        # Clean up.
        await adapter.stop_typing("channel-3")
        assert "channel-3" not in adapter._typing_tasks