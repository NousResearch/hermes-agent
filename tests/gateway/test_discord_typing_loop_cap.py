"""Regression tests for the typing-loop lifetime cap (2026-08-30 np1 incident).

The Discord adapter keeps a per-channel background loop that pings the
/typing endpoint every 12s until stop_typing() cancels it.  When a
stop_typing() is missed (interrupted delivery, exception path), the loop
used to run forever and the channel showed a permanently stuck
"is typing…" bubble.  ``TYPING_LOOP_MAX_SECONDS`` now bounds the loop's
lifetime as a *hard* cap (asyncio.wait_for), so even a loop parked on a
hung HTTP request or a long rate-limit backoff is reaped.  An active
run's _keep_typing refresher recreates the loop within ~2s, so only
orphaned loops die.
"""

from __future__ import annotations

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
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_adapter(request=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter._client = SimpleNamespace(
        http=SimpleNamespace(request=request or AsyncMock())
    )
    return adapter


async def _wait_reaped(adapter, chat_id, timeout=2.0):
    """Wait until the typing task for chat_id finishes and is deregistered."""
    task = adapter._typing_tasks.get(chat_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    # finally-block cleanup runs inside the task; yield once for safety.
    await asyncio.sleep(0)
    assert chat_id not in adapter._typing_tasks


class TestTypingLoopLifetimeCap:
    @pytest.mark.asyncio
    async def test_orphaned_loop_expires_at_cap(self):
        """A loop whose stop_typing() never arrives dies at the cap."""
        adapter = _make_adapter()
        adapter.TYPING_LOOP_MAX_SECONDS = 0.05
        await adapter.send_typing("123")
        await _wait_reaped(adapter, "123")

    @pytest.mark.asyncio
    async def test_hung_request_is_reaped_at_cap(self):
        """The cap is hard: a loop parked on an HTTP request that never
        returns is cancelled at the deadline (the audit-flagged path)."""

        async def _hang(route):
            await asyncio.Event().wait()  # never set — hangs forever

        adapter = _make_adapter(request=_hang)
        adapter.TYPING_LOOP_MAX_SECONDS = 0.1
        await adapter.send_typing("124")
        await _wait_reaped(adapter, "124")

    @pytest.mark.asyncio
    async def test_long_rate_limit_backoff_is_reaped_at_cap(self):
        """A 429 retry_after longer than the remaining lifetime cannot keep
        the loop alive past the cap."""

        async def _rate_limited(route):
            raise RuntimeError("429 Too Many Requests")

        adapter = _make_adapter(request=_rate_limited)
        adapter._extract_discord_retry_after = lambda e: 3600.0
        adapter.TYPING_LOOP_MAX_SECONDS = 0.1
        await adapter.send_typing("125")
        await _wait_reaped(adapter, "125")

    @pytest.mark.asyncio
    async def test_loop_pings_until_stopped_within_cap(self):
        """Inside the cap the loop behaves as before: ping, then cancel on stop."""
        adapter = _make_adapter()
        await adapter.send_typing("456")
        assert "456" in adapter._typing_tasks
        await asyncio.sleep(0.05)
        assert adapter._client.http.request.await_count >= 1
        await adapter.stop_typing("456")
        assert "456" not in adapter._typing_tasks

    @pytest.mark.asyncio
    async def test_expired_loop_can_be_recreated(self):
        """After cap expiry a fresh send_typing() starts a new loop (the
        _keep_typing refresher path during a genuine run)."""
        adapter = _make_adapter()
        adapter.TYPING_LOOP_MAX_SECONDS = 0.05
        await adapter.send_typing("789")
        await _wait_reaped(adapter, "789")
        adapter.TYPING_LOOP_MAX_SECONDS = 300
        await adapter.send_typing("789")
        assert "789" in adapter._typing_tasks
        await adapter.stop_typing("789")
