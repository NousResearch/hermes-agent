import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_discord_bot_task_runtime_exit_notifies_gateway_for_reconnect(monkeypatch):
    """A post-ready discord.py websocket task crash must not leave the gateway split-brained.

    Regression: producers stayed systemd-active while Discord stopped responding after
    a runtime ClientOSError/ConnectionResetError. The adapter must mark Discord as a
    retryable fatal platform error and notify the gateway supervisor so the existing
    reconnect watcher can replace the dead adapter.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
    adapter._running = True
    adapter._ready_event.set()
    adapter._notify_fatal_error = AsyncMock()

    async def crash():
        raise ConnectionResetError("Cannot write to closing transport")

    task = asyncio.create_task(crash())
    await asyncio.sleep(0)

    adapter._handle_bot_task_done(task)
    await asyncio.sleep(0)

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_retryable is True
    assert adapter.fatal_error_code == "discord_gateway_task_exited"
    assert adapter.fatal_error_message is not None
    assert "Cannot write to closing transport" in adapter.fatal_error_message
    adapter._notify_fatal_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_bot_task_done_ignored_during_intentional_disconnect():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="token"))
    adapter._running = True
    adapter._ready_event.set()
    adapter._disconnecting = True
    adapter._notify_fatal_error = AsyncMock()

    async def stop_cleanly():
        return None

    task = asyncio.create_task(stop_cleanly())
    await asyncio.sleep(0)

    adapter._handle_bot_task_done(task)
    await asyncio.sleep(0)

    assert adapter.has_fatal_error is False
    adapter._notify_fatal_error.assert_not_awaited()
