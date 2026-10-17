"""Regression for #26859: transport badges must not mutate adapter lifecycle."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_connect import FakeBot, _ensure_discord_mock

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


class _LiveBot(FakeBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stopped = asyncio.Event()

    async def start(self, token):
        await self._events["on_ready"]()
        await self.stopped.wait()

    async def close(self):
        self.stopped.set()

    def is_closed(self):
        return self.stopped.is_set()

    def is_ready(self):
        return True


@pytest.mark.asyncio
async def test_transport_badges_preserve_live_watchdog_and_fatal_state(monkeypatch):
    adapter = discord_platform.DiscordAdapter(
        PlatformConfig(enabled=True, token="test-token")
    )
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock", lambda *a, **k: (True, None)
    )
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda *a, **k: None)
    monkeypatch.setattr(discord_platform.commands, "Bot", lambda **k: _LiveBot(**k))
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    monkeypatch.setattr(adapter, "_run_post_connect_initialization", AsyncMock())
    monkeypatch.setattr(
        adapter, "_read_websocket_health", lambda client: (True, "healthy")
    )
    writes = []
    monkeypatch.setattr(
        adapter,
        "_write_runtime_status_safe",
        lambda state, **kw: writes.append((state, kw)),
    )
    try:
        assert await adapter.connect()
        bot, watchdog = adapter._client, adapter._liveness_task
        assert bot is not None
        assert watchdog is not None and not watchdog.done()
        for event, expected in (
            ("on_disconnect", "disconnected"),
            ("on_resumed", "connected"),
            ("on_ready", "connected"),
        ):
            writes.clear()
            assert event in bot._events, f"Missing transport handler: {event}"
            await bot._events[event]()
            assert writes[-1][1]["platform_state"] == expected
            assert (
                adapter._running
                and adapter._liveness_task is watchdog
                and not watchdog.done()
            )
        adapter._set_fatal_error("transport_fatal", "test fatal", retryable=True)
        writes.clear()
        for event in ("on_ready", "on_resumed", "on_disconnect"):
            await bot._events[event]()
        assert writes == [] and adapter.has_fatal_error
        assert adapter._fatal_error_code == "transport_fatal"
    finally:
        await adapter.disconnect()
