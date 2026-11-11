"""A relay-only gateway (hosted Hermes Cloud) answers its first message as soon as it can after a boot.

Its connector buffers inbound until the gateway dials, and every cold boot or wake waits on this path,
so neither the settle pause other platforms need before lifecycle sends nor the 30s first retry of a
failed connect applies to it.
"""
import asyncio
import sys
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner

SETTLE_S = 1.0


class _Adapter(BasePlatformAdapter):
    def __init__(self, platform):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, platform) -> GatewayRunner:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(enabled=True, token="***", extra={"relay_url": "wss://relay.test"})
    runner = GatewayRunner(GatewayConfig(platforms={platform: config}, sessions_dir=tmp_path / "sessions"))
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _Adapter(platform))

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


class _StartupAsyncio:
    """``asyncio`` as gateway.run_startup sees it: records the pauses the post-connect wiring takes
    before it opens the inbound gate (the watchers it spawns sleep on their own schedules)."""

    def __init__(self):
        self.wiring_pauses: list[float] = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay, result=None):
        if sys._getframe(1).f_code.co_name == "_start_finish_wiring":
            self.wiring_pauses.append(delay)
            delay = 0
        return await asyncio.sleep(delay, result)  # health: allow HX013 -- passes the startup code's own sleeps through unchanged


async def _settle_pauses(monkeypatch, tmp_path, platform) -> int:
    startup_asyncio = _StartupAsyncio()
    runner = _runner(monkeypatch, tmp_path, platform)
    monkeypatch.setattr("gateway.run_startup.asyncio", startup_asyncio)
    try:
        assert await runner.start() is True
    finally:
        await runner.stop()
    return startup_asyncio.wiring_pauses.count(SETTLE_S)


@pytest.mark.asyncio
async def test_a_relay_only_gateway_opens_without_the_adapter_settle_pause(monkeypatch, tmp_path):
    assert await _settle_pauses(monkeypatch, tmp_path, Platform.RELAY) == 0


@pytest.mark.asyncio
async def test_a_direct_platform_still_settles_before_lifecycle_sends(monkeypatch, tmp_path):
    assert await _settle_pauses(monkeypatch, tmp_path, Platform.DISCORD) == 1


def test_a_failed_relay_connect_retries_within_seconds(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, Platform.RELAY)
    now = time.monotonic()
    relay = runner._startup_retry_entry(Platform.RELAY, None, PlatformConfig(enabled=True))
    telegram = runner._startup_retry_entry(Platform.TELEGRAM, None, PlatformConfig(enabled=True, token="***"))
    assert relay["next_retry"] - now == pytest.approx(5, abs=1)
    assert telegram["next_retry"] - now == pytest.approx(30, abs=1)
