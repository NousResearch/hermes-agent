"""A plugin adapter that subclasses ``TelegramAdapter`` under its own platform id must get
Telegram's connect budget.

``_connect_adapter_bounded`` picked the outer deadline by ``platform == Platform.TELEGRAM``, so a
subclass registered under a custom id (e.g. a second bot token run as its own platform) got the
generic 30s. Telegram's connect walks a multi-attempt retry ladder that needs far longer than that,
so every reconnect of such an adapter was cancelled part-way through the ladder.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    GatewayRunner,
    _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT,
    _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT,
    _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT,
)
from plugins.platforms.telegram.adapter import TelegramAdapter

# A custom id only resolves once a plugin registers it; mint test-only members directly.
_LANE = Platform._add_pseudo_member("telegram_subclass_budget_test")
_OTHER = Platform._add_pseudo_member("generic_subclass_budget_test")


class _Hang:
    """Records the budget the gateway waits with and never finishes connecting."""

    def __init__(self, platform):
        self.platform = platform

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        await asyncio.Event().wait()  # never set: the bounded wait owns the outcome
        return True


class _TelegramLane(TelegramAdapter):
    """The plugin shape: TelegramAdapter under a custom platform id."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="123:abc"))
        self.platform = _LANE

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        await asyncio.Event().wait()  # never set: the bounded wait owns the outcome
        return True


async def _waited_budget(tmp_path, monkeypatch, adapter, *, initial):
    monkeypatch.delenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", raising=False)
    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=tmp_path / "sessions"))
    seen = []

    async def _wait(task, timeout):
        seen.append(timeout)
        task.cancel()
        return False

    monkeypatch.setattr(runner, "_wait_or_detach", _wait)
    with pytest.raises(TimeoutError):
        await runner._connect_adapter_bounded(adapter, adapter.platform, is_reconnect=not initial, initial=initial)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize(("initial", "expected"), [
    (False, _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT),
    (True, _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT),
], ids=["reconnect", "cold-start"])
async def test_telegram_subclass_under_custom_id_gets_telegram_budget(tmp_path, monkeypatch, initial, expected):
    assert await _waited_budget(tmp_path, monkeypatch, _TelegramLane(), initial=initial) == [expected]


@pytest.mark.asyncio
async def test_unrelated_custom_platform_keeps_generic_budget(tmp_path, monkeypatch):
    assert await _waited_budget(tmp_path, monkeypatch, _Hang(_OTHER), initial=False) == [
        _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT]
