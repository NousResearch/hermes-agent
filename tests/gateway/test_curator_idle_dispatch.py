"""The existing curator command remains reachable on the idle gateway path."""
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent

pytest_plugins = ("tests.gateway.test_compression_exhaustion_reset_policy",)


@pytest.mark.asyncio
async def test_curator_and_fallback_dispatch_on_idle_native_message(native_env, monkeypatch):
    runner = native_env.runner
    curator = AsyncMock(return_value="curator handled")
    fallback = AsyncMock(return_value="fallback handled")
    monkeypatch.setattr(runner, "_handle_curator_command", curator)
    monkeypatch.setattr(runner, "_handle_fallback_command", fallback)

    assert await runner._handle_message(MessageEvent(text="/curator", source=native_env.source)) == "curator handled"
    assert await runner._handle_message(MessageEvent(text="/fallback status", source=native_env.source)) == "fallback handled"
    curator.assert_awaited_once()
    fallback.assert_awaited_once()
