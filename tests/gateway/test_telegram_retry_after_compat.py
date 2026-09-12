from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_send_preserves_timedelta_retry_after_after_inline_retries(monkeypatch):
    class TimedeltaFlood(Exception):
        retry_after = timedelta(seconds=3)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(side_effect=TimedeltaFlood("Retry after 3"))

    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.retry_after == 3.0
    assert sleep.await_count == 2
