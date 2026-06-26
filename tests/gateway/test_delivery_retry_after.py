"""Regression tests for flood-control-aware gateway delivery retries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform
from gateway.platforms import base as base_platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer


class _RetryAdapter(BasePlatformAdapter):
    def __init__(self, results):
        super().__init__(SimpleNamespace(extra={}), Platform.TELEGRAM)
        self._results = list(results)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((chat_id, content, reply_to, metadata))
        return self._results.pop(0)

    async def get_chat_info(self, chat_id: str):
        return {}


def test_retry_after_parser_handles_telegram_text_and_attribute():
    assert base_platform.retry_after_seconds_from_error(
        "Flood control exceeded. Retry in 24 seconds"
    ) == 24.0
    assert base_platform.retry_after_seconds_from_error(
        "Too Many Requests: retry after 3"
    ) == 3.0
    assert base_platform.retry_after_seconds_from_error(
        SimpleNamespace(retry_after="7")
    ) == 7.0


def test_send_with_retry_waits_for_provider_retry_after(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(base_platform.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(base_platform.random, "uniform", lambda _a, _b: 0.5)

    adapter = _RetryAdapter([
        SendResult(
            success=False,
            error="Flood control exceeded. Retry in 24 seconds",
            retryable=True,
        ),
        SendResult(success=True, message_id="ok"),
    ])

    import asyncio

    result = asyncio.run(adapter._send_with_retry("chat", "payload", max_retries=2))

    assert result.success is True
    assert result.message_id == "ok"
    assert len(adapter.sent) == 2
    assert sleeps == [24.5]


def test_stream_fallback_final_waits_for_retry_after(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", fake_sleep)

    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.message_len_fn = len
    adapter.send = AsyncMock(side_effect=[
        SimpleNamespace(
            success=False,
            error="Flood control exceeded. Retry in 7 seconds",
            retry_after=None,
        ),
        SimpleNamespace(success=True, message_id="m1"),
    ])

    consumer = GatewayStreamConsumer(adapter, "chat")
    import asyncio

    asyncio.run(consumer._send_fallback_final("final answer"))

    assert consumer._final_content_delivered is True
    assert adapter.send.await_count == 2
    assert sleeps == [7.5]
