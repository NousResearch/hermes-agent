"""#24851 — DingTalk stream reconnect loop must not storm on a persistent error."""

import logging
from unittest.mock import MagicMock, patch

import pytest


def _adapter():
    from gateway.config import Platform
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter.__new__(DingTalkAdapter)
    adapter.platform = Platform.DINGTALK
    adapter._running = True
    adapter._stream_client = MagicMock()
    adapter._stream_task = None
    return adapter


async def _run_failing(adapter, exc, calls):
    count = 0
    sleeps = []

    async def fake_start():
        nonlocal count
        count += 1
        if count > calls:
            adapter._running = False
            return
        raise exc

    async def fake_sleep(secs):
        sleeps.append(secs)

    adapter._stream_client.start = fake_start
    with patch("asyncio.sleep", side_effect=fake_sleep):
        await adapter._run_stream()
    return sleeps


@pytest.mark.asyncio
async def test_repeated_identical_error_trips_breaker_and_suppresses_log(caplog):
    adapter = _adapter()
    with caplog.at_level(logging.WARNING):
        sleeps = await _run_failing(adapter, ConnectionError("boom"), calls=20)
    assert any(s >= 300 for s in sleeps), sleeps
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "Stream client error" in r.message]
    assert len(warnings) < 20


@pytest.mark.asyncio
async def test_sdk_type_error_is_logged_as_error_with_upgrade_hint(caplog):
    adapter = _adapter()
    exc = TypeError("'coroutine' object does not support the asynchronous context manager protocol")
    with caplog.at_level(logging.WARNING):
        await _run_failing(adapter, exc, calls=3)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "pip install -U dingtalk-stream" in errors[0].getMessage()
