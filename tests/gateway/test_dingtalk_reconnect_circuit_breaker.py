"""#24851 — DingTalk stream reconnect loop must not storm on a persistent error."""

import asyncio
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
    adapter._fatal_error_handler = None
    return adapter


@pytest.mark.asyncio
async def test_repeated_identical_error_trips_breaker_and_stays_silent(caplog):
    adapter = _adapter()
    count = 0
    sleeps = []

    async def fake_start():
        nonlocal count
        count += 1
        if count > 20:
            adapter._running = False
            return
        raise ConnectionError("boom")

    async def fake_sleep(secs, *a, **k):
        sleeps.append(secs)

    adapter._stream_client.start = fake_start
    with caplog.at_level(logging.WARNING), patch("asyncio.sleep", new=fake_sleep):
        await adapter._run_stream()
    trips = 5  # RECONNECT_CIRCUIT_BREAKER_TRIPS
    records = [r for r in caplog.records if "Stream client error" in r.getMessage()]
    assert len(records) <= trips + 1, [r.getMessage() for r in records]
    assert sleeps[trips:] == [300] * (len(sleeps) - trips), sleeps


@pytest.mark.asyncio
async def test_real_sdk_start_with_issue_type_error_is_bounded_and_fatal(caplog):
    dingtalk_stream = pytest.importorskip("dingtalk_stream")
    import websockets.exceptions  # noqa: F401 — in-gateway state: the SDK's own loop swallows the error

    adapter = _adapter()
    client = dingtalk_stream.DingTalkStreamClient(dingtalk_stream.Credential("id", "secret"))
    client.open_connection = lambda: {"endpoint": "wss://example.invalid", "ticket": "t"}
    adapter._stream_client = client

    def bad_connect(uri):
        raise TypeError("'coroutine' object does not support the asynchronous context manager protocol")

    real_sleep = asyncio.sleep

    async def fast_sleep(secs, *a, **k):
        await real_sleep(0)

    with caplog.at_level(logging.INFO), \
            patch.object(dingtalk_stream.stream.websockets, "connect", bad_connect), \
            patch("asyncio.sleep", new=fast_sleep):
        try:
            await asyncio.wait_for(adapter._run_stream(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    sdk = [r for r in caplog.records if r.name == "dingtalk_stream.client"]
    assert len(sdk) <= 5, len(sdk)
    assert getattr(adapter, "_fatal_error_code", None) == "dingtalk_stream_error"
    assert adapter._fatal_error_retryable is False  # reinstall + restart only; no gateway reconnect churn
