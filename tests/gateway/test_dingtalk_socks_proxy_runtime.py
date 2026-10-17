"""#135646 — a SOCKS proxy without python-socks must fail loud once, not retry every 3 s forever."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

_SOCKS_ERROR = "python-socks is required to use a SOCKS proxy"


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


def test_missing_socks_runtime_is_recognised_directly_and_when_chained():
    from plugins.platforms.dingtalk.adapter import _fatal_sdk_error_message, _is_missing_socks_runtime

    assert _is_missing_socks_runtime(ImportError(_SOCKS_ERROR))
    try:
        try:
            raise ImportError(_SOCKS_ERROR)
        except ImportError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert _is_missing_socks_runtime(outer)
    message = _fatal_sdk_error_message(ImportError(_SOCKS_ERROR))
    assert message is not None and "python-socks" in message and "hermes-agent[dingtalk]" in message


def test_unrelated_import_and_connection_errors_stay_retryable():
    from plugins.platforms.dingtalk.adapter import _fatal_sdk_error_message

    assert _fatal_sdk_error_message(ImportError("No module named 'h2'")) is None
    assert _fatal_sdk_error_message(ConnectionError("reset by peer")) is None


@pytest.mark.asyncio
async def test_socks_import_error_escaping_start_is_fatal_without_backoff():
    adapter = _adapter()
    sleeps = []
    calls = 0

    async def fake_start():
        nonlocal calls
        calls += 1
        if calls > 3:  # bound the loop so a regression fails instead of spinning
            adapter._running = False
        raise ImportError(_SOCKS_ERROR)

    async def fake_sleep(secs, *a, **k):
        sleeps.append(secs)

    adapter._stream_client.start = fake_start
    with patch("asyncio.sleep", new=fake_sleep):
        await adapter._run_stream()
    assert calls == 1 and sleeps == []
    assert adapter._fatal_error_code == "dingtalk_stream_error"
    assert adapter._fatal_error_retryable is False
    assert "python-socks" in adapter._fatal_error_message


@pytest.mark.asyncio
async def test_real_sdk_retry_loop_with_socks_proxy_and_no_runtime_is_bounded_and_fatal(caplog):
    dingtalk_stream = pytest.importorskip("dingtalk_stream")
    import websockets.exceptions  # noqa: F401 — the SDK's except clause needs it loaded

    adapter = _adapter()
    client = dingtalk_stream.DingTalkStreamClient(dingtalk_stream.Credential("id", "secret"))
    client.open_connection = lambda: {"endpoint": "wss://example.invalid", "ticket": "t"}
    adapter._stream_client = client

    def socks_without_runtime(uri):
        raise ImportError(_SOCKS_ERROR)

    real_sleep = asyncio.sleep

    async def fast_sleep(secs, *a, **k):
        await real_sleep(0)

    with caplog.at_level(logging.INFO), \
            patch.object(dingtalk_stream.stream.websockets, "connect", socks_without_runtime), \
            patch("asyncio.sleep", new=fast_sleep):
        try:
            await asyncio.wait_for(adapter._run_stream(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    sdk = [r for r in caplog.records if r.name == "dingtalk_stream.client"]
    assert len(sdk) <= 5, len(sdk)
    assert adapter._fatal_error_code == "dingtalk_stream_error"
    assert adapter._fatal_error_retryable is False
    assert "python-socks" in adapter._fatal_error_message
