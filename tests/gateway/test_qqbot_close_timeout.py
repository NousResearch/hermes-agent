import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.qqbot import adapter as qq_adapter


class Closable:
    def __init__(self, *, closed=False, error=None, wait_forever=False):
        self.closed = closed
        self.error = error
        self.wait_forever = wait_forever
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        if self.wait_forever:
            await asyncio.Future()
        if self.error:
            raise self.error


@pytest.mark.asyncio
async def test_close_ws_closes_and_clears_websocket_and_session():
    adapter = SimpleNamespace(
        _ws=Closable(),
        _session=Closable(),
        _log_tag="QQBot:test",
    )

    await qq_adapter.QQAdapter._close_ws(adapter)

    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_close_ws_logs_and_clears_resources_when_close_times_out(caplog, monkeypatch):
    monkeypatch.setattr(qq_adapter, "QQ_WS_CLOSE_TIMEOUT_SECONDS", 0.001)
    adapter = SimpleNamespace(
        _ws=Closable(wait_forever=True),
        _session=Closable(wait_forever=True),
        _log_tag="QQBot:test",
    )

    with caplog.at_level("WARNING", logger=qq_adapter.logger.name):
        await qq_adapter.QQAdapter._close_ws(adapter)

    assert adapter._ws is None
    assert adapter._session is None
    assert "WebSocket close timed out" in caplog.text
    assert "session close timed out" in caplog.text


@pytest.mark.asyncio
async def test_close_ws_logs_and_continues_after_close_failure(caplog):
    adapter = SimpleNamespace(
        _ws=Closable(error=RuntimeError("ws broken")),
        _session=Closable(error=RuntimeError("session broken")),
        _log_tag="QQBot:test",
    )

    with caplog.at_level("WARNING", logger=qq_adapter.logger.name):
        await qq_adapter.QQAdapter._close_ws(adapter)

    assert adapter._ws is None
    assert adapter._session is None
    assert "WebSocket close failed" in caplog.text
    assert "session close failed" in caplog.text
