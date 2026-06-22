"""Feishu WebSocket idle watchdog tests.

The watchdog detects half-open WS connections (TCP ESTABLISHED but no data
flowing, e.g. due to NAT table expiry after hours of inactivity) and forces
a reconnect.  Pattern borrowed from NanoClaw's FeishuChannel.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    """Build a minimal FeishuAdapter with mocked SDK dependencies."""
    from gateway.platforms.feishu import FeishuAdapter

    with patch.object(FeishuAdapter, "__init__", lambda self, cfg: None):
        adapter = FeishuAdapter.__new__(FeishuAdapter)

    adapter._loop = None
    adapter._ws_client = MagicMock()
    adapter._ws_future = None
    adapter._ws_thread_loop = None
    adapter._last_ws_event_time = time.monotonic()
    adapter._ws_watchdog_task = None
    adapter._ws_idle_threshold = 300
    adapter._ws_watchdog_interval = 180
    adapter._connection_mode = "websocket"
    adapter._running = True
    return adapter


class TestWsEventWrapper:
    """_ws_event_wrapper updates _last_ws_event_time on every callback."""

    def test_wrapper_updates_timestamp(self):
        adapter = _make_adapter()
        old_time = adapter._last_ws_event_time

        inner = MagicMock(return_value="result")
        wrapped = adapter._ws_event_wrapper(inner)

        time.sleep(0.01)
        result = wrapped("arg1", key="val")

        assert result == "result"
        inner.assert_called_once_with("arg1", key="val")
        assert adapter._last_ws_event_time > old_time

    def test_wrapper_updates_even_if_callback_raises(self):
        adapter = _make_adapter()
        old_time = adapter._last_ws_event_time

        inner = MagicMock(side_effect=ValueError("boom"))
        wrapped = adapter._ws_event_wrapper(inner)

        time.sleep(0.01)
        with pytest.raises(ValueError, match="boom"):
            wrapped()

        assert adapter._last_ws_event_time > old_time


class TestWsWatchdogLifecycle:
    """Watchdog task starts/stops at correct lifecycle points."""

    def test_start_creates_task(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._start_ws_watchdog()
            assert adapter._ws_watchdog_task is not None
            assert not adapter._ws_watchdog_task.done()
            adapter._stop_ws_watchdog()

        asyncio.run(_run())

    def test_stop_cancels_task(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._start_ws_watchdog()
            task = adapter._ws_watchdog_task
            adapter._stop_ws_watchdog()
            await asyncio.sleep(0)
            assert adapter._ws_watchdog_task is None
            assert task.cancelled() or task.done()

        asyncio.run(_run())

    def test_start_replaces_existing_task(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._start_ws_watchdog()
            first_task = adapter._ws_watchdog_task
            adapter._start_ws_watchdog()
            second_task = adapter._ws_watchdog_task
            await asyncio.sleep(0)
            assert second_task is not first_task
            assert first_task.cancelled() or first_task.done()
            adapter._stop_ws_watchdog()

        asyncio.run(_run())


class TestWsWatchdogTrigger:
    """Watchdog triggers reconnect when idle exceeds threshold."""

    def test_triggers_reconnect_when_idle(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_idle_threshold = 0.05
            adapter._ws_watchdog_interval = 0.02
            adapter._last_ws_event_time = time.monotonic() - 1.0

            reconnect_called = asyncio.Event()
            original_reconnect = AsyncMock()

            async def mock_reconnect():
                await original_reconnect()
                reconnect_called.set()

            adapter._reconnect_websocket = mock_reconnect
            adapter._start_ws_watchdog()

            try:
                await asyncio.wait_for(reconnect_called.wait(), timeout=2.0)
            finally:
                adapter._stop_ws_watchdog()

            original_reconnect.assert_called_once()

        asyncio.run(_run())

    def test_no_reconnect_when_active(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_idle_threshold = 10.0
            adapter._ws_watchdog_interval = 0.02
            adapter._last_ws_event_time = time.monotonic()

            reconnect_mock = AsyncMock()
            adapter._reconnect_websocket = reconnect_mock

            adapter._start_ws_watchdog()
            await asyncio.sleep(0.1)
            adapter._stop_ws_watchdog()

            reconnect_mock.assert_not_called()

        asyncio.run(_run())

    def test_skips_when_not_websocket_mode(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._connection_mode = "webhook"
            adapter._ws_idle_threshold = 0.01
            adapter._ws_watchdog_interval = 0.02
            adapter._last_ws_event_time = time.monotonic() - 1.0

            reconnect_mock = AsyncMock()
            adapter._reconnect_websocket = reconnect_mock

            adapter._start_ws_watchdog()
            await asyncio.sleep(0.1)
            adapter._stop_ws_watchdog()

            reconnect_mock.assert_not_called()

        asyncio.run(_run())

    def test_skips_when_ws_client_none(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_client = None
            adapter._ws_idle_threshold = 0.01
            adapter._ws_watchdog_interval = 0.02
            adapter._last_ws_event_time = time.monotonic() - 1.0

            reconnect_mock = AsyncMock()
            adapter._reconnect_websocket = reconnect_mock

            adapter._start_ws_watchdog()
            await asyncio.sleep(0.1)
            adapter._stop_ws_watchdog()

            reconnect_mock.assert_not_called()

        asyncio.run(_run())


class TestReconnectWebsocket:
    """_reconnect_websocket tears down and rebuilds WS."""

    def test_reconnect_calls_teardown_and_rebuild(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_reconnect_in_progress = False

            adapter._disable_websocket_auto_reconnect = MagicMock()
            adapter._connect_websocket = AsyncMock()

            await adapter._reconnect_websocket()

            adapter._disable_websocket_auto_reconnect.assert_called_once()
            adapter._connect_websocket.assert_called_once()
            assert adapter._last_ws_event_time > 0
            assert not adapter._ws_reconnect_in_progress

        asyncio.run(_run())

    def test_reentrant_call_skipped(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_reconnect_in_progress = True

            adapter._disable_websocket_auto_reconnect = MagicMock()
            adapter._connect_websocket = AsyncMock()

            await adapter._reconnect_websocket()

            adapter._disable_websocket_auto_reconnect.assert_not_called()
            adapter._connect_websocket.assert_not_called()

        asyncio.run(_run())

    def test_flag_cleared_on_error(self):
        async def _run():
            adapter = _make_adapter()
            adapter._loop = asyncio.get_running_loop()
            adapter._ws_reconnect_in_progress = False

            adapter._disable_websocket_auto_reconnect = MagicMock()
            adapter._connect_websocket = AsyncMock(side_effect=RuntimeError("fail"))

            with pytest.raises(RuntimeError, match="fail"):
                await adapter._reconnect_websocket()

            assert not adapter._ws_reconnect_in_progress

        asyncio.run(_run())


class TestWsWatchdogSettings:
    """Settings for idle threshold and watchdog interval."""

    def test_default_settings_values(self):
        from gateway.platforms.feishu import FeishuAdapterSettings
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(FeishuAdapterSettings)}
        assert "ws_idle_threshold" in fields
        assert fields["ws_idle_threshold"].default == 300
        assert "ws_watchdog_interval" in fields
        assert fields["ws_watchdog_interval"].default == 180
