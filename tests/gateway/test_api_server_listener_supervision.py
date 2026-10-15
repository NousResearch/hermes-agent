"""Listener-loss recovery, independently of the unconfirmed hosted outage cause."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner


async def _connect(adapter):
    # Drive heartbeat passes explicitly, without waiting for the production interval.
    with patch.object(adapter, "_heartbeat_loop", new_callable=AsyncMock):
        assert await adapter.connect()


async def _close(adapter):
    await adapter.disconnect()
    tasks = list(adapter._background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _runner(adapter):
    async def disconnect(a, platform):
        await a.disconnect()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.API_SERVER: adapter.config})
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._failed_platforms = {}
    runner.delivery_router = MagicMock()
    runner._ensure_reconnect_watcher_running = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner._safe_adapter_disconnect = disconnect
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    runner._bind_voice_input_callback = MagicMock()
    runner._schedule_planned_restart_replay = MagicMock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_resume_pending_sessions = MagicMock()
    runner._run_in_executor_with_context = AsyncMock()
    runner._wire_adapter_handlers = lambda a: a.set_fatal_error_handler(runner._handle_adapter_fatal_error)
    adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["site_stop", "runner_cleanup", "server_close"])
async def test_listener_loss_queues_reconnect_while_dashboard_remains_live(loss):
    import hermes_cli.web_server as dashboard

    adapter = APIServerAdapter(PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": "test-listener-supervision-key"}))
    replacement = None
    await _connect(adapter)
    port = adapter._runner.addresses[0][1]
    runner = _runner(adapter)
    try:
        if loss == "site_stop":
            await adapter._site.stop()
        elif loss == "runner_cleanup":
            await adapter._runner.cleanup()
        else:
            # Inject closure below aiohttp's site bookkeeping.
            adapter._site._server.close()
        assert not adapter._runner.addresses
        async with aiohttp.ClientSession() as client:
            with pytest.raises(aiohttp.ClientConnectorError):
                await client.get(f"http://127.0.0.1:{port}/health")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dashboard.app), base_url="http://dashboard"
        ) as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
            assert response.json()["ok"] is True

        await asyncio.wait_for(adapter._heartbeat_loop(), timeout=2)
        assert not adapter.is_connected
        assert adapter.fatal_error_code == "api_server_listener_lost"
        assert adapter.fatal_error_retryable is True
        assert Platform.API_SERVER not in runner.adapters
        assert runner._failed_platforms[Platform.API_SERVER]["predecessor"] is adapter

        replacement = APIServerAdapter(PlatformConfig(
            enabled=True, extra={"host": "127.0.0.1", "port": port, "key": "test-listener-supervision-key"}))
        runner._create_adapter = MagicMock(return_value=replacement)

        async def connect(a, platform, **kwargs):
            await _connect(a)
            return True

        runner._connect_adapter_with_timeout = connect
        with (
            patch("gateway.session_api_turn.recover_api_turns"),
            patch("gateway.channel_directory.build_channel_directory", new_callable=AsyncMock),
        ):
            await runner._reconnect_failed_platform(Platform.API_SERVER, time.monotonic())
        assert runner.adapters[Platform.API_SERVER] is replacement
        assert not runner._failed_platforms
        async with aiohttp.ClientSession() as client:
            async with client.get(f"http://127.0.0.1:{port}/health") as response:
                assert response.status == 200
    finally:
        if replacement is not None:
            await _close(replacement)
        await _close(adapter)


@pytest.mark.asyncio
async def test_intentional_disconnect_does_not_report_listener_loss():
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": "test-listener-supervision-key"}))
    notify = AsyncMock()
    adapter.set_fatal_error_handler(notify)
    try:
        await _connect(adapter)
        await adapter.disconnect()
        await adapter._heartbeat_loop()
        assert not adapter.has_fatal_error
        notify.assert_not_awaited()
    finally:
        await _close(adapter)
