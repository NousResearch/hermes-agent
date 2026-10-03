"""Real bind conflicts must recover during startup, or park before the outer timeout.

The previous two-second retry window lost a listener released eight seconds later.
Use the real gateway startup/aggregation path and isolated HERMES_HOME; only the
unrelated messaging transport and secondary-profile discovery are stubbed.
"""
import asyncio
import logging
import socket
from unittest.mock import AsyncMock

import aiohttp
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _HealthyPeer(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        self._mark_connected()
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, port):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.API_SERVER: PlatformConfig(enabled=True, extra={
                "host": "127.0.0.1", "port": port, "key": "bind-fixture-not-a-real-secret",
            }),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    api = APIServerAdapter(config.platforms[Platform.API_SERVER])
    peer = _HealthyPeer(config.platforms[Platform.TELEGRAM], Platform.TELEGRAM)
    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: api if p == Platform.API_SERVER else peer)
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", AsyncMock(return_value=0))
    return runner, api


@pytest.mark.asyncio
async def test_startup_recovers_when_predecessor_releases_port(monkeypatch, tmp_path, caplog):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        runner, api = _runner(monkeypatch, tmp_path, port)
        first_bind = asyncio.Event()
        real_start = aiohttp.web.TCPSite.start

        async def observed_start(site):
            first_bind.set()
            return await real_start(site)

        monkeypatch.setattr(aiohttp.web.TCPSite, "start", observed_start)

        async def release_port():
            await first_bind.wait()
            await asyncio.sleep(8)
            holder.close()

        releaser = asyncio.create_task(release_port())
        try:
            with caplog.at_level(logging.INFO):
                assert await runner.start() is True
            state = read_runtime_status()
            assert state["platforms"]["api_server"]["state"] == "connected", caplog.text
            assert state["gateway_state"] == "running"
            assert not api.has_fatal_error
            assert "retrying bind" in caplog.text
            assert "API server listening on" in caplog.text
            async with aiohttp.ClientSession() as client:
                async with client.get(f"http://127.0.0.1:{port}/health") as response:
                    assert response.status == 200
                    assert (await response.json())["status"] == "ok"
            assert len(api._runner.sites) == 1
        finally:
            releaser.cancel()
            await asyncio.gather(releaser, return_exceptions=True)
            await runner.stop()


@pytest.mark.asyncio
async def test_persistent_conflict_parks_instead_of_outer_timeout(monkeypatch, tmp_path, caplog):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        runner, api = _runner(monkeypatch, tmp_path, holder.getsockname()[1])
        held_since = asyncio.get_running_loop().time()
        try:
            with caplog.at_level(logging.INFO):
                assert await runner.start() is True
            # Keep the conflict beyond the outer connect deadline, then prove
            # the terminal failure stayed parked rather than silently retrying.
            await asyncio.sleep(max(0, 31 - (asyncio.get_running_loop().time() - held_since)))
            state = read_runtime_status()
            assert state["gateway_state"] == "degraded"
            assert state["platforms"]["api_server"]["error_code"] == "api_server_port_in_use"
            assert state["platforms"]["api_server"]["state"] == "fatal"
            assert Platform.API_SERVER not in runner._failed_platforms
            assert not api.fatal_error_retryable
            assert api._runner is None
            assert api._site is None
            assert any(r.levelno == logging.INFO and "retrying bind" in r.message for r in caplog.records)
            assert any(r.levelno == logging.ERROR and "Could not bind" in r.message for r in caplog.records)
            assert "parked" in caplog.text
        finally:
            await runner.stop()
