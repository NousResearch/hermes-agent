"""A webhook configuration error is reported as fatal, not retried for the life of the process.

The reconnect watcher rebuilds a failed adapter from the platform config captured at startup, so a
configuration refusal (a route with no HMAC secret, a Microsoft Graph listener with no
``client_state``) can never heal in-process. Reported as a bare connect failure it is queued as
transient and retried every 30-300 s forever, with the platform shown as ``retrying``. Each test
starts a real ``GatewayRunner`` with the real adapter built from config.
"""

import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


@pytest.fixture(autouse=True)
def _host_attach_gate_is_not_under_test(monkeypatch):
    monkeypatch.setattr("gateway.run._host_attach_or_none", AsyncMock(return_value=None))
    yield
    from gateway import host_rendezvous as hr

    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.clear_record(hr.ROLE_GATEWAY)


async def _start_with(platform: Platform, extra: dict, tmp_path) -> GatewayRunner:
    runner = GatewayRunner(GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra=extra)}, sessions_dir=tmp_path / "sessions"))
    await runner.start()
    return runner


def _assert_fatal_not_queued(runner: GatewayRunner, platform: Platform) -> None:
    assert platform not in runner._failed_platforms, "a configuration error was queued for endless reconnects"
    assert read_runtime_status()["platforms"][platform.value]["state"] == "fatal"


@pytest.mark.asyncio
async def test_webhook_route_without_secret_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = await _start_with(
        Platform.WEBHOOK, {"host": "127.0.0.1", "port": 0, "routes": {"github": {"prompt": "{body}"}}}, tmp_path)
    _assert_fatal_not_queued(runner, Platform.WEBHOOK)


@pytest.mark.asyncio
async def test_msgraph_webhook_without_client_state_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = await _start_with(Platform.MSGRAPH_WEBHOOK, {"host": "127.0.0.1", "port": 0}, tmp_path)
    _assert_fatal_not_queued(runner, Platform.MSGRAPH_WEBHOOK)
