"""An invalid built-in adapter must not prevent healthy siblings from connecting."""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner, MultiplexConfigError
from gateway.status import flush_runtime_status_async, read_runtime_status


class _HealthyAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        self._mark_connected()
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(tmp_path, monkeypatch, port):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.WEBHOOK: PlatformConfig.from_dict({"enabled": True, "port": port}),
            Platform.SLACK: PlatformConfig.from_dict({"enabled": True}),
        },
        multiplex_profiles=False,
        sessions_dir=tmp_path / "sessions",
    )
    runner.session_store = MagicMock()
    runner._shutdown_event = asyncio.Event()
    runner._failed_platforms = {}
    runner._platform_lock_takeover_on_start = False
    healthy = _HealthyAdapter(runner.config.platforms[Platform.SLACK], Platform.SLACK)
    # Model an installed optional transport without loading aiohttp or binding a
    # listener. The built-in lookup and WebhookAdapter constructor stay real.
    monkeypatch.setattr("gateway.platforms.webhook.check_webhook_requirements", lambda: True)
    monkeypatch.setattr(platform_registry, "is_registered", lambda name: name == "slack")
    monkeypatch.setattr(platform_registry, "create_adapter", lambda name, config: healthy)
    return runner, healthy


@pytest.mark.asyncio
@pytest.mark.parametrize("port", ["abc", None])
async def test_invalid_webhook_constructor_does_not_block_healthy_sibling(tmp_path, monkeypatch, caplog, port):
    runner, healthy = _runner(tmp_path, monkeypatch, port)
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        aborted, enabled, skipped, pending = await runner._start_prefilter_platforms()
    assert not aborted and enabled == 2 and not skipped
    assert [platform for platform, _, _ in pending] == [Platform.SLACK]
    assert callable(healthy._message_handler)
    results = await runner._start_connect_pending(pending)
    assert results[0][0] == Platform.SLACK
    assert results[0][3] == "ok"
    assert runner._serving_state() == "degraded"
    assert Platform.WEBHOOK not in runner._failed_platforms
    await flush_runtime_status_async()
    failed = read_runtime_status()["platforms"]["webhook"]
    assert failed["state"] == "fatal" and failed["needs_attention"]
    assert failed["error_code"] == "adapter_creation_failed"
    assert "restart" in failed["error_message"]
    assert any("webhook" in record.getMessage() and record.exc_info for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [MultiplexConfigError("invalid routing"), asyncio.CancelledError(), SystemExit(78)])
async def test_global_startup_failures_are_not_isolated(tmp_path, monkeypatch, error):
    runner, _ = _runner(tmp_path, monkeypatch, "abc")

    def fail(platform, config):
        raise error

    monkeypatch.setattr(runner, "_create_adapter", fail)
    with pytest.raises(type(error)):
        await runner._start_prefilter_platforms()
