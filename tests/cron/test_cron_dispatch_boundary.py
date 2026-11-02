"""Cron fallback respects the real adapter's provider dispatch boundary."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig


def _deliver(monkeypatch, platform, adapter, content):
    config = GatewayConfig()
    config.platforms[platform] = PlatformConfig(enabled=True, token="test-token")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr("cron.scheduler_delivery._record_delivery_verification", lambda *a: None)
    monkeypatch.setattr("cron.scheduler_delivery._maybe_mirror_cron_delivery", lambda *a, **kw: None)
    standalone = []

    async def send_standalone(*args, **kwargs):
        standalone.append(args[3])
        return {"success": True, "message_id": "standalone-1"}

    def schedule(coro, loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as error:
            future.set_exception(error)
        return future

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send_standalone)
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", schedule)
    loop = MagicMock()
    loop.is_running.return_value = True
    job = {"id": "dispatch-boundary", "deliver": f"{platform.value}:123", "name": "Report"}
    error = _deliver_result(job, content, adapters={platform: adapter}, loop=loop)
    return error, standalone


def test_discord_chunk_accepted_before_transport_failure_is_not_replayed(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = MagicMock()
    channel = SimpleNamespace(send=AsyncMock(side_effect=[
        SimpleNamespace(id=1), ConnectionResetError("Cannot write to closing transport")]))
    monkeypatch.setattr(adapter, "_resolve_channel", AsyncMock(return_value=channel))
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda channel: False)
    monkeypatch.setattr(adapter, "truncate_message", lambda *a: ["accepted head", "uncertain tail"])
    monkeypatch.setattr(adapter, "_record_response_async", AsyncMock(side_effect=lambda reply, result, *a: result))

    error, standalone = _deliver(monkeypatch, Platform.DISCORD, adapter, "Report content")

    assert channel.send.await_count == 2
    assert standalone == []
    assert "delivery uncertain" in error


def test_slack_missing_app_rejection_delivers_through_standalone(monkeypatch):
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._app = None
    error, standalone = _deliver(monkeypatch, Platform.SLACK, adapter, "Report content")

    assert error is None
    assert standalone == ["Report content"]
