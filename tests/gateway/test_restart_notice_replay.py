"""Planned-restart online notice survives an offline home-channel transport at boot.

The ``.restart_pending.json`` marker used to be consumed in ``finally`` even when no live transport
existed for the home channel, so the "Gateway online" notice was never sent and never replayed.
See #112109. Runs the real boot pass, marker helpers, home-channel sender and DeliveryTransport.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.delivery as delivery
import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult

ONLINE_NOTICE = "♻️ Gateway online — Hermes is back and ready."


def _adapter():
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )


@pytest.fixture
def boot_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # Await the boot task to completion and propagate failures deterministically.
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="unit-test-home", name="Test home"),
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._failed_platforms = {}
    runner._sync_voice_mode_state_to_adapter = Mock()
    runner._bind_voice_input_callback = Mock()
    runner._update_platform_runtime_status = Mock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_resume_pending_sessions = Mock()
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", AsyncMock())
    # Unrelated conversation recovery and optional account-status text are isolated.
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._redeliver_claimed_obligations = AsyncMock(return_value=0)
    runner._free_tier_startup_line = Mock(return_value=None)
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    return runner, marker


async def _boot(runner):
    await runner._await_startup_boot_sends(
        planned_restart_notification_pending=gateway_run._planned_restart_notification_pending()
    )


async def _reconnect(runner, platform, adapter):
    runner._failed_platforms[platform] = {}
    await runner._install_reconnected_adapter(platform, adapter)
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True], ids=["offline-at-boot-replayed-on-reconnect", "live-at-boot"])
async def test_planned_restart_notice_reaches_home_channel(boot_notice, live):
    runner, marker = boot_notice
    adapter = _adapter()
    if live:
        runner.adapters[Platform.DISCORD] = adapter
    transport = delivery.resolve_delivery_transport(Platform.DISCORD, runner.config, runner.adapters)
    assert (transport is not None) is live

    await _boot(runner)

    runner._redeliver_claimed_obligations.assert_awaited_once_with([])
    if not live:
        adapter.send.assert_not_called()
        assert marker.exists(), "marker must survive a boot with no live transport"
        await _reconnect(runner, Platform.DISCORD, adapter)
    adapter.send.assert_awaited_once_with("unit-test-home", ONLINE_NOTICE, metadata={"non_conversational": True})
    assert not marker.exists()


@pytest.mark.asyncio
async def test_partial_delivery_is_persisted_and_not_repeated(boot_notice):
    runner, marker = boot_notice
    telegram, discord = _adapter(), _adapter()
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="other-home", thread_id="7", name="Other"),
    )
    runner.config.platforms[Platform.SLACK] = PlatformConfig(
        enabled=True, gateway_restart_notification=False,
        home_channel=HomeChannel(platform=Platform.SLACK, chat_id="muted-home", name="Muted"),
    )
    runner.adapters[Platform.TELEGRAM] = telegram

    await _boot(runner)

    telegram.send.assert_awaited_once()
    discord.send.assert_not_called()
    assert json.loads(marker.read_text(encoding="utf-8"))["delivered_targets"] == [["telegram", "other-home", "7"]]

    # A fresh process has no in-memory history: dedupe must come from the marker. The opted-out
    # Slack home is never owed a notice, so Discord's delivery completes the set.
    recovered = object.__new__(gateway_run.GatewayRunner)
    recovered.__dict__.update(runner.__dict__)
    await _reconnect(recovered, Platform.DISCORD, discord)

    discord.send.assert_awaited_once()
    telegram.send.assert_awaited_once()
    assert not marker.exists()

    # Nothing pending: a later reconnect stays silent.
    await _reconnect(recovered, Platform.DISCORD, discord)
    discord.send.assert_awaited_once()
