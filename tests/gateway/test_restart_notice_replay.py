"""Planned-restart online notice survives an offline home-channel transport at boot.

The ``.restart_pending.json`` marker used to be consumed in ``finally`` even when no live transport
existed for the home channel, so the "Gateway online" notice was never sent and never replayed.
See #112109. Runs the real boot pass, marker helpers, home-channel sender and DeliveryTransport.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery
import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult

ONLINE_NOTICE = "♻️ Gateway online — Hermes is back and ready."


def _adapter():
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )


def _degraded_adapter():
    """Telegram reconnect publishes the adapter while the send path is degraded.

    ``send()`` short-circuits to ``send_path_degraded`` without contacting Telegram.
    """
    return SimpleNamespace(
        send_path_degraded=True,
        send=AsyncMock(return_value=SendResult(success=False, error="send_path_degraded", retryable=True)),
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


@pytest.mark.asyncio
async def test_degraded_transport_is_unreachable_so_aged_marker_expires(boot_notice):
    """A send-degraded transport must not pin an aged marker (#129707 review).

    ``resolve_delivery_transport`` only proves the adapter is present; a Telegram
    reconnect publishes it while ``send_path_degraded=True`` and ``send()`` returns
    ``send_path_degraded`` without contacting Telegram. The replay records no
    delivery, so the aged unreachable residue must expire instead of lingering
    until another process restart.
    """
    runner, marker = boot_notice
    adapter = _degraded_adapter()
    runner.adapters[Platform.DISCORD] = adapter
    marker.write_text(json.dumps({"requested_at": time.time() - 3900}), encoding="utf-8")

    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once_with("unit-test-home", ONLINE_NOTICE, metadata={"non_conversational": True})
    assert not marker.exists(), "aged marker with only a send-degraded transport must expire"


@pytest.mark.asyncio
async def test_fresh_marker_survives_degraded_then_recovery_delivers(boot_notice):
    """A fresh marker survives a degraded send and is delivered after in-place recovery."""
    runner, marker = boot_notice
    adapter = _degraded_adapter()
    runner.adapters[Platform.DISCORD] = adapter
    marker.write_text(json.dumps({"requested_at": time.time()}), encoding="utf-8")

    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once()
    assert marker.exists(), "fresh marker must survive a degraded send for the recovery replay"

    # In-place recovery: polling proves the send path and the adapter can send again.
    adapter.send_path_degraded = False
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="recovered"))
    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once_with("unit-test-home", ONLINE_NOTICE, metadata={"non_conversational": True})
    assert not marker.exists(), "recovered transport must deliver the owed notice"


def _bare_telegram_adapter_for_recovery():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_teardown_started = False
    a._polling_progress_accepting = True
    a._polling_generation = 1
    a._polling_progress_event = asyncio.Event()
    a._polling_network_error_count = 0
    a._polling_conflict_count = 0
    a._polling_conflict_recovery_generation = None
    a._send_path_degraded = True
    a._running = True
    a._write_runtime_status_safe = Mock()
    return a


@pytest.mark.asyncio
async def test_record_polling_progress_schedules_restart_replay_on_recovery():
    """Clearing the degraded flag in place schedules a planned-restart replay.

    Without this, a notice that failed with ``send_path_degraded`` stays pending
    until another process restart even though polling already recovered.
    """
    adapter = _bare_telegram_adapter_for_recovery()
    schedule = Mock()
    adapter.gateway_runner = SimpleNamespace(_schedule_planned_restart_replay=schedule)

    assert adapter._record_polling_progress(1) is True
    assert adapter._send_path_degraded is False
    schedule.assert_called_once_with()

    # No degraded -> no recovery -> no extra replay.
    schedule.reset_mock()
    assert adapter._record_polling_progress(1) is True
    schedule.assert_not_called()

    # Stale generation never records progress and never schedules.
    adapter._send_path_degraded = True
    schedule.reset_mock()
    assert adapter._record_polling_progress(0) is False
    schedule.assert_not_called()
    assert adapter._send_path_degraded is True
