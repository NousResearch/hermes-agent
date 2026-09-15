"""Regression for #112109: retain planned-restart notices until delivery succeeds.

Uses the real boot notification pass, marker helpers, home-channel sender, and
DeliveryTransport. Patterns follow tests/gateway/test_restart_notification.py
and test_restart_resume_pending.py. Run with scripts/run_tests.sh for isolation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio
import json

import pytest

import gateway.delivery as delivery
import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult


@pytest.fixture
def boot_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # Await the boot task to completion and propagate failures deterministically.
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    runner = object.__new__(gateway_run.GatewayRunner)
    platform_config = PlatformConfig(
        enabled=True,
        gateway_restart_notification=True,
        home_channel=HomeChannel(
            platform=Platform.DISCORD, chat_id="unit-test-home", name="Test home"
        ),
    )
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: platform_config},
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
    # Keep the real requester-marker check; this case has only the planned marker.
    assert not (tmp_path / ".restart_notify.json").exists()
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    adapter = SimpleNamespace(send_path_degraded=False, send=AsyncMock(
        return_value=SendResult(success=True, message_id="unit-test-notice")
    ))
    return runner, platform_config, marker, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True], ids=[
    "no-live-transport-retained-and-replayed",
    "live-transport-notice-sent-marker-consumed",
])
async def test_planned_restart_boot_notice(boot_notice, monkeypatch, live):
    runner, platform_config, marker, adapter = boot_notice
    if live:
        runner.adapters[Platform.DISCORD] = adapter
    transport = (
        delivery.DeliveryTransport(adapter, platform_config, Platform.DISCORD)
        if live else None
    )
    # Confirm the stub represents the real resolver's result for this adapter map.
    resolved = delivery.resolve_delivery_transport(
        Platform.DISCORD, runner.config, runner.adapters
    )
    assert resolved == transport
    resolver = Mock(return_value=transport)
    monkeypatch.setattr(delivery, "resolve_delivery_transport", resolver)
    assert marker.exists()
    assert gateway_run._planned_restart_notification_pending()

    await runner._await_startup_boot_sends(
        planned_restart_notification_pending=gateway_run._planned_restart_notification_pending()
    )

    resolver.assert_called_once_with(Platform.DISCORD, runner.config, runner.adapters)
    runner._claim_pending_obligations.assert_awaited_once_with()
    runner._redeliver_claimed_obligations.assert_awaited_once_with([])
    if live:
        adapter.send.assert_awaited_once_with(
            "unit-test-home",
            "♻️ Gateway online — Hermes is back and ready.",
            metadata={"non_conversational": True},
        )
    else:
        adapter.send.assert_not_called()
    if not live:
        assert marker.exists()
        assert gateway_run._planned_restart_notification_pending()
        resolver.return_value = delivery.DeliveryTransport(adapter, platform_config, Platform.DISCORD)
        runner._failed_platforms[Platform.DISCORD] = {}
        await runner._install_reconnected_adapter(Platform.DISCORD, adapter)
        await asyncio.gather(*runner._background_tasks)
        adapter.send.assert_awaited_once_with(
            "unit-test-home", "♻️ Gateway online — Hermes is back and ready.",
            metadata={"non_conversational": True},
        )
    assert not marker.exists()
    assert not gateway_run._planned_restart_notification_pending()


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", ["unavailable", "rejected", "exception", "cancelled"])
async def test_partial_notice_delivery_survives_restart_and_concurrent_replay(boot_notice, outage):
    runner, _, marker, adapter = boot_notice
    other = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)))
    # Deliver one destination first, then encounter an unavailable/failing/hung transport.
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="other-home", thread_id="7", name="Other home"),
        ),
        **runner.config.platforms,
        Platform.SLACK: PlatformConfig(
            enabled=True, gateway_restart_notification=False,
            home_channel=HomeChannel(platform=Platform.SLACK, chat_id="muted-home", name="Muted home"),
        ),
    }
    runner.adapters[Platform.TELEGRAM] = other
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_send(*args, **kwargs):
        started.set()
        await release.wait()
        return SendResult(success=True)

    if outage != "unavailable":
        runner.adapters[Platform.DISCORD] = adapter
    if outage == "rejected":
        adapter.send.return_value = SendResult(success=False, error="temporarily unavailable")
    elif outage == "exception":
        adapter.send.side_effect = RuntimeError("transport disconnected")
    elif outage == "cancelled":
        adapter.send.side_effect = slow_send

    boot = asyncio.create_task(runner._await_startup_boot_sends(planned_restart_notification_pending=True))
    if outage == "cancelled":
        await asyncio.wait_for(started.wait(), timeout=5)
        # Acknowledgments must already be on disk while a later destination is hung.
        assert json.loads(marker.read_text())["delivered_targets"] == [["telegram", "other-home", "7"]]
        boot.cancel()
        with pytest.raises(asyncio.CancelledError):
            await boot
    else:
        await boot
    data = json.loads(marker.read_text())
    assert data["delivered_targets"] == [["telegram", "other-home", "7"]]
    assert data["pending_targets"] == [["discord", "unit-test-home", None]]
    other.send.assert_awaited_once()

    # A new runner has no in-memory delivery history: dedupe must come from the marker.
    recovered = object.__new__(gateway_run.GatewayRunner)
    recovered.__dict__.update(runner.__dict__)
    recovered.__dict__.pop("_planned_restart_notice_lock", None)
    started.clear()
    adapter.send.reset_mock()
    adapter.send.side_effect = slow_send
    recovered._failed_platforms[Platform.DISCORD] = {}
    await asyncio.wait_for(recovered._install_reconnected_adapter(Platform.DISCORD, adapter), timeout=5)
    await asyncio.wait_for(started.wait(), timeout=5)
    # Installation completed even though notification delivery is still blocked.
    assert marker.exists()
    concurrent = asyncio.create_task(recovered._replay_pending_planned_restart_notification())
    release.set()
    await asyncio.gather(concurrent, *recovered._background_tasks)
    assert not marker.exists()
    adapter.send.assert_awaited_once()
    other.send.assert_awaited_once()

    recovered._failed_platforms[Platform.DISCORD] = {}
    await recovered._install_reconnected_adapter(Platform.DISCORD, adapter)
    await asyncio.gather(*recovered._background_tasks)
    adapter.send.assert_awaited_once()
    other.send.assert_awaited_once()
