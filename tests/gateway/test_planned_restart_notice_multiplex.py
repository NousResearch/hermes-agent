"""A planned restart notifies EVERY served profile's home channels, not just the launch profile's.

One host process multiplexes every profile, so ``self.config`` — the launch profile's — is not
the fleet: the owed set and the online notice were both built from it alone, and a secondary
profile's chat never heard that its gateway had restarted. The marker must also survive until
every served profile was reached, or the missed channels are lost for good.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.delivery as gateway_delivery
import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult

ONLINE_NOTICE = "♻️ Gateway online — Hermes is back and ready."


def _adapter():
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )


def _home_config(platform: Platform, chat_id: str) -> GatewayConfig:
    return GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(platform=platform, chat_id=chat_id, name=chat_id),
            )
        }
    )


@pytest.fixture
def multiplex_runner(tmp_path, monkeypatch):
    """A host multiplexer: launch profile on Discord, served profile ``coder`` on Telegram."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _home_config(Platform.DISCORD, "launch-home")
    runner.config.sessions_dir = tmp_path / "sessions"
    runner.adapters = {}
    runner._profile_configs = {"coder": _home_config(Platform.TELEGRAM, "coder-home")}
    runner._profile_adapters = {"coder": {}}
    runner._free_tier_startup_line = Mock(return_value=None)
    runner._planned_restart_notice_lock = None
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    return runner, marker


@pytest.mark.asyncio
async def test_planned_restart_notifies_every_served_profile(multiplex_runner):
    runner, marker = multiplex_runner
    launch, coder = _adapter(), _adapter()
    runner.adapters[Platform.DISCORD] = launch
    runner._profile_adapters["coder"][Platform.TELEGRAM] = coder

    await runner._replay_pending_planned_restart_notification()

    launch.send.assert_awaited_once()
    coder.send.assert_awaited_once(), "a served profile's home channel is owed the restart notice"
    assert coder.send.await_args.args[:2] == ("coder-home", ONLINE_NOTICE)
    assert not marker.exists(), "every owed target was notified — the obligation is discharged"


@pytest.mark.asyncio
async def test_marker_survives_until_a_served_profile_is_reachable(multiplex_runner):
    """A served profile whose platform is down at boot keeps the notice owed for its reconnect."""
    runner, marker = multiplex_runner
    launch = _adapter()
    runner.adapters[Platform.DISCORD] = launch

    await runner._replay_pending_planned_restart_notification()

    launch.send.assert_awaited_once()
    assert marker.exists(), "coder's channel was never notified; the marker must not be consumed"
    delivered = json.loads(marker.read_text(encoding="utf-8"))["delivered_targets"]
    assert [target for target in delivered if target[0] == "discord"], "the reached target is recorded"

    coder = _adapter()
    runner._profile_adapters["coder"][Platform.TELEGRAM] = coder
    await runner._replay_pending_planned_restart_notification()

    coder.send.assert_awaited_once()
    assert launch.send.await_count == 1, "a reached home is never notified twice"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_profiles_sharing_one_home_chat_get_one_notice(tmp_path, monkeypatch):
    """One host process restarting once owes a shared chat ONE notice, not one per profile.

    A single Telegram group as the home channel of both the launch profile and a served profile
    is a common setup; keyed per profile it received two "Gateway online" messages.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _home_config(Platform.TELEGRAM, "-100999")
    runner.config.sessions_dir = tmp_path / "sessions"
    launch, coder = _adapter(), _adapter()
    runner.adapters = {Platform.TELEGRAM: launch}
    runner._profile_configs = {"coder": _home_config(Platform.TELEGRAM, "-100999")}
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: coder}}
    runner._free_tier_startup_line = Mock(return_value=None)
    runner._planned_restart_notice_lock = None
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")

    await runner._replay_pending_planned_restart_notification()

    assert launch.send.await_count + coder.send.await_count == 1, "one chat, one restart, one notice"
    assert not marker.exists(), "the shared chat was reached, so every owed profile is discharged"


@pytest.mark.asyncio
async def test_one_broken_profile_does_not_starve_the_rest(multiplex_runner, monkeypatch):
    """A profile whose transport resolution raises is skipped; the fan-out continues."""
    runner, marker = multiplex_runner
    runner.adapters[Platform.DISCORD] = _adapter()
    ok = _adapter()
    runner._profile_configs = {
        "b": _home_config(Platform.TELEGRAM, "b-home"),
        "c": _home_config(Platform.SLACK, "c-home"),
    }
    runner._profile_adapters = {"b": {Platform.TELEGRAM: _adapter()}, "c": {Platform.SLACK: ok}}
    real = gateway_delivery.resolve_delivery_transport

    def resolve(platform, config, adapters):
        if platform is Platform.TELEGRAM:
            raise RuntimeError("broken adapter")
        return real(platform, config, adapters)

    monkeypatch.setattr(gateway_delivery, "resolve_delivery_transport", resolve)

    await runner._send_home_channel_startup_notifications()

    ok.send.assert_awaited_once(), "a profile after the broken one is still notified"


@pytest.mark.asyncio
async def test_unserved_profile_config_is_pruned_from_the_fan_out(tmp_path, monkeypatch):
    """A profile whose adapters failed keeps no cached config, or it is owed a notice forever.

    ``owed`` is built from ``_profile_configs`` while delivery needs a live transport, so a stale
    entry makes ``owed <= delivered`` permanently false and ``.restart_pending.json`` immortal.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _home_config(Platform.DISCORD, "launch-home")
    runner._profile_configs = {"ghost": _home_config(Platform.TELEGRAM, "-200")}
    runner._profile_adapters = {}
    runner._multiplex_on = Mock(return_value=True)
    runner._primary_resource_claims = Mock(return_value={})
    runner._record_served_profiles = Mock()
    runner._restore_secondary_completion_ledgers = Mock()
    runner._start_one_profile_adapters = AsyncMock(side_effect=RuntimeError("adapters failed"))
    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda cfg: [("ghost", tmp_path / "ghost")])
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "gateway.run_profile_reconcile.profile_serve_signature", lambda home: ("sig",))

    await runner._start_secondary_profile_adapters()

    assert "ghost" not in runner._profile_configs
    assert list(runner._served_home_channel_configs()) == [
        (None, Platform.DISCORD, runner.config.platforms[Platform.DISCORD])]


@pytest.mark.asyncio
async def test_a_served_profiles_reconnect_replays_the_owed_notice(multiplex_runner):
    """A served profile whose bot was down at boot keeps the notice owed "for its reconnect" -- but only
    the primary reconnect replayed it, so the marker outlived the outage and the notice never went out."""
    import asyncio

    runner, marker = multiplex_runner
    runner.adapters[Platform.DISCORD] = _adapter()
    await runner._replay_pending_planned_restart_notification()  # boot: coder's Telegram is down
    assert marker.exists()

    coder = SimpleNamespace(send_path_degraded=False, has_fatal_error=False, fatal_error_retryable=True,
                            send=AsyncMock(return_value=SendResult(success=True, message_id="n")))
    runner._running = True
    runner._background_tasks = set()
    runner._profile_failed_platforms = {}
    runner._failed_platforms = {}
    runner._sync_voice_mode_state_to_adapter = Mock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    runner._schedule_resume_pending_sessions = Mock(return_value=0)
    runner._secondary_reconnect_attempt = AsyncMock(return_value=(coder, True))

    await runner._run_secondary_profile_reconnect("coder", Platform.TELEGRAM)
    for _ in range(50):
        await asyncio.sleep(0)

    assert runner._profile_adapters["coder"][Platform.TELEGRAM] is coder
    coder.send.assert_awaited_once()
    assert coder.send.await_args.args[:2] == ("coder-home", ONLINE_NOTICE)
    assert not marker.exists(), "the owed target was reached on reconnect: the obligation is discharged"
