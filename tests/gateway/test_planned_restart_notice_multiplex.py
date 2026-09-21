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
