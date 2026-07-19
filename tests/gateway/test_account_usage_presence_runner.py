from contextlib import nullcontext
from unittest.mock import AsyncMock

import pytest

from gateway.config import AccountUsagePresenceConfig, GatewayConfig
from gateway.run import GatewayRunner


class _Controller:
    instances = []

    def __init__(self, config, adapters, **kwargs):
        self.config = config
        self.adapters = adapters
        self.kwargs = kwargs
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.recover_saved_baselines = AsyncMock()
        self.__class__.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_instances():
    _Controller.instances.clear()


def _runner(config):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {"telegram": object()}
    runner._profile_adapters = {}
    runner._profile_runtime_scope = lambda profile: nullcontext()
    runner._account_usage_presence_controller = None
    return runner


@pytest.mark.asyncio
async def test_runner_starts_account_usage_presence_with_live_adapter_getter(monkeypatch):
    monkeypatch.setattr("gateway.run.AccountUsagePresenceController", _Controller)
    config = GatewayConfig(
        account_usage_presence=AccountUsagePresenceConfig.from_dict(
            {
                "enabled": True,
                "provider": "anthropic",
                "platforms": ["telegram"],
            }
        )
    )
    runner = _runner(config)

    await runner._start_account_usage_presence()

    controller = _Controller.instances[0]
    assert controller.config == config.account_usage_presence
    assert controller.adapters() is runner.adapters
    controller.start.assert_awaited_once_with()
    assert runner._account_usage_presence_controller is controller


@pytest.mark.asyncio
async def test_runner_multiplex_mode_only_runs_disabled_recovery(monkeypatch):
    monkeypatch.setattr("gateway.run.AccountUsagePresenceController", _Controller)
    config = GatewayConfig(
        multiplex_profiles=True,
        account_usage_presence=AccountUsagePresenceConfig.from_dict(
            {
                "enabled": True,
                "provider": "openai-codex",
                "platforms": ["telegram", "discord"],
            }
        ),
    )
    runner = _runner(config)
    secondary = {"telegram": object()}
    runner._profile_adapters = {"work": secondary}

    await runner._start_account_usage_presence()

    assert len(_Controller.instances) == 2
    secondary_controller, controller = _Controller.instances
    assert secondary_controller.config.enabled is False
    secondary_controller.recover_saved_baselines.assert_awaited_once_with()
    assert controller.config.enabled is False
    assert controller.config.is_configured is False
    controller.start.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_runner_notifies_presence_controller_after_adapter_reconnect():
    config = GatewayConfig()
    runner = _runner(config)
    controller = _Controller(config.account_usage_presence, lambda: runner.adapters)
    runner._account_usage_presence_controller = controller

    await runner._notify_account_usage_presence_adapters_changed()

    controller.recover_saved_baselines.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_runner_stops_and_drops_account_usage_presence_controller():
    config = GatewayConfig()
    runner = _runner(config)
    controller = _Controller(config.account_usage_presence, lambda: runner.adapters)
    runner._account_usage_presence_controller = controller

    await runner._stop_account_usage_presence()

    controller.stop.assert_awaited_once_with()
    assert runner._account_usage_presence_controller is None


@pytest.mark.asyncio
async def test_runner_capacity_stop_is_idempotent():
    runner = _runner(GatewayConfig())

    await runner._stop_account_usage_presence()

    assert runner._account_usage_presence_controller is None


@pytest.mark.asyncio
async def test_runner_real_controller_lifecycle_with_journal(tmp_path):
    """Exercise real controller wiring through GatewayRunner stop path."""

    import json
    from datetime import datetime, timezone

    from agent.account_usage import AccountUsageFetchOutcome, AccountUsageSnapshot, AccountUsageWindow
    from gateway.account_usage_presence import (
        AccountUsagePresenceApplyResult,
        AccountUsagePresenceCapabilities,
        AccountUsagePresenceController,
        AccountUsagePresenceRestoreResult,
        account_usage_presence_state_path,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    class _LiveAdapter:
        def __init__(self):
            self.current_name = "Hermes"

        @property
        def account_usage_presence_capabilities(self):
            return AccountUsagePresenceCapabilities(display_name=True)

        def account_usage_presence_state_key(self):
            return "telegram:1"

        async def capture_account_usage_presence_baseline(self):
            return {"display_name": self.current_name}

        def build_account_usage_presence_owned_state(self, payload, baseline):
            return {
                "display_name": (
                    f"{baseline['display_name']} · {payload.label} "
                    f"{payload.remaining_percent}%"
                )
            }

        async def apply_account_usage_presence(self, payload, baseline):
            owned = self.build_account_usage_presence_owned_state(payload, baseline)
            self.current_name = owned["display_name"]
            return True

        async def apply_account_usage_presence_if_owned(
            self,
            payload,
            baseline,
            expected_owned,
        ):
            if self.current_name != expected_owned["display_name"]:
                return AccountUsagePresenceApplyResult.EXTERNAL
            await self.apply_account_usage_presence(payload, baseline)
            return AccountUsagePresenceApplyResult.APPLIED

        async def restore_account_usage_presence(self, baseline, owned):
            if self.current_name == baseline["display_name"]:
                return AccountUsagePresenceRestoreResult.ALREADY_BASELINE
            if self.current_name != owned["display_name"]:
                return AccountUsagePresenceRestoreResult.EXTERNAL
            self.current_name = baseline["display_name"]
            return AccountUsagePresenceRestoreResult.RESTORED

    token = set_hermes_home_override(tmp_path)
    try:
        snapshot = AccountUsageSnapshot(
            provider="openai-codex",
            source="test",
            fetched_at=datetime.now(timezone.utc),
            windows=(AccountUsageWindow(label="Session", used_percent=25.0),),
        )
        config = GatewayConfig(
            account_usage_presence=AccountUsagePresenceConfig.from_dict(
                {
                    "enabled": True,
                    "provider": "openai-codex",
                    "platforms": ["telegram"],
                }
            )
        )
        runner = _runner(config)
        adapter = _LiveAdapter()
        runner.adapters = {"telegram": adapter}
        controller = AccountUsagePresenceController(
            config.account_usage_presence,
            lambda: runner.adapters,
            fetcher=lambda provider: AccountUsageFetchOutcome(snapshot=snapshot),
            state_path=account_usage_presence_state_path(),
        )
        runner._account_usage_presence_controller = controller

        await controller.refresh_once()
        assert adapter.current_name == "Hermes · Session 75%"
        journal = tmp_path / "state" / "account-usage-presence" / "journal.json"
        assert journal.is_file()

        await runner._stop_account_usage_presence()
        assert adapter.current_name == "Hermes"
        assert runner._account_usage_presence_controller is None
        assert journal.exists()
        assert json.loads(journal.read_text(encoding="utf-8"))["entries"] == {}
    finally:
        reset_hermes_home_override(token)
