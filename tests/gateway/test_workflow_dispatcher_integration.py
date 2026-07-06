import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig
from gateway.delivery import DeliveryRouter
from gateway.run import GatewayRunner
from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli import workflows_dispatcher


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._running = True
    return runner


def test_workflow_dispatcher_disabled_does_not_tick(monkeypatch):
    runner = _runner()
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"workflow": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(workflows_dispatcher, "tick", lambda *, limit: calls.append(limit))

    asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0))

    assert calls == []


@pytest.mark.parametrize(
    "workflow_cfg",
    [
        {"dispatch_in_gateway": "false"},
        {"dispatch_in_gateway": "0"},
        "not-a-dict",
    ],
)
def test_workflow_dispatcher_string_false_and_non_dict_config_do_not_tick(
    monkeypatch,
    workflow_cfg,
):
    runner = _runner()
    calls = []

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"workflow": workflow_cfg})
    monkeypatch.setattr(workflows_dispatcher, "tick", lambda *, limit: calls.append(limit))

    async def fail_sleep(_delay):
        raise AssertionError("disabled workflow dispatcher should not sleep")

    asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0, sleep=fail_sleep))

    assert calls == []


@pytest.mark.parametrize("enabled_value", ["true", "1"])
def test_workflow_dispatcher_string_true_ticks(monkeypatch, enabled_value):
    runner = _runner()
    calls = []
    sleeps = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "workflow": {
                "dispatch_in_gateway": enabled_value,
                "tick_interval_seconds": 1,
                "max_executions_per_tick": 4,
            }
        },
    )
    monkeypatch.setattr(workflows_dispatcher, "tick", lambda *, limit: calls.append(limit) or 1)

    async def fake_sleep(delay):
        sleeps.append(delay)
        runner._running = False

    asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0, sleep=fake_sleep))

    assert calls == [4]
    assert sleeps == [1.0]


@pytest.mark.parametrize("raw_interval", ["nan", "inf", "-inf"])
def test_workflow_dispatch_settings_reject_non_finite_intervals(raw_interval, caplog):
    def load_config():
        return {
            "workflow": {
                "dispatch_in_gateway": True,
                "tick_interval_seconds": raw_interval,
            }
        }

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        enabled, interval, limit = gateway_run._resolve_workflow_dispatch_settings(load_config)

    assert enabled is True
    assert interval == 30.0
    assert limit == 50
    assert "invalid tick_interval_seconds" in caplog.text


def test_workflow_dispatch_settings_reject_oversized_integer_interval(caplog):
    def load_config():
        return {
            "workflow": {
                "dispatch_in_gateway": True,
                "tick_interval_seconds": 10**400,
            }
        }

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        enabled, interval, limit = gateway_run._resolve_workflow_dispatch_settings(load_config)

    assert enabled is True
    assert interval == 30.0
    assert limit == 50
    assert "invalid tick_interval_seconds" in caplog.text


def test_workflow_dispatcher_enabled_ticks_on_cadence(monkeypatch):
    runner = _runner()
    calls = []
    sleeps = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "workflow": {
                "dispatch_in_gateway": True,
                "tick_interval_seconds": 2,
                "max_executions_per_tick": 7,
            }
        },
    )
    monkeypatch.setattr(workflows_dispatcher, "tick", lambda *, limit: calls.append(limit) or 1)

    async def fake_sleep(delay):
        sleeps.append(delay)
        runner._running = False

    asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0, sleep=fake_sleep))
    assert calls == [7]
    assert sleeps == [2.0]


def test_workflow_dispatcher_failure_is_logged_and_loop_survives(monkeypatch, caplog):
    runner = _runner()
    sleeps = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "workflow": {
                "dispatch_in_gateway": True,
                "tick_interval_seconds": 1,
                "max_executions_per_tick": 3,
            }
        },
    )

    def fail_tick(*, limit):
        raise RuntimeError(f"boom limit={limit}")

    monkeypatch.setattr(workflows_dispatcher, "tick", fail_tick)

    async def fake_sleep(delay):
        sleeps.append(delay)
        runner._running = False

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0, sleep=fake_sleep))

    assert sleeps == [1.0]
    assert "workflow dispatcher: tick failed" in caplog.text
    assert "boom limit=3" in caplog.text


def test_workflow_dispatcher_cancelled_error_propagates(monkeypatch):
    runner = _runner()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"workflow": {"dispatch_in_gateway": True}},
    )

    def cancel_tick(*, limit):
        raise asyncio.CancelledError()

    monkeypatch.setattr(workflows_dispatcher, "tick", cancel_tick)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._workflow_dispatcher_watcher(initial_delay=0, sleep=AsyncMock()))


def test_start_schedules_workflow_dispatcher_watcher(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(sessions_dir=tmp_path, platforms={})
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._failed_platforms = {}
    runner.delivery_router = DeliveryRouter(runner.config, {})
    runner.session_store = MagicMock()
    runner.session_store.suspend_recently_active.return_value = 0
    runner.hooks = MagicMock()
    runner.hooks.loaded_hooks = []
    runner.hooks.emit = AsyncMock()
    runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
    runner._suspend_stuck_loop_sessions = MagicMock(return_value=0)
    runner._start_secondary_profile_adapters = AsyncMock(return_value=0)
    runner._wire_teams_pipeline_runtime = MagicMock()
    runner._update_runtime_status = MagicMock()
    runner._send_update_notification = AsyncMock(return_value=True)
    runner._send_restart_notification = AsyncMock()
    runner._schedule_resume_pending_sessions = MagicMock()
    runner._finish_startup_restore = AsyncMock()
    runner._scale_to_zero_should_arm = MagicMock(return_value=False)
    runner._log_scale_to_zero_not_armed_reason = MagicMock()
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro.cr_code.co_name)
        coro.close()
        return MagicMock()

    with patch("gateway.status.write_runtime_status"):
        with patch("hermes_cli.plugins.discover_plugins"):
            with patch("hermes_cli.config.load_config", return_value={}):
                with patch("agent.shell_hooks.register_from_config"):
                    with patch(
                        "tools.process_registry.process_registry.recover_from_checkpoint",
                        return_value=0,
                    ):
                        with patch(
                            "gateway.channel_directory.build_channel_directory",
                            new=AsyncMock(return_value={"platforms": {}}),
                        ):
                            with patch("gateway.run.asyncio.create_task", side_effect=fake_create_task):
                                assert asyncio.run(runner.start()) is True

    assert "_workflow_dispatcher_watcher" in scheduled


def test_workflow_config_defaults_are_disabled():
    assert DEFAULT_CONFIG["workflow"] == {
        "dispatch_in_gateway": False,
        "tick_interval_seconds": 30,
        "max_executions_per_tick": 50,
    }
