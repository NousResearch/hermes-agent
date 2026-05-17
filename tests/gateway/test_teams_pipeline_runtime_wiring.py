"""Tests for Teams pipeline runtime wiring into the gateway."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.teams_pipeline.runtime import (
    bind_gateway_runtime,
    build_pipeline_runtime,
    build_pipeline_runtime_config,
)


def test_gateway_runner_wires_teams_pipeline_runtime(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MSGRAPH_WEBHOOK: object()}
    runner._teams_pipeline_runtime_error = None

    calls: list[object] = []

    def _bind(gateway_runner):
        calls.append(gateway_runner)
        return True

    monkeypatch.setattr("plugins.teams_pipeline.runtime.bind_gateway_runtime", _bind)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"plugins": {"enabled": ["teams_pipeline"]}},
    )

    GatewayRunner._wire_teams_pipeline_runtime(runner)

    assert calls == [runner]


def test_gateway_runner_skips_wiring_without_msgraph_adapter(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._teams_pipeline_runtime_error = None

    called = False

    def _bind(_gateway_runner):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("plugins.teams_pipeline.runtime.bind_gateway_runtime", _bind)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"plugins": {"enabled": ["teams_pipeline"]}},
    )

    GatewayRunner._wire_teams_pipeline_runtime(runner)

    assert called is False


def test_gateway_runner_skips_wiring_when_teams_pipeline_plugin_disabled(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MSGRAPH_WEBHOOK: object()}
    runner._teams_pipeline_runtime_error = None

    called = False

    def _bind(_gateway_runner):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("plugins.teams_pipeline.runtime.bind_gateway_runtime", _bind)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"plugins": {"enabled": []}},
    )

    GatewayRunner._wire_teams_pipeline_runtime(runner)

    assert called is False


def test_runtime_config_disables_teams_delivery_without_target():
    gateway_config = SimpleNamespace(
        platforms={
            Platform("teams"): PlatformConfig(enabled=True, extra={}),
        }
    )

    config = build_pipeline_runtime_config(gateway_config)

    assert "teams_delivery" not in config


def test_build_pipeline_runtime_only_wires_sender_when_delivery_configured(monkeypatch):
    gateway = SimpleNamespace(
        config=SimpleNamespace(
            platforms={
                Platform("teams"): PlatformConfig(enabled=True, extra={}),
            }
        )
    )

    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.build_graph_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.resolve_teams_pipeline_store_path",
        lambda: "/tmp/teams-pipeline-store.json",
    )
    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.TeamsPipelineStore",
        lambda path: {"path": path},
    )

    runtime = build_pipeline_runtime(gateway)

    assert runtime.teams_sender is None


def test_build_pipeline_runtime_skips_sender_when_adapter_layer_is_unavailable(monkeypatch):
    gateway = SimpleNamespace(
        config=SimpleNamespace(
            platforms={
                Platform("teams"): PlatformConfig(
                    enabled=True,
                    extra={
                        "delivery_mode": "graph",
                        "team_id": "team-1",
                        "channel_id": "channel-1",
                    },
                ),
            }
        )
    )

    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.build_graph_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.resolve_teams_pipeline_store_path",
        lambda: "/tmp/teams-pipeline-store.json",
    )
    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.TeamsPipelineStore",
        lambda path: {"path": path},
    )
    monkeypatch.setitem(
        sys.modules,
        "plugins.platforms.teams.adapter",
        ModuleType("plugins.platforms.teams.adapter"),
    )

    runtime = build_pipeline_runtime(gateway)

    assert runtime.teams_sender is None


def test_bind_gateway_runtime_installs_drop_scheduler_on_failure(monkeypatch):
    """When the runtime can't build, install a drop-scheduler so Graph
    notifications still ack cleanly rather than leaving the adapter's
    scheduler unbound.
    """
    class FakeAdapter:
        def __init__(self):
            self.scheduler = None

        def set_notification_scheduler(self, scheduler):
            self.scheduler = scheduler

    gateway = SimpleNamespace(
        adapters={Platform.MSGRAPH_WEBHOOK: FakeAdapter()},
        config=SimpleNamespace(
            platforms={
                Platform("teams"): PlatformConfig(enabled=True, extra={}),
            }
        ),
        _teams_pipeline_runtime=None,
        _teams_pipeline_runtime_error=None,
    )

    monkeypatch.setattr(
        "plugins.teams_pipeline.runtime.build_pipeline_runtime",
        lambda _gateway: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    bound = bind_gateway_runtime(gateway)

    assert bound is False
    assert callable(gateway.adapters[Platform.MSGRAPH_WEBHOOK].scheduler)
    assert gateway._teams_pipeline_runtime_error == "boom"
