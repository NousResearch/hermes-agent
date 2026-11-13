"""Tests for Teams pipeline runtime wiring into the gateway."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.plugins.teams_pipeline.runtime import (
    bind_gateway_runtime,
    build_pipeline_runtime,
    build_pipeline_runtime_config,
)


def test_gateway_runner_skips_wiring_without_msgraph_adapter(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._teams_pipeline_runtime_error = None

    called = False

    def _bind(_gateway_runner):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("hermes_agent.plugins.teams_pipeline.runtime.bind_gateway_runtime", _bind)
    monkeypatch.setattr(
        "hermes_agent.gateway.run._load_gateway_config",
        lambda: {"plugins": {"enabled": ["teams_pipeline"]}},
    )

    GatewayRunner._wire_teams_pipeline_runtime(runner)

    assert called is False


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
        "hermes_agent.plugins.teams_pipeline.runtime.build_graph_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "hermes_agent.plugins.teams_pipeline.runtime.resolve_teams_pipeline_store_path",
        lambda: "/tmp/teams-pipeline-store.json",
    )
    monkeypatch.setattr(
        "hermes_agent.plugins.teams_pipeline.runtime.TeamsPipelineStore",
        lambda path: {"path": path},
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_agent.plugins.platforms.teams.adapter",
        ModuleType("hermes_agent.plugins.platforms.teams.adapter"),
    )

    runtime = build_pipeline_runtime(gateway)

    assert runtime.teams_sender is None


