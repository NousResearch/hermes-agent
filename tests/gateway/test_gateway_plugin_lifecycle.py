from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_gateway_plugin_lifecycle_receives_live_runner_and_adapters(monkeypatch) -> None:
    runner = object.__new__(GatewayRunner)
    adapter = object()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    invoke = AsyncMock(return_value=[])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook_async", invoke)

    await runner._invoke_plugin_lifecycle("gateway_ready")

    invoke.assert_awaited_once_with(
        "gateway_ready",
        callback_timeout=10.0,
        gateway=runner,
        adapters=runner.adapters,
        profile_adapters=runner._profile_adapters,
    )


@pytest.mark.asyncio
async def test_gateway_plugin_lifecycle_failures_do_not_break_gateway(monkeypatch) -> None:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {}
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook_async",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    await runner._invoke_plugin_lifecycle("gateway_ready")


def test_start_and_stop_bridge_existing_gateway_lifecycle_to_plugins() -> None:
    start_source = inspect.getsource(GatewayRunner.start)
    stop_source = inspect.getsource(GatewayRunner.stop)

    assert 'await GatewayRunner._invoke_plugin_lifecycle(self, "gateway_ready")' in start_source
    assert 'await GatewayRunner._invoke_plugin_lifecycle(self, "gateway_stopping")' in stop_source
    assert stop_source.index("self._draining = True") < stop_source.index(
        'GatewayRunner._invoke_plugin_lifecycle(self, "gateway_stopping")'
    )
    assert stop_source.index("await self._finalize_shutdown_agents(active_agents)") < stop_source.index(
        'GatewayRunner._invoke_plugin_lifecycle(self, "gateway_stopping")'
    )
    assert stop_source.index(
        'GatewayRunner._invoke_plugin_lifecycle(self, "gateway_stopping")'
    ) < stop_source.index("for platform, adapter in list(self.adapters.items())")
