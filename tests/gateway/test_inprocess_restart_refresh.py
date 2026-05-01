from unittest.mock import AsyncMock, patch

import pytest

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_stop_impl_refreshes_unit_before_service_restart_exit_code(monkeypatch):
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    calls = []

    def fake_refresh():
        calls.append(runner._exit_code)

    monkeypatch.setattr(
        runner,
        "_refresh_installed_unit_before_service_restart",
        fake_refresh,
        raising=False,
    )

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop(restart=True, service_restart=True)

    assert calls == [None]
    assert runner._exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.asyncio
async def test_stop_impl_swallows_refresh_exception(monkeypatch):
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    calls = []

    def boom():
        calls.append("called")
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(
        runner,
        "_refresh_installed_unit_before_service_restart",
        boom,
        raising=False,
    )

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop(restart=True, service_restart=True)

    assert calls == ["called"]
    assert runner._exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE
