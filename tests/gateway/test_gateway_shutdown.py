from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_gateway_stop_disconnects_adapters(monkeypatch, tmp_path):
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    runner = GatewayRunner(config)
    runner._running = True

    adapter1 = AsyncMock()
    adapter2 = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter1, Platform.DISCORD: adapter2}

    # Avoid touching filesystem pid/status in unit test.
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.write_runtime_status", lambda **_kw: None)
    monkeypatch.setattr(runner, "_shutdown_all_gateway_honcho", lambda: None)

    await runner.stop()

    adapter1.disconnect.assert_awaited_once()
    adapter2.disconnect.assert_awaited_once()
    assert runner.adapters == {}
    assert runner._running is False
    assert runner._shutdown_event.is_set()
