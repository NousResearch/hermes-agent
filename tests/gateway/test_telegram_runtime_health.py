from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


@pytest.mark.asyncio
async def test_polling_network_error_marks_runtime_reconnecting_before_recovery(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(
            running=False,
            start_polling=AsyncMock(return_value=None),
        )
    )
    adapter._polling_error_callback_ref = object()

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", _no_sleep)

    with patch("gateway.status.write_runtime_status") as write_runtime_status:
        await adapter._handle_polling_network_error(RuntimeError("Bad Gateway"))

    states = [call.kwargs.get("platform_state") for call in write_runtime_status.call_args_list]
    assert "reconnecting" in states
    assert "connected" in states
    reconnecting_call = next(
        call for call in write_runtime_status.call_args_list if call.kwargs.get("platform_state") == "reconnecting"
    )
    assert reconnecting_call.kwargs["platform"] == "telegram"
    assert reconnecting_call.kwargs["error_code"] == "telegram_polling_reconnect"
    assert "Bad Gateway" in reconnecting_call.kwargs["error_message"]
