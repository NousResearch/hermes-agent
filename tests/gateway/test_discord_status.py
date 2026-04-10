from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.discord import DiscordAdapter


def test_disconnect_marks_runtime_status(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(token="test-token"))
    client = AsyncMock()
    adapter._client = client
    adapter._token_lock_identity = "test-token"

    release_mock = MagicMock()
    status_mock = MagicMock()

    monkeypatch.setattr("gateway.status.release_scoped_lock", release_mock)
    monkeypatch.setattr("gateway.status.write_runtime_status", status_mock)

    adapter._mark_connected()

    assert adapter.is_connected is True
    status_mock.assert_called_with(
        platform="discord",
        platform_state="connected",
        error_code=None,
        error_message=None,
    )

    status_mock.reset_mock()

    import asyncio

    asyncio.run(adapter.disconnect())

    assert adapter.is_connected is False
    client.close.assert_awaited_once()
    release_mock.assert_called_once_with("discord-bot-token", "test-token")
    status_mock.assert_called_once_with(
        platform="discord",
        platform_state="disconnected",
        error_code=None,
        error_message=None,
    )
