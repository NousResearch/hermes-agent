"""The general Bot API pool must also be reset after a *generic* network error.

Only httpx pool timeouts used to qualify, on the reasoning that a pool timeout
proves the request was never sent. A plain ConnectError/TimedOut from a dead
tunnel leaves the same corpse behind — a CLOSE-WAIT socket that
``HTTPXRequest.initialize()`` refuses to rebuild because ``is_closed`` is still
false — and then every rung of the reconnect ladder and the heartbeat probe
times out on that socket until the whole process is restarted.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram import adapter as tg_adapter
from gateway.config import PlatformConfig


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _make_mock_app():
    mock_polling_req = AsyncMock()
    mock_bot = MagicMock()
    mock_bot._request = (mock_polling_req, MagicMock())  # (getUpdates, general)
    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = mock_bot
    return mock_app, mock_polling_req


def _open_client():
    """A client that still claims to be open — the CLOSE-WAIT case."""
    client = MagicMock()
    client.is_closed = False
    return client


@pytest.mark.asyncio
async def test_generic_network_error_also_resets_general_pool(monkeypatch):
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1
    mock_app, _ = _make_mock_app()
    adapter._app = mock_app

    drained = AsyncMock()
    monkeypatch.setattr(adapter, "_drain_general_connections_after_pool_timeout", drained)

    error = ConnectionError("proxy connect failed")
    # Not the pool-timeout path: the pre-fix condition would have skipped the reset.
    assert TelegramAdapter._looks_like_pool_timeout(error) is False
    assert TelegramAdapter._looks_like_network_error(error) is True

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await asyncio.wait_for(adapter._handle_polling_network_error(error), timeout=5)

    drained.assert_awaited_once()


@pytest.mark.asyncio
async def test_wedged_general_shutdown_rebuilds_the_client(monkeypatch):
    """A hanging ``shutdown()``/``initialize()`` must swap in a fresh client.

    ``initialize()`` only rebuilds when the current client reports ``is_closed``;
    an abandoned ``aclose()`` leaves that flag false, so the next request reuses
    the dead socket.
    """
    adapter = _make_adapter()
    mock_app, _ = _make_mock_app()
    adapter._app = mock_app

    general_req = AsyncMock()

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    general_req.shutdown = AsyncMock(side_effect=_hang)
    general_req.initialize = AsyncMock(side_effect=_hang)
    general_req._client = _open_client()
    mock_app.bot._request = (AsyncMock(), general_req)
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.01, raising=False)

    rebuilt = MagicMock()
    monkeypatch.setattr(adapter, "_orphan_and_rebuild_polling_client", rebuilt)

    await asyncio.wait_for(adapter._drain_general_connections_after_pool_timeout(), timeout=5)

    assert rebuilt.call_count == 2, "both the hung shutdown and the hung initialize must rebuild"
    assert all(call.args[0] is general_req for call in rebuilt.call_args_list)
