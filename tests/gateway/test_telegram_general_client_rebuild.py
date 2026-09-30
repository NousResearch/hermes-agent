"""A wedged general HTTP client must be REPLACED, not silently reused.

Root cause of the hub's endless `polling reconnect failed after 30.0s`
(2026-09-17): PTB's HTTPXRequest.initialize() only rebuilds its httpx client
when the old one reports is_closed. If shutdown() was abandoned on a CLOSE-WAIT
socket that flag stays False, initialize() is a no-op, and the dead client
survives the "drain". start_polling() bootstraps through the general pool, so
that dead client made every reconnect hang for the full deadline, forever.

The polling pool already rescued this via _orphan_and_rebuild_polling_client.
These tests pin the same rescue on the general pool.
"""
import asyncio
from unittest.mock import MagicMock

from gateway.platforms.base import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _WedgedClient:
    """Mimics httpx after an abandoned aclose(): still reports itself open."""

    is_closed = False

    async def aclose(self):
        await asyncio.sleep(3600)


def _adapter(*, shutdown_hangs, initialize_hangs=False):
    a = object.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a.config = PlatformConfig(enabled=True, token="x", extra={})
    a._background_tasks = set()

    req = MagicMock()
    req._client = _WedgedClient()
    req._build_client = lambda: _WedgedClient()

    async def _shutdown():
        if shutdown_hangs:
            await asyncio.sleep(3600)

    async def _initialize():
        if initialize_hangs:
            await asyncio.sleep(3600)

    req.shutdown = _shutdown
    req.initialize = _initialize

    bot = MagicMock()
    bot._request = (MagicMock(), req)
    a._app = MagicMock()
    a._app.bot = bot
    a._bot = bot
    return a, req


def _run(adapter):
    async def go():
        async def _step(awaitable, _msg):
            try:
                await asyncio.wait_for(awaitable, 0.05)
                return True
            except Exception:
                return False

        adapter._bounded_request_step = _step
        await TelegramAdapter._drain_general_connections_after_pool_timeout(adapter)

    asyncio.run(go())


def test_hung_shutdown_replaces_the_general_client():
    adapter, req = _adapter(shutdown_hangs=True)
    before = req._client
    _run(adapter)
    assert req._client is not before, (
        "a wedged general client must be replaced; reusing it is what made every "
        "Telegram reconnect hang for the full 30s deadline"
    )


def test_hung_initialize_also_replaces_the_general_client():
    adapter, req = _adapter(shutdown_hangs=False, initialize_hangs=True)
    before = req._client
    _run(adapter)
    assert req._client is not before


def test_healthy_drain_leaves_the_client_alone():
    adapter, req = _adapter(shutdown_hangs=False)
    before = req._client
    _run(adapter)
    assert req._client is before, "a healthy drain must not churn the client"
