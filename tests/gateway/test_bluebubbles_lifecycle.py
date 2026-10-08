"""BlueBubbles lifecycle behavior."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.gateway.bluebubbles_test_support import _make_adapter

pytestmark = pytest.mark.usefixtures("_isolate_bluebubbles_environment")


class TestBlueBubblesConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_registration_failure_cleans_up_and_reports_disconnected(
        self, monkeypatch
    ):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(
                side_effect=[
                    {"status": 200},
                    {"data": {"private_api": True, "helper_connected": True}},
                ]
            ),
        )
        monkeypatch.setattr(adapter, "_register_webhook", AsyncMock(return_value=False))

        connected = await adapter.connect()

        assert connected is False
        assert adapter.is_connected is False
        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_listener_bind_failure_cleans_client_and_runner(self, monkeypatch):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        site.start.side_effect = OSError(13, "permission denied")
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(side_effect=[{"status": 200}, {"data": {}}]),
        )

        assert await adapter.connect() is False
        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_listener_start_cancellation_cleans_client_and_runner(
        self, monkeypatch
    ):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        site.start.side_effect = asyncio.CancelledError
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(side_effect=[{"status": 200}, {"data": {}}]),
        )

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()

        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outbound_only", [True, False])
async def test_listener_failure_preserves_gateway_hook(monkeypatch, outbound_only):
    import errno
    from aiohttp import web

    adapter = _make_adapter(monkeypatch)
    adapter._api_get = AsyncMock(return_value={"data": {}})
    register = AsyncMock(return_value=True)
    unregister = AsyncMock(return_value=False)
    monkeypatch.setattr(adapter, "_register_webhook", register)
    monkeypatch.setattr(adapter, "_unregister_webhook", unregister)
    error = errno.EADDRINUSE if outbound_only else errno.EACCES
    monkeypatch.setattr(
        web.TCPSite, "start", AsyncMock(side_effect=OSError(error, "bind failed"))
    )

    assert await adapter.connect() is outbound_only
    register.assert_not_awaited()
    assert adapter._runner is None
    assert (adapter.client is not None) is outbound_only
    if outbound_only:
        await adapter.disconnect()
        unregister.assert_not_awaited()
    assert adapter.client is None
