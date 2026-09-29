"""Regression for #30920: the live adapter must honor its explicit proxy."""
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gateway.config import PlatformConfig
from gateway.platforms import base
from plugins.platforms.mattermost.adapter import MattermostAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["explicit", "direct", "bypass", "connector"])
async def test_live_operations_share_the_configured_proxy(monkeypatch, tmp_path, route):
    proxy = "http://127.0.0.1:18765"
    monkeypatch.setenv("MATTERMOST_PROXY", proxy if route != "direct" else "")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "mm.example.test" if route == "bypass" else "")
    monkeypatch.setattr(base, "gateway_trust_env", lambda: False)
    connector = object() if route == "connector" else None
    monkeypatch.setattr(base, "_aiohttp_socks_connector", lambda url: connector)
    # No sockets: only the final transport is replaced; proxy resolution is real.
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"id": "bot", "file_infos": [{"id": "file"}]})
    response.read = AsyncMock(return_value=b"attachment")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock(closed=False)
    for method in ("get", "post", "put", "delete"):
        getattr(session, method).return_value = context
    session.close = AsyncMock()
    ws = MagicMock()
    ws.__aiter__.return_value = []
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    session.ws_connect = AsyncMock(return_value=ws)
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    monkeypatch.setattr(base, "cache_document_from_bytes_async", AsyncMock(return_value=str(tmp_path / "file")))
    adapter = MattermostAdapter(PlatformConfig(token="fixture-token", extra={"url": "https://mm.example.test"}))
    monkeypatch.setattr(adapter, "_ws_loop", AsyncMock())
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda _: None)
    try:
        assert await adapter.connect()
        assert (await adapter.send("channel", "hello")).success
        assert await adapter._upload_file("channel", b"data", "file.txt") == "file"
        assert (await adapter._download_attachments(["file"]))[0]
        await adapter._api("PUT", "posts/post", {"message": "edit"})
        await adapter._api("DELETE", "posts/post")
        await adapter._ws_connect_and_listen()
        expected = proxy if route == "explicit" else None
        for method in ("get", "post", "put", "delete", "ws_connect"):
            for call in getattr(session, method).call_args_list:
                assert call.kwargs.get("proxy") == expected, (method, call)
        assert factory.call_args.kwargs.get("connector") is connector
        if connector is not None:
            assert factory.call_args.kwargs["trust_env"] is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "network_error"])
async def test_proxy_auth_failure_keeps_existing_cleanup(monkeypatch, outcome):
    monkeypatch.setenv("MATTERMOST_PROXY", "http://127.0.0.1:18765")
    monkeypatch.setattr(base, "_aiohttp_socks_connector", lambda url: None)
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={})
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    if outcome == "network_error":
        context.__aenter__.side_effect = aiohttp.ClientConnectionError("fixture failure")
    session = MagicMock()
    session.get.return_value = context
    session.close = AsyncMock()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
    adapter = MattermostAdapter(PlatformConfig(token="fixture", extra={"url": "https://mm.example.test"}))
    assert await adapter.connect() is False
    session.close.assert_awaited_once()
    assert adapter._ws_task is None
