"""Managed Webex callbacks recover their signing key and active status."""

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from plugins.platforms.webex.adapter import WebexAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "inactive"])
async def test_existing_webhooks_use_current_secret_and_are_active(monkeypatch, status):
    adapter = WebexAdapter(PlatformConfig(token="token", extra={
        "connection_mode": "webhook", "public_url": "https://bot.example",
        "secret": "new-secret",
    }))
    remote = {key: {
        "id": key, "name": f"{adapter._webhook_prefix}-{key}",
        "resource": "messages", "event": "created", "filter": filter_value,
        "targetUrl": adapter._target_url, "status": status, "secret": "old-secret",
    } for key, filter_value in (("direct", "roomType=direct"), ("mentions", "mentionedPeople=me"))}

    async def listing(request):
        return web.json_response({"items": [
            {key: value for key, value in item.items() if key != "secret"}
            for item in remote.values()
        ]})

    async def update(request):
        remote[request.match_info["id"]].update(await request.json())
        return web.json_response(remote[request.match_info["id"]])

    app = web.Application()
    app.router.add_get("/webhooks", listing)
    app.router.add_put("/webhooks/{id}", update)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter._session = session
        await adapter._ensure_managed_webhooks()
        for item in remote.values():
            assert item["status"] == "active"
            body = b'{"resource":"messages"}'
            signature = hmac.new(item["secret"].encode(), body, hashlib.sha1).hexdigest()
            assert adapter._verify_signature({"X-Spark-Signature": signature}, body)


@pytest.mark.asyncio
async def test_failed_existing_webhook_update_is_not_ready():
    adapter = WebexAdapter(PlatformConfig(token="token", extra={
        "connection_mode": "webhook", "public_url": "https://bot.example", "secret": "secret",
    }))
    adapter._api_get_json = AsyncMock(return_value={"items": [{
        "id": "existing", "name": adapter._webhook_prefix + "-direct",
        "resource": "messages", "event": "created", "filter": "roomType=direct",
        "targetUrl": adapter._target_url, "status": "inactive",
    }]})
    adapter._api_put_json = AsyncMock(return_value={})
    adapter._api_post_json = AsyncMock(return_value={"id": "new"})
    with pytest.raises(RuntimeError, match="update managed Webex webhook"):
        await adapter._ensure_managed_webhooks()


@pytest.mark.asyncio
async def test_shared_listener_rejects_callback_without_profile_prefix(monkeypatch):
    adapter = WebexAdapter(PlatformConfig(token="token", extra={
        "connection_mode": "webhook", "public_url": "https://bot.example", "secret": "secret",
    }))
    adapter._shared_listener_profile = "work"
    adapter._load_bot_identity = AsyncMock()
    adapter._start_webhook_listener = AsyncMock()
    adapter._ensure_managed_webhooks = AsyncMock()
    monkeypatch.setattr(adapter, "_acquire_platform_lock", MagicMock(return_value=True))
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    try:
        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "webex_invalid_config"
        adapter._ensure_managed_webhooks.assert_not_awaited()
    finally:
        await adapter.disconnect()
