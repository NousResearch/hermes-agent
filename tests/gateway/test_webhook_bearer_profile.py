"""Bearer authentication respects served-profile route bindings."""
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_webhook_bearer_contract import BODY, SECRET, make_app


@pytest.mark.asyncio
async def test_bearer_is_bound_to_named_profile(tmp_path, monkeypatch):
    routes = {"probe": {"secret": SECRET, "profile": "worker", "events": ["pull_request"]}}
    adapter, app = make_app(routes=routes)
    async with TestClient(TestServer(app)) as client:
        # Use the established served-profile fixture after default authority startup;
        # profiles_to_serve is also called by unrelated multiplexer boot preflight.
        from tests.gateway.test_webhook_adapter import TestMultiplexProfileWebhookAuthentication
        TestMultiplexProfileWebhookAuthentication._configure_profiles(adapter, tmp_path, monkeypatch)
        adapter.handle_message = AsyncMock()
        headers = {"Authorization": "Bearer " + SECRET, "X-GitHub-Event": "push"}
        for path in ("/webhooks/probe", "/p/other/webhooks/probe"):
            response = await client.post(path, data=BODY, headers=headers)
            assert response.status == 404
        adapter.handle_message.assert_not_awaited()
        response = await client.post("/p/worker/webhooks/probe", data=BODY, headers=headers)
        assert response.status == 200
        assert (await response.json())["status"] == "ignored"
        adapter.handle_message.assert_not_awaited()
