"""Bearer authentication respects served-profile route bindings."""
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_webhook_bearer_contract import BODY, SECRET, make_app


@pytest.mark.asyncio
async def test_bearer_is_bound_to_named_profile(tmp_path, monkeypatch):
    routes = {"probe": {"secret": SECRET, "profile": "worker", "events": ["pull_request"]}}
    adapter, app = make_app(routes=routes)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex, *, include_parked=False: [
        ("default", tmp_path), ("worker", tmp_path / "worker"), ("other", tmp_path / "other"),
    ])
    async with TestClient(TestServer(app)) as client:
        adapter.gateway_runner.config.multiplex_profiles = True
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
