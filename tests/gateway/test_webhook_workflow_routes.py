"""Workflow hook routes on the webhook adapter: only a hook the workflow store minted may take an
unsigned POST (its URL is the credential); a route merely *named* ``wf-*`` keeps its HMAC."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _client(routes: dict) -> tuple[WebhookAdapter, TestClient]:
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes}))
    adapter.handle_message = AsyncMock()
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return adapter, TestClient(TestServer(app))


@pytest.mark.asyncio
async def test_a_static_route_named_like_a_workflow_hook_keeps_its_hmac():
    routes = {
        "wf-release": {"secret": "s3cret", "prompt": "ship {x}"},
        # A forged marker on a name the store never derived is still not a workflow hook.
        "wf-forged": {"secret": "s3cret", "prompt": "x", "workflow": "nope", "hermes_workflow": True},
    }
    adapter, client = _client(routes)
    async with client as cli:
        for route in routes:
            resp = await cli.post(f"/webhooks/{route}", json={"x": 1})
            assert resp.status == 401, route
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_a_minted_workflow_hook_publishes_its_start_on_an_unsigned_post():
    from workflow import events, triggers

    hook = triggers.hook_info("ship")
    adapter, client = _client({hook["route"]: {
        "secret": hook["secret"], "workflow": "ship", "trigger": "go", "hermes_workflow": True,
    }})
    async with client as cli:
        resp = await cli.post(f"/webhooks/{hook['route']}", json={"x": 1})
        assert resp.status == 200
        run_id = (await resp.json())["run_id"]
        # A sender that signs is still verified: a wrong signature is refused.
        bad = await cli.post(f"/webhooks/{hook['route']}", json={"x": 2}, headers={"X-Webhook-Signature": "nope"})
        assert bad.status == 401
    (start,) = events.drain()
    assert start["name"] == events.START
    assert start["payload"] == {"workflowId": "ship", "runId": run_id, "payload": {"x": 1}, "source": "webhook", "trigger": "go"}
    adapter.handle_message.assert_not_called()
