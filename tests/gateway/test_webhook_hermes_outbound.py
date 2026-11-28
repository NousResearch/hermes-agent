"""Hermes -> Hermes: a ``hooks.outbound`` delivery is accepted by Hermes's own inbound webhook route (#127471).

The delivery is built by the real ``agent.outbound_webhooks._build_delivery`` and POSTed unchanged.
"""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import outbound_webhooks
from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter

SECRET = "peer-secret"
BODY = b'{"hook_event_name":"kanban_task_blocked","delivery_id":"d1"}'


def _adapter(route: dict) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "secret": SECRET, "routes": {"r": {"prompt": "x", **route}}}))


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _delivery(secret, delivery_id: str = "d1") -> dict:
    target = outbound_webhooks.WebhookTarget(url="http://unused/webhooks/r", events=["kanban_task_blocked"],
                                             secret=secret)
    return outbound_webhooks._build_delivery("kanban_task_blocked", target, BODY, delivery_id)


async def _post_all(adapter: WebhookAdapter, *deliveries: dict) -> list:
    out = []
    async with TestClient(TestServer(_app(adapter))) as cli:
        for delivery in deliveries:
            resp = await cli.post("/webhooks/r", data=delivery["body"], headers=delivery["headers"])
            body = await resp.json() if resp.content_type == "application/json" else {}
            out.append((resp.status, body.get("status")))
    return out


@pytest.mark.parametrize("route", [{}, {"signature": "generic"}], ids=["unpinned", "pinned-generic"])
@pytest.mark.asyncio
async def test_outbound_delivery_accepted_by_inbound_route(route):
    [(status, _)] = await _post_all(_adapter(route), _delivery(SECRET))
    assert status == 202


@pytest.mark.asyncio
async def test_outbound_delivery_with_other_secret_rejected():
    [(status, _)] = await _post_all(_adapter({"signature": "generic"}), _delivery("not-the-route-secret"))
    assert status not in (200, 202)


@pytest.mark.asyncio
async def test_sender_retry_is_deduped_by_delivery_id():
    # The sender retries the SAME delivery dict (MAX_DELIVERY_ATTEMPTS); the receiver must not run it twice.
    delivery = _delivery(SECRET, "d-retry")
    first, second = await _post_all(_adapter({"signature": "generic"}), delivery, delivery)
    assert first[0] == 202
    assert second == (200, "duplicate")


def test_github_style_header_kept_for_existing_receivers():
    headers = _delivery(SECRET)["headers"]
    assert headers["X-Hermes-Signature-256"].startswith("sha256=")
    assert headers["X-Request-ID"] == headers["X-Hermes-Delivery"] == "d1"
    assert headers["X-Webhook-Timestamp"].isdigit()


def test_unsigned_target_sends_no_signature_headers():
    headers = _delivery(None)["headers"]
    assert not {"X-Hermes-Signature-256", "X-Webhook-Signature-V2", "X-Webhook-Timestamp"} & set(headers)
