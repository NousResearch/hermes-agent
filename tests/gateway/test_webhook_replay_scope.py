"""Webhook replay/credential invariants that the durable ledger does not cover (PR #106742 security lane)."""
import hashlib
import hmac

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _adapter(routes):
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes}))


def _signed(secret, body, delivery):
    return {"X-Hub-Signature-256": "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(),
            "X-GitHub-Delivery": delivery, "X-GitHub-Event": "issues", "Content-Type": "application/json"}


@pytest.mark.asyncio
async def test_cron_route_replay_fires_the_job_once():
    """cron_job routes never reach the durable ledger, so a replayed signed delivery must not re-fire."""
    a = _adapter({"nightly": {"secret": "s", "prompt": "x", "cron_job": "job1"}})
    fired = []
    a._handle_cron_trigger = lambda *args, **kw: (fired.append(args[4]),
                                                 web.json_response({"status": "accepted"}, status=202))[1]
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", a._handle_webhook)
    body = b'{"k":1}'
    async with TestClient(TestServer(app)) as c:
        statuses = [(await c.post("/webhooks/nightly", data=body, headers=_signed("s", body, "CAP"))).status
                    for _ in range(3)]
    assert fired == ["CAP"], (statuses, fired)


def test_adding_a_dynamic_route_keeps_the_adapter_credential():
    """An unrelated hot-reloaded subscription must not re-key the connector of admitted deliveries."""
    from gateway.run import GatewayRunner
    a = _adapter({"victim": {"secret": "v", "prompt": "x"}})
    before = GatewayRunner._adapter_credential_fingerprint(a)
    a._dynamic_routes = {"agent-made": {"secret": "n", "prompt": "y"}}
    a._routes = {**a._dynamic_routes, **a._static_routes}
    assert GatewayRunner._adapter_credential_fingerprint(a) == before
