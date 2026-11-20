"""Test that HMAC signature validation happens BEFORE rate limiting.

This verifies the fix for bug #12544: invalid signature requests must NOT
consume rate-limit quota. Before the fix, rate limiting was applied before
signature validation, so an attacker could exhaust a victim's rate limit
with invalidly-signed requests and then make valid requests that get rejected
with 429.

The correct order is:
1. Read body
2. Validate HMAC signature (reject 404 if invalid - same as unknown
   route so a bad signature cannot enumerate configured route names)
3. Rate limit check (reject 429 if over limit)
4. Process the webhook
"""

import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.webhook import WebhookAdapter
from gateway.config import PlatformConfig


def _make_adapter(routes, rate_limit=5, **extra_kw) -> WebhookAdapter:
    """Create a WebhookAdapter with the given routes."""
    extra = {
        "host": "0.0.0.0",
        "port": 0,
        "routes": routes,
        "rate_limit": rate_limit,
    }
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter."""
    app = web.Application()
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    mount_authority(app, adapter)
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


SIMPLE_PAYLOAD = {"event": "test", "data": "hello"}


@pytest.mark.asyncio
async def test_malformed_body_spends_quota_and_overquota_does_not_parse(monkeypatch):
    adapter = _make_adapter({'fixture': {'secret': 'secret', 'prompt': '{text}'}}, rate_limit=1)
    body = b'\xff\xfe{\x00'
    headers = {'X-Hub-Signature-256': _github_signature(body, 'secret')}
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 400
        assert len(adapter._rate_counts.get('fixture', ())) == 1

        def forbidden(_body):
            pytest.fail('over-quota request was parsed')

        monkeypatch.setattr(adapter, '_parse_body', forbidden)
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 429


@pytest.mark.asyncio
async def test_parked_refusal_refunds_reserved_quota(monkeypatch):
    from aiohttp import web
    adapter = _make_adapter({'fixture': {'secret': 'secret', 'prompt': '{text}'}}, rate_limit=1)
    body = b'{"text":"hello"}'
    headers = {'X-Hub-Signature-256': _github_signature(body, 'secret')}
    monkeypatch.setattr('gateway.platforms.webhook_ingress.parked_profile_refusal',
                        lambda *args: web.json_response({'error': 'parked'}, status=503))
    async with TestClient(TestServer(_create_app(adapter))) as client:
        for _ in range(2):
            response = await client.post('/webhooks/fixture', data=body, headers=headers)
            assert response.status == 503
        assert not adapter._rate_counts.get('fixture', ())


class TestSignatureBeforeRateLimit:
    """Verify that invalid signatures do NOT consume rate limit quota."""

    @pytest.mark.asyncio
    async def test_invalid_signature_does_not_consume_rate_limit(self):
        """Send requests with invalid signatures up to the rate limit, then
        send a valid-signed request and verify it succeeds.

        BEFORE FIX: Invalid signatures consume the rate limit bucket, so
        after 'rate_limit' bad requests the valid one would get 429.
        AFTER FIX: Invalid signatures are rejected first (before
        rate limiting), so the rate limit bucket is untouched. The valid
        request after many bad ones still succeeds.
        """
        secret = "test-secret-key"
        route_name = "test-route"
        routes = {
            route_name: {
                "secret": secret,
                "events": ["push"],
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        rate_limit = 5
        adapter = _make_adapter(routes, rate_limit=rate_limit)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            # First exhaust the rate limit with invalid signatures
            for i in range(rate_limit):
                resp = await cli.post(
                    f"/webhooks/{route_name}",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": "sha256=invalid",  # bad sig
                        "X-GitHub-Delivery": f"bad-{i}",
                    },
                )
                # Each invalid signature is rejected with the same 404 an
                # unknown route gets (no route-existence leak).
                assert resp.status == 404, (
                    f"Expected 404 for invalid signature, got {resp.status}"
                )
                data_out = await resp.json()
                assert data_out["error"] == f"Unknown route: {route_name}"

            # Now send a valid-signed request — it MUST succeed (202)
            # BEFORE FIX: This would return 429 because the 5 bad requests
            # consumed the rate limit bucket.
            # AFTER FIX: Bad requests don't touch rate limiting, so valid
            # request succeeds.
            valid_sig = _github_signature(body, secret)
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": valid_sig,
                    "X-GitHub-Delivery": "good-001",
                },
            )
            assert resp.status == 202, (
                f"Expected 202 for valid request after invalid signatures, "
                f"got {resp.status}. Rate limit may have been consumed by "
                f"invalid requests (bug #12544 not fixed)."
            )

            data = await resp.json()
            assert data["status"] == "accepted"

        # The valid event should have been captured
        assert len(captured_events) == 1


