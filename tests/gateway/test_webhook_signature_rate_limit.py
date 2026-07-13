"""Test that HMAC signature validation happens BEFORE rate limiting.

This verifies the fix for bug #12544: invalid signature requests must NOT
consume rate-limit quota. Before the fix, rate limiting was applied before
signature validation, so an attacker could exhaust a victim's rate limit
with invalidly-signed requests and then make valid requests that get rejected
with 429.

The correct order is:
1. Read body
2. Validate HMAC signature (reject 401 if invalid)
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
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


SIMPLE_PAYLOAD = {"event": "test", "data": "hello"}


class TestSignatureBeforeRateLimit:
    """Verify that invalid signatures do NOT consume rate limit quota."""

    @pytest.mark.asyncio
    async def test_invalid_signature_does_not_consume_rate_limit(self):
        """Send requests with invalid signatures up to the rate limit, then
        send a valid-signed request and verify it succeeds.

        BEFORE FIX: Invalid signatures consume the rate limit bucket, so
        after 'rate_limit' bad requests the valid one would get 429.
        AFTER FIX: Invalid signatures are rejected with 401 first (before
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
                # Each invalid signature should be rejected with 401
                assert resp.status == 401, (
                    f"Expected 401 for invalid signature, got {resp.status}"
                )

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

    @pytest.mark.asyncio
    async def test_valid_signature_still_rate_limited(self):
        """Verify that VALID requests still respect rate limiting normally."""
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
        rate_limit = 3
        adapter = _make_adapter(routes, rate_limit=rate_limit)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            # Send 'rate_limit' valid requests — all should succeed
            for i in range(rate_limit):
                valid_sig = _github_signature(body, secret)
                resp = await cli.post(
                    f"/webhooks/{route_name}",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": valid_sig,
                        "X-GitHub-Delivery": f"good-{i}",
                    },
                )
                assert resp.status == 202

            # The next valid request SHOULD be rate-limited
            valid_sig = _github_signature(body, secret)
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": valid_sig,
                    "X-GitHub-Delivery": "good-over-limit",
                },
            )
            assert resp.status == 429, (
                f"Expected 429 when exceeding rate limit with valid requests, "
                f"got {resp.status}"
            )

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_signatures(self):
        """Interleave invalid and valid requests. Only valid ones count
        against the rate limit."""
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
        rate_limit = 3
        adapter = _make_adapter(routes, rate_limit=rate_limit)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            # Send 2 valid requests (should succeed)
            for i in range(2):
                valid_sig = _github_signature(body, secret)
                resp = await cli.post(
                    f"/webhooks/{route_name}",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": valid_sig,
                        "X-GitHub-Delivery": f"good-{i}",
                    },
                )
                assert resp.status == 202

            # Send 10 invalid requests (should all get 401, not consume quota)
            for i in range(10):
                resp = await cli.post(
                    f"/webhooks/{route_name}",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": "sha256=invalid",
                        "X-GitHub-Delivery": f"bad-{i}",
                    },
                )
                assert resp.status == 401

            # One more valid request should STILL succeed (only 2 consumed)
            valid_sig = _github_signature(body, secret)
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": valid_sig,
                    "X-GitHub-Delivery": "good-3",
                },
            )
            assert resp.status == 202, (
                f"Expected 202 for 3rd valid request after many invalid ones, "
                f"got {resp.status}"
            )

            # The 4th valid request should be rate-limited (2 + 2 = 4 = limit)
            valid_sig = _github_signature(body, secret)
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": valid_sig,
                    "X-GitHub-Delivery": "good-4",
                },
            )
            assert resp.status == 429

        assert len(captured_events) == 3


class TestBearerTokenAuth:
    """Static bearer-token auth for producers that cannot compute HMAC."""

    @pytest.mark.parametrize(
        "auth_headers",
        [
            pytest.param({"Authorization": "Bearer TOKEN"}, id="authorization-bearer"),
            pytest.param({"X-Webhook-Token": "TOKEN"}, id="x-webhook-token"),
        ],
    )
    @pytest.mark.asyncio
    async def test_static_token_auth_accepts_valid_token_from_trusted_source(
        self, auth_headers
    ):
        secret = "test-bearer-token"
        route_name = "pbs-alerts"
        routes = {
            route_name: {
                "secret": secret,
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        body = json.dumps(SIMPLE_PAYLOAD).encode()
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": "static-token-good-001",
        }
        headers.update({k: v.replace("TOKEN", secret) for k, v in auth_headers.items()})

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers=headers,
            )
            assert resp.status == 202

        assert len(captured_events) == 1

    @pytest.mark.parametrize(
        "auth_headers",
        [
            pytest.param({"Authorization": "Bearer TOKEN"}, id="authorization-bearer"),
            pytest.param({"X-Webhook-Token": "TOKEN"}, id="x-webhook-token"),
        ],
    )
    @pytest.mark.asyncio
    async def test_static_token_auth_rejects_public_forwarded_source(
        self, auth_headers
    ):
        secret = "test-bearer-token"
        route_name = "pbs-alerts"
        routes = {
            route_name: {
                "secret": secret,
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        body = json.dumps(SIMPLE_PAYLOAD).encode()
        headers = {
            "Content-Type": "application/json",
            "X-Forwarded-For": "8.8.8.8",
            "X-Request-ID": "static-token-public-001",
        }
        headers.update({k: v.replace("TOKEN", secret) for k, v in auth_headers.items()})

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers=headers,
            )
            assert resp.status == 401

        assert captured_events == []

    @pytest.mark.asyncio
    async def test_hmac_signature_accepts_public_forwarded_source(self):
        secret = "test-hmac-secret"
        route_name = "github-alerts"
        routes = {
            route_name: {
                "secret": secret,
                "events": ["push"],
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": _github_signature(body, secret),
                    "X-Forwarded-For": "8.8.8.8",
                    "X-GitHub-Delivery": "hmac-public-001",
                },
            )
            assert resp.status == 202

        assert len(captured_events) == 1

    @pytest.mark.asyncio
    async def test_query_token_auth_is_rejected(self):
        secret = "test-query-token"
        route_name = "pbs-alerts"
        routes = {
            route_name: {
                "secret": secret,
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/webhooks/{route_name}?token={secret}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": "query-token-rejected-001",
                },
            )
            assert resp.status == 401

        assert captured_events == []

    @pytest.mark.asyncio
    async def test_bearer_token_auth_rejects_wrong_token(self):
        secret = "test-bearer-token"
        route_name = "pbs-alerts"
        routes = {
            route_name: {
                "secret": secret,
                "prompt": "Event: {event}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = _create_app(adapter)
        body = json.dumps(SIMPLE_PAYLOAD).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/webhooks/{route_name}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer wrong-token",
                    "X-Request-ID": "bearer-bad-001",
                },
            )
            assert resp.status == 401
