"""Tests for the webhook adapter's ``deliver_only`` route mode.

``deliver_only`` lets external services (Supabase webhooks, monitoring
alerts, background jobs, other agents) push plain-text notifications to
a user's chat via the webhook adapter WITHOUT invoking the agent.  The
rendered prompt template becomes the literal message body.

Covers:
- Agent is NOT invoked (``handle_message`` never called)
- Rendered content is delivered to the target platform adapter
- HTTP returns 200 OK on success, 502 on delivery failure
- Startup validation rejects ``deliver_only`` without a real delivery target
- HMAC auth, rate limiting, and idempotency still apply
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _wire_mock_target(adapter: WebhookAdapter, platform_name: str = "telegram"):
    """Attach a gateway_runner with a mocked target adapter."""
    mock_target = AsyncMock()
    mock_target.send = AsyncMock(return_value=SendResult(success=True))

    mock_runner = MagicMock()
    mock_runner.adapters = {Platform(platform_name): mock_target}
    mock_runner.config.get_home_channel.return_value = None

    adapter.gateway_runner = mock_runner
    return mock_target


# ===================================================================
# Core behaviour: agent bypass
# ===================================================================

class TestDeliverOnlyBypassesAgent:
    """The whole point of the feature — handle_message must not be called."""

    @pytest.mark.asyncio
    async def test_post_delivers_directly_without_agent(self):
        routes = {
            "match-alert": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "12345"},
                "prompt": "{payload.user} matched with {payload.other}!",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)

        # Guard: handle_message must NOT be called in deliver_only mode
        handle_message_calls: list[MessageEvent] = []

        async def _capture(event):
            handle_message_calls.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        body = json.dumps(
            {"payload": {"user": "alice", "other": "bob"}}
        ).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/match-alert",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Delivery": "delivery-1",
                },
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "delivered"
            assert data["route"] == "match-alert"
            assert data["target"] == "telegram"

        # Let any background tasks settle before asserting no agent call
        await asyncio.sleep(0.05)

        # Agent was NOT invoked
        assert handle_message_calls == []

        # Target adapter.send() WAS called with the rendered template
        mock_target.send.assert_awaited_once()
        call_args = mock_target.send.await_args
        chat_id_arg, content_arg = call_args.args[0], call_args.args[1]
        assert chat_id_arg == "12345"
        assert content_arg == "alice matched with bob!"


# ===================================================================
# HTTP status codes
# ===================================================================

class TestDeliverOnlyStatusCodes:

    @pytest.mark.asyncio
    async def test_delivery_failure_returns_502(self):
        """If the target adapter returns SendResult(success=False), 502."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        mock_target.send = AsyncMock(
            return_value=SendResult(success=False, error="rate limited by tg")
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "d-fail-1"},
            )
            assert resp.status == 502
            data = await resp.json()
            # Generic error — no adapter-level detail leaks
            assert data["error"] == "Delivery failed"
            assert "rate limited" not in json.dumps(data)


# ===================================================================
# Startup validation
# ===================================================================

class TestDeliverOnlyStartupValidation:


    @pytest.mark.asyncio
    async def test_deliver_only_with_real_target_accepted(self):
        """Sanity check — a valid deliver_only config passes validation."""
        routes = {
            "good": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        # connect() does more than validation (binds a socket) — we just
        # want to verify the validation doesn't raise.  Call it and tear
        # down immediately.
        try:
            started = await adapter.connect()
            if started:
                await adapter.disconnect()
        except ValueError:
            pytest.fail("valid deliver_only config should not raise ValueError")


# ===================================================================
# Security + reliability invariants still hold
# ===================================================================

class TestDeliverOnlySecurityInvariants:

    @pytest.mark.asyncio
    async def test_hmac_still_enforced(self):
        """deliver_only does NOT bypass HMAC validation."""
        secret = "real-secret-123"
        routes = {
            "r": {
                "secret": secret,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # No signature header → reject
            resp = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "d-noauth-1"},
            )
            assert resp.status == 401

        # Target never called
        mock_target.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_delivery_can_retry_with_same_delivery_id(self):
        """A failed direct delivery releases its idempotency claim."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        mock_target.send = AsyncMock(
            side_effect=[
                SendResult(success=False, error="temporary outage"),
                SendResult(success=True),
            ]
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "retry-1"}
            first = await cli.post("/webhooks/r", json={}, headers=headers)
            second = await cli.post("/webhooks/r", json={}, headers=headers)

            assert first.status == 502
            assert second.status == 200
            assert (await second.json())["status"] == "delivered"

        assert mock_target.send.await_count == 2

    @pytest.mark.asyncio
    async def test_delivery_exception_can_retry_with_same_delivery_id(self):
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        mock_target.send = AsyncMock(
            side_effect=[RuntimeError("temporary outage"), SendResult(success=True)]
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "retry-exception-1"}
            first = await cli.post("/webhooks/r", json={}, headers=headers)
            second = await cli.post("/webhooks/r", json={}, headers=headers)

            assert first.status == 502
            assert second.status == 200
            assert (await second.json())["status"] == "delivered"

        assert mock_target.send.await_count == 2

    @pytest.mark.asyncio
    async def test_cancelled_delivery_releases_claim_for_retry(self):
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        mock_target.send = AsyncMock(
            side_effect=[asyncio.CancelledError(), SendResult(success=True)]
        )
        request = make_mocked_request(
            "POST",
            "/webhooks/r",
            headers={"X-GitHub-Delivery": "retry-cancel-1"},
            match_info={"route_name": "r"},
        )
        request.read = AsyncMock(return_value=b"{}")

        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_webhook(request)

        retry = await adapter._handle_webhook(request)
        assert retry.status == 200
        assert json.loads(retry.text)["status"] == "delivered"
        assert mock_target.send.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_delivery_id_sends_once(self):
        """The in-flight claim still suppresses concurrent duplicates."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _send(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return SendResult(success=True)

        mock_target.send = AsyncMock(side_effect=_send)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "same-direct-id"}
            first = asyncio.create_task(cli.post("/webhooks/r", json={}, headers=headers))
            await asyncio.wait_for(entered.wait(), timeout=1)

            # Even a send that outlives the configured TTL remains reserved
            # until completion; an expired timestamp must not admit a second.
            adapter._idempotency_ttl = 0
            second = await cli.post("/webhooks/r", json={}, headers=headers)
            assert second.status == 200
            assert (await second.json())["status"] == "duplicate"

            adapter._idempotency_ttl = 3600
            release.set()
            first_response = await first
            assert first_response.status == 200
            assert (await first_response.json())["status"] == "delivered"

            retry = await cli.post("/webhooks/r", json={}, headers=headers)
            assert retry.status == 200
            assert (await retry.json())["status"] == "duplicate"

        assert mock_target.send.await_count == 1


# ===================================================================
# Unit: _direct_deliver dispatch
# ===================================================================

class TestDirectDeliverUnit:


    @pytest.mark.asyncio
    async def test_dispatches_to_github_comment(self):
        adapter = _make_adapter({})
        with patch.object(
            adapter, "_deliver_github_comment",
            new=AsyncMock(return_value=SendResult(success=True)),
        ) as mock_gh:
            result = await adapter._direct_deliver(
                "review body",
                {
                    "deliver": "github_comment",
                    "deliver_extra": {"repo": "org/r", "pr_number": "1"},
                },
            )
            assert result.success is True
            mock_gh.assert_awaited_once()
