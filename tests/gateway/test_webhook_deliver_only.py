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
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes}
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

    @pytest.mark.asyncio
    async def test_template_rendering_works(self):
        """Dot-notation template variables resolve in deliver_only mode."""
        routes = {
            "alert": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "chat-1"},
                "prompt": "Build {build.number} status: {build.status}",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alert",
                json={"build": {"number": 77, "status": "FAILED"}},
                headers={"X-GitHub-Delivery": "d-render-1"},
            )
            assert resp.status == 200

        mock_target.send.assert_awaited_once()
        content_arg = mock_target.send.await_args.args[1]
        assert content_arg == "Build 77 status: FAILED"

    @pytest.mark.asyncio
    async def test_thread_id_passed_through(self):
        """deliver_extra.thread_id flows through to the target adapter."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1", "thread_id": "topic-42"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        mock_target = _wire_mock_target(adapter)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "d-thread-1"},
            )
            assert resp.status == 200

        assert mock_target.send.await_args.kwargs["metadata"] == {
            "thread_id": "topic-42"
        }


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

    @pytest.mark.asyncio
    async def test_delivery_exception_returns_502(self):
        """If adapter.send() raises, we return 502 (not 500)."""
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
        mock_target.send = AsyncMock(side_effect=RuntimeError("tg exploded"))

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "d-exc-1"},
            )
            assert resp.status == 502
            data = await resp.json()
            assert data["error"] == "Delivery failed"
            # Exception message must not leak
            assert "exploded" not in json.dumps(data)

    @pytest.mark.asyncio
    async def test_target_platform_not_connected_returns_502(self):
        """deliver_only to a platform the gateway doesn't have → 502."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "discord",  # not configured in mock runner
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        _wire_mock_target(adapter, platform_name="telegram")  # only TG wired

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "d-no-platform-1"},
            )
            assert resp.status == 502


# ===================================================================
# Startup validation
# ===================================================================

class TestDeliverOnlyStartupValidation:

    @pytest.mark.asyncio
    async def test_deliver_only_with_log_deliver_rejected(self):
        """deliver_only=true + deliver=log is nonsense — reject at connect()."""
        routes = {
            "bad": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "log",
                "deliver_only": True,
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="deliver_only=true but deliver is 'log'"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_deliver_only_with_missing_deliver_rejected(self):
        """deliver_only=true with no deliver field defaults to 'log' → reject."""
        routes = {
            "bad": {
                "secret": _INSECURE_NO_AUTH,
                # no deliver field
                "deliver_only": True,
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="deliver_only=true"):
            await adapter.connect()

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
    async def test_idempotency_still_applies(self):
        """Same delivery_id posted twice → second is suppressed."""
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

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            r1 = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "dup-1"},
            )
            assert r1.status == 200

            r2 = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "dup-1"},
            )
            # Existing webhook adapter treats duplicates as 200 + status=duplicate
            assert r2.status == 200
            data = await r2.json()
            assert data["status"] == "duplicate"

        # Target was called exactly once
        assert mock_target.send.await_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_still_applies(self):
        """Route-level rate limit caps deliver_only POSTs too."""
        routes = {
            "r": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": "telegram",
                "deliver_only": True,
                "deliver_extra": {"chat_id": "c-1"},
                "prompt": "hi",
            }
        }
        adapter = _make_adapter(routes, rate_limit=2)
        _wire_mock_target(adapter)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for i in range(2):
                r = await cli.post(
                    "/webhooks/r",
                    json={},
                    headers={"X-GitHub-Delivery": f"rl-{i}"},
                )
                assert r.status == 200

            # Third within the window → 429
            r3 = await cli.post(
                "/webhooks/r",
                json={},
                headers={"X-GitHub-Delivery": "rl-3"},
            )
            assert r3.status == 429


# ===================================================================
# Unit: _direct_deliver dispatch
# ===================================================================

class TestDirectDeliverUnit:

    @pytest.mark.asyncio
    async def test_dispatches_to_cross_platform_for_messaging_targets(self):
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        result = await adapter._direct_deliver(
            "hello",
            {"deliver": "telegram", "deliver_extra": {"chat_id": "c-1"}},
        )
        assert result.success is True
        mock_target.send.assert_awaited_once_with(
            "c-1", "hello", metadata=None
        )

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
