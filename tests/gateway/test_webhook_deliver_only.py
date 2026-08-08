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
from aiohttp.test_utils import TestClient, TestServer

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


# ===================================================================
# platform:chat_id inline format — _direct_deliver (deliver_only mode)
# ===================================================================

class TestDirectDeliverInlineChatId:
    """Regression tests for platform:chat_id format in deliver_only mode.

    Sweeper review (teknium1) found that ``_direct_deliver()`` bypassed
    the parser added to ``send()``, so ``deliver: "telegram:-100..."``
    still reached ``Platform(platform_name)`` and failed.
    """

    @pytest.mark.asyncio
    async def test_inline_chat_id_routed_correctly(self):
        """deliver_only with telegram:<chat_id> delivers to that chat."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        result = await adapter._direct_deliver(
            "alert!",
            {"deliver": "telegram:-1003774178835", "deliver_extra": {}},
        )
        assert result.success is True
        mock_target.send.assert_awaited_once()
        chat_id_arg = mock_target.send.await_args.args[0]
        assert chat_id_arg == "-1003774178835"

    @pytest.mark.asyncio
    async def test_inline_chat_id_takes_precedence_over_deliver_extra(self):
        """target_chat_id wins over deliver_extra.chat_id."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        result = await adapter._direct_deliver(
            "alert!",
            {
                "deliver": "telegram:-1003774178835",
                "deliver_extra": {"chat_id": "should-be-ignored"},
            },
        )
        assert result.success is True
        chat_id_arg = mock_target.send.await_args.args[0]
        assert chat_id_arg == "-1003774178835"

    @pytest.mark.asyncio
    async def test_bare_platform_falls_back_to_deliver_extra(self):
        """deliver_only with bare 'telegram' still uses deliver_extra.chat_id."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        result = await adapter._direct_deliver(
            "hello",
            {"deliver": "telegram", "deliver_extra": {"chat_id": "c-1"}},
        )
        assert result.success is True
        chat_id_arg = mock_target.send.await_args.args[0]
        assert chat_id_arg == "c-1"

    @pytest.mark.asyncio
    async def test_bare_platform_falls_back_to_home_channel(self):
        """deliver_only with bare 'telegram' and no deliver_extra uses home channel."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        # Configure mock home channel
        mock_home = MagicMock()
        mock_home.chat_id = "home-chat-99"
        adapter.gateway_runner.config.get_home_channel.return_value = mock_home

        result = await adapter._direct_deliver(
            "hello",
            {"deliver": "telegram", "deliver_extra": {}},
        )
        assert result.success is True
        chat_id_arg = mock_target.send.await_args.args[0]
        assert chat_id_arg == "home-chat-99"


# ===================================================================
# platform:chat_id inline format — send() (agent mode)
# ===================================================================

class TestSendInlineChatId:
    """Regression tests for platform:chat_id format in agent-mode send()."""

    @pytest.mark.asyncio
    async def test_inline_chat_id_routed_correctly(self):
        """Agent-mode send() with telegram:<chat_id> delivers to that chat."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        # Populate _delivery_info as the webhook handler would
        chat_id = "webhook:route:delivery-1"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram:-1003774178835",
            "deliver_extra": {},
        }

        result = await adapter.send(chat_id, "agent response")
        assert result.success is True
        mock_target.send.assert_awaited_once()
        target_chat_id = mock_target.send.await_args.args[0]
        assert target_chat_id == "-1003774178835"

    @pytest.mark.asyncio
    async def test_inline_chat_id_takes_precedence_over_deliver_extra(self):
        """In agent mode, target_chat_id wins over deliver_extra.chat_id."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        chat_id = "webhook:route:delivery-2"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram:-1003774178835",
            "deliver_extra": {"chat_id": "should-be-ignored"},
        }

        result = await adapter.send(chat_id, "agent response")
        assert result.success is True
        target_chat_id = mock_target.send.await_args.args[0]
        assert target_chat_id == "-1003774178835"

    @pytest.mark.asyncio
    async def test_bare_platform_falls_back_to_deliver_extra(self):
        """Agent-mode send() with bare 'telegram' uses deliver_extra.chat_id."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        chat_id = "webhook:route:delivery-3"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "c-extra"},
        }

        result = await adapter.send(chat_id, "agent response")
        assert result.success is True
        target_chat_id = mock_target.send.await_args.args[0]
        assert target_chat_id == "c-extra"

    @pytest.mark.asyncio
    async def test_bare_platform_falls_back_to_home_channel(self):
        """Agent-mode send() with bare 'telegram' and no extra uses home channel."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        mock_home = MagicMock()
        mock_home.chat_id = "home-chat-42"
        adapter.gateway_runner.config.get_home_channel.return_value = mock_home

        chat_id = "webhook:route:delivery-4"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram",
            "deliver_extra": {},
        }

        result = await adapter.send(chat_id, "agent response")
        assert result.success is True
        target_chat_id = mock_target.send.await_args.args[0]
        assert target_chat_id == "home-chat-42"


# ===================================================================
# _parse_deliver_target unit tests
# ===================================================================

class TestParseDeliverTarget:
    """Unit tests for the shared parser used by send() and _direct_deliver()."""

    def test_bare_platform(self):
        p, c = WebhookAdapter._parse_deliver_target("telegram")
        assert p == "telegram"
        assert c is None

    def test_platform_with_chat_id(self):
        p, c = WebhookAdapter._parse_deliver_target("telegram:-1003774178835")
        assert p == "telegram"
        assert c == "-1003774178835"

    def test_platform_uppercase_normalized(self):
        p, c = WebhookAdapter._parse_deliver_target("Telegram:-100123")
        assert p == "telegram"
        assert c == "-100123"

    def test_platform_with_empty_chat_id(self):
        """'telegram:' → platform='telegram', chat_id=None (falls back)."""
        p, c = WebhookAdapter._parse_deliver_target("telegram:")
        assert p == "telegram"
        assert c is None

    def test_discord_with_chat_id(self):
        p, c = WebhookAdapter._parse_deliver_target("discord:123456789")
        assert p == "discord"
        assert c == "123456789"
