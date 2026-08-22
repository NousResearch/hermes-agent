"""Deterministic tests for webhook → configured-platform delivery bridge.

Covers the Command Center / operator pattern:
  signed webhook POST → agent final response → exactly one Discord (etc.)
  delivery to the route's configured chat_id — not the webhook session key,
  not origin/DM, and not a silent log-only fallback.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


DISCORD_CHANNEL = "1532416269628739694"


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _wire_discord(adapter: WebhookAdapter) -> AsyncMock:
    mock_discord = AsyncMock()
    mock_discord.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    mock_runner = MagicMock()
    mock_runner.adapters = {Platform.DISCORD: mock_discord}
    mock_runner.config.get_home_channel.return_value = MagicMock(chat_id="HOME_DM_OR_CHANNEL")
    mock_runner._profile_adapters = {}
    adapter.gateway_runner = mock_runner
    return mock_discord


@pytest.mark.asyncio
async def test_stored_route_preserves_discord_channel_target():
    """Subscription deliver_extra.chat_id is stored on the per-delivery session."""
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "operator: {prompt}",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/cc-operator",
            data=json.dumps({"prompt": "ping", "chat_id": "SHOULD_NOT_OVERRIDE"}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Delivery": "deliv-store-1",
            },
        )
        assert resp.status == 202

    chat_id = "webhook:cc-operator:deliv-store-1"
    info = adapter._delivery_info[chat_id]
    assert info["deliver"] == "discord"
    assert info["explicit_chat_id"] is True
    assert info["deliver_extra"]["chat_id"] == DISCORD_CHANNEL
    # Payload must not rewrite the configured channel target.
    assert info["deliver_extra"]["chat_id"] != "SHOULD_NOT_OVERRIDE"
    mock_discord.send.assert_not_awaited()  # agent path — only stored so far


@pytest.mark.asyncio
async def test_final_response_dispatches_to_configured_discord_adapter():
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/cc-operator",
            json={"prompt": "x"},
            headers={"X-GitHub-Delivery": "deliv-final-1"},
        )

    result = await adapter.send(
        "webhook:cc-operator:deliv-final-1",
        "Parked — concise operator outcome.",
    )
    assert result.success is True
    mock_discord.send.assert_awaited_once_with(
        DISCORD_CHANNEL,
        "Parked — concise operator outcome.",
        metadata=None,
    )
    # Must not use home/DM fallback when explicit chat_id is configured.
    home = adapter.gateway_runner.config.get_home_channel.return_value.chat_id
    assert mock_discord.send.await_args.args[0] != home


@pytest.mark.asyncio
async def test_exactly_once_identical_final_content():
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/cc-operator",
            json={},
            headers={"X-GitHub-Delivery": "deliv-once-1"},
        )

    body = "exactly-once body"
    chat = "webhook:cc-operator:deliv-once-1"
    r1 = await adapter.send(chat, body)
    r2 = await adapter.send(chat, body)
    assert r1.success and r2.success
    assert mock_discord.send.await_count == 1


@pytest.mark.asyncio
async def test_missing_target_adapter_does_not_claim_delivered():
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    # Gateway runner present but Discord not connected.
    mock_runner = MagicMock()
    mock_runner.adapters = {}
    mock_runner._profile_adapters = {}
    mock_runner.config.get_home_channel.return_value = None
    adapter.gateway_runner = mock_runner
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/cc-operator",
            json={},
            headers={"X-GitHub-Delivery": "deliv-miss-1"},
        )

    result = await adapter.send("webhook:cc-operator:deliv-miss-1", "outcome")
    assert result.success is False
    assert "not connected" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_failing_adapter_does_not_claim_delivered():
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    mock_discord.send = AsyncMock(
        return_value=SendResult(success=False, error="boom")
    )
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/cc-operator",
            json={},
            headers={"X-GitHub-Delivery": "deliv-fail-1"},
        )

    result = await adapter.send("webhook:cc-operator:deliv-fail-1", "outcome")
    assert result.success is False
    assert result.error == "boom"
    # Failure clears the exactly-once claim so a later retry can proceed.
    mock_discord.send = AsyncMock(return_value=SendResult(success=True, message_id="m2"))
    retry = await adapter.send("webhook:cc-operator:deliv-fail-1", "outcome")
    assert retry.success is True
    mock_discord.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_normal_log_route_preserves_webhook_response_behavior():
    routes = {
        "plain": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "hello {x}",
            "deliver": "log",
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/plain",
            json={"x": "world"},
            headers={"X-GitHub-Delivery": "deliv-log-1"},
        )

    result = await adapter.send("webhook:plain:deliv-log-1", "agent said this")
    assert result.success is True
    mock_discord.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_delivery_info_with_deliver_route_refuses_log_fallback():
    """If per-delivery state was lost but the route still targets Discord, fail."""
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    # Simulate lost delivery_info (TTL race / restart mid-flight).
    adapter._delivery_info.clear()

    result = await adapter.send(
        "webhook:cc-operator:ghost-delivery",
        "should not silently log",
    )
    assert result.success is False
    assert "delivery_info missing" in (result.error or "")
    mock_discord.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_payload_cannot_override_configured_chat_id():
    routes = {
        "cc-operator": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "{prompt}",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    adapter = _make_adapter(routes)
    # Hostile payload tries to supply alternate destinations.
    rendered = adapter._render_delivery_extra(
        {"chat_id": DISCORD_CHANNEL},
        {"chat_id": "999999999999999999", "prompt": "x"},
    )
    assert rendered["chat_id"] == DISCORD_CHANNEL


@pytest.mark.asyncio
async def test_explicit_chat_id_blank_refuses_home_fallback():
    adapter = _make_adapter({})
    mock_discord = _wire_discord(adapter)
    delivery = {
        "deliver": "discord",
        "route": "cc-operator",
        "explicit_chat_id": True,
        "deliver_extra": {"chat_id": ""},
    }
    result = await adapter._deliver_cross_platform("discord", "hi", delivery)
    assert result.success is False
    assert "explicit chat_id" in (result.error or "").lower()
    mock_discord.send.assert_not_awaited()
    adapter.gateway_runner.config.get_home_channel.assert_not_called()


@pytest.mark.asyncio
async def test_no_token_leakage_in_delivery_logs(caplog):
    routes = {
        "cc-operator": {
            "secret": "super-secret-hmac-value",
            "prompt": "x",
            "deliver": "discord",
            "deliver_extra": {"chat_id": DISCORD_CHANNEL},
        }
    }
    # Use INSECURE for the HTTP path; keep a secret string in route for log checks.
    routes["cc-operator"]["secret"] = _INSECURE_NO_AUTH
    adapter = _make_adapter(routes)
    mock_discord = _wire_discord(adapter)
    adapter.handle_message = AsyncMock()

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        await cli.post(
            "/webhooks/cc-operator",
            json={"token": "LEAKME_TOKEN_VALUE_SHOULD_NOT_APPEAR"},
            headers={"X-GitHub-Delivery": "deliv-noleak-1"},
        )

    with caplog.at_level("INFO"):
        await adapter.send(
            "webhook:cc-operator:deliv-noleak-1",
            "safe outcome text",
        )

    joined = "\n".join(r.message for r in caplog.records)
    assert "LEAKME_TOKEN_VALUE_SHOULD_NOT_APPEAR" not in joined
    assert "super-secret-hmac-value" not in joined
    assert DISCORD_CHANNEL in joined
    assert "Delivering response" in joined or "Delivered response" in joined
    mock_discord.send.assert_awaited_once()
