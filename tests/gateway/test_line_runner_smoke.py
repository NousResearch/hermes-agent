"""Smoke tests for LineAdapter connect/disconnect lifecycle and route registration."""
import logging

import pytest
from aiohttp import web

from gateway.platforms.line import LineAdapter
from tests.gateway.conftest import make_line_platform_config


@pytest.mark.asyncio
async def test_connect_starts_standalone_server(monkeypatch, line_lock_noop):
    """connect() spins up its own aiohttp listener and tracks the runner."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_WEBHOOK_PORT", "0")  # let OS pick free port
    adapter = LineAdapter(make_line_platform_config(token="t"))
    try:
        result = await adapter.connect()
        assert result is True
        assert adapter._runner is not None
        assert adapter.is_connected is True
    finally:
        await adapter.disconnect()
        assert adapter.is_connected is False


def test_register_routes_exposes_webhook_and_health_endpoints(monkeypatch):
    """register_routes() injects the LINE webhook + health endpoints
    on a caller-owned aiohttp app — used by tests and reusable for
    future shared-app integration."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    adapter = LineAdapter(make_line_platform_config(token="t"))
    app = web.Application()
    adapter.register_routes(app)
    routes = [r.resource.canonical for r in app.router.routes()]
    assert "/line/webhook" in routes
    assert "/line/webhook/health" in routes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, allow_env, free_env, allow_id, free_id",
    [
        ("groups", "LINE_ALLOWED_GROUPS", "LINE_FREE_RESPONSE_GROUPS", "Callowed", "Cunreachable"),
        ("rooms", "LINE_ALLOWED_ROOMS", "LINE_FREE_RESPONSE_ROOMS", "Rallowed", "Runreachable"),
    ],
    ids=["groups", "rooms"],
)
async def test_connect_warns_unreachable_free_response(
    kind, allow_env, free_env, allow_id, free_id, caplog, monkeypatch, line_lock_noop
):
    """Operator footgun: free_response_<kind> contains an ID missing from
    allowed_<kind> → allowlist drops the message before free-response fires.
    connect() must log a WARNING so operators catch the misconfiguration."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv(allow_env, allow_id)
    monkeypatch.setenv(free_env, f"{free_id},{allow_id}")
    monkeypatch.setenv("LINE_WEBHOOK_PORT", "0")
    adapter = LineAdapter(make_line_platform_config(token="t"))
    try:
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.line"):
            await adapter.connect()
    finally:
        await adapter.disconnect()
    expected_setting = f"free_response_{kind}"
    assert any(
        free_id in r.message and expected_setting in r.message
        for r in caplog.records
    )
