"""Durable webhook receipts: redelivery after a restart is a duplicate, and every outcome is on record."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.platforms.webhook_receipts import read_history, receipts_path
from hermes_constants import get_hermes_home

ROUTES = {
    "alerts": {"secret": _INSECURE_NO_AUTH, "deliver": "telegram", "deliver_only": True,
               "deliver_extra": {"chat_id": "fake-chat"}, "prompt": "alert {n}", "events": ["push"]},
}


def _gateway(sent: list, reject: bool = False) -> WebhookAdapter:
    """A fresh adapter process: same HERMES_HOME, empty memory, a recording Telegram target."""
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": ROUTES}))
    target = AsyncMock()

    async def send(chat_id, content, **_kw):
        sent.append(content)
        return SendResult(success=not reject, error="chat not found" if reject else None)

    target.send = send
    runner = MagicMock()
    runner.adapters = {Platform.TELEGRAM: target}
    runner._authorization_adapter = lambda platform, profile=None: runner.adapters.get(platform)
    runner.config.get_home_channel.return_value = None
    adapter.gateway_runner = runner
    return adapter


async def _post(adapter, delivery_id, event="push", n=1):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/webhooks/alerts", data=json.dumps({"n": n}),
                                     headers={"X-GitHub-Delivery": delivery_id, "X-GitHub-Event": event,
                                              "Content-Type": "application/json"})
        return response.status, (await response.json()).get("status")


def _history(**kw):
    return [(r["outcome"], r["delivery_id"], r["reason"]) for r in
            read_history(receipts_path(get_hermes_home()), **kw)]


@pytest.mark.asyncio
async def test_redelivery_after_restart_is_a_duplicate_and_a_rejection_can_be_retried():
    sent = []
    assert await _post(_gateway(sent), "d-1") == (200, "delivered")
    restarted = _gateway(sent)  # gateway restart: the in-memory cache is gone
    assert await _post(restarted, "d-1") == (200, "duplicate")
    assert sent == ["alert 1"]

    # Two copies of one delivery racing each other: exactly one send.
    results = await asyncio.gather(_post(restarted, "d-2"), _post(restarted, "d-2"))
    assert sorted(results) == [(200, "delivered"), (200, "duplicate")]
    assert sent == ["alert 1", "alert 1"]

    # A target that refused the message sent nothing, so the provider's retry goes through.
    assert await _post(_gateway(sent, reject=True), "d-3") == (502, "error")
    assert await _post(_gateway(sent), "d-3") == (200, "delivered")
    assert len(sent) == 4


@pytest.mark.asyncio
async def test_ignored_and_failed_deliveries_leave_receipts_with_the_reason():
    adapter = _gateway([])
    assert await _post(adapter, "skip-1", event="issues") == (200, "ignored")
    assert await _post(adapter, "ok-1") == (200, "delivered")
    assert await _post(adapter, "ok-1") == (200, "duplicate")
    history = _history()
    assert [h[:2] for h in history] == [("duplicate", "ok-1"), ("accepted", "ok-1"), ("ignored", "skip-1")]
    assert "issues" in history[2][2]
    assert _history(route="other") == []
    assert len(_history(limit=1)) == 1

    from hermes_cli.webhook import webhook_command
    from io import StringIO
    from contextlib import redirect_stdout
    out = StringIO()
    with redirect_stdout(out):
        webhook_command(SimpleNamespace(webhook_action="history", route="alerts", limit=5))
    text = out.getvalue()
    assert "ignored" in text and "skip-1" in text and "deliver:telegram" in text
