"""A webhook turn is an automation lane: silence must stand, not raise the fallback.

The turn layer's silence rule (gateway/run_turn.py::_hmwa_shape_agent_response) only lets a
bare silence marker stand on machinery turns or when ``MessageEvent.reply_expected is False``.
The webhook adapter's own delivery path (``WebhookAdapter.send``) suppresses bare silence
markers for every route, so the adapter must mark its admitted events ``reply_expected=False``
— otherwise the turn layer emits the visible "model returned only a silence marker" fallback
for the exact responses the delivery layer was about to swallow, and webhook lanes (AgentMail
intake, cron-triggered routes) spam that warning on every routine quiet turn.

Driven through the real HTTP handler so the admission wiring, not just the field default, is
covered (mirrors tests/gateway/test_slack_reply_expected.py).
"""
import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_webhook_adapter import (  # noqa: F401 - fixtures/helpers
    _create_app,
    _make_adapter,
    _INSECURE_NO_AUTH,
)


@pytest.mark.asyncio
async def test_admitted_webhook_event_carries_reply_expected_false():
    """Every route-admitted webhook message is unaddressed by construction."""
    routes = {"lane": {"secret": _INSECURE_NO_AUTH, "prompt": "process {data}"}}
    adapter = _make_adapter(routes=routes)
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/lane",
            json={"data": "value"},
            headers={"X-GitHub-Delivery": "silence-lane-1"},
        )
        assert resp.status == 202

    await asyncio.sleep(0.05)
    assert len(captured) == 1
    assert captured[0].reply_expected is False
