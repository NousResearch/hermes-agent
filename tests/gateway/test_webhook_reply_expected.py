"""Webhook turns are machine traffic: reply_expected must be False.

A route prompt that answers ``[SILENT]`` on a quiet tick is the designed
contract for monitor-style subscriptions. Before this change the webhook
adapter left ``MessageEvent.reply_expected`` unset, the gateway's turn
shaping treated the webhook turn as a human turn, and a bare silence
marker was re-inflated into an unexpected-silence warning (and retry)
on lanes nobody is reading. This pins the fix at the event-creation site.
"""

import asyncio

from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.webhook import _INSECURE_NO_AUTH
from tests.gateway.test_webhook_adapter import (
    _create_app,
    _make_adapter,
)


def test_webhook_event_reply_expected_false():
    """A dispatched webhook POST creates a machine turn, not a human turn."""
    # No "events" key: the route accepts any delivery (event filtering is
    # opt-in). Loopback host so INSECURE_NO_AUTH is honored without a secret.
    routes = {
        "notify": {
            "secret": _INSECURE_NO_AUTH,
            "prompt": "Monitor tick: {payload.state}. Answer [SILENT] if nothing needs attention.",
        }
    }
    adapter = _make_adapter(routes=routes, host="127.0.0.1")
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    async def _run():
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/notify",
                json={"state": "quiet"},
                headers={"X-GitHub-Delivery": "reply-expected-1"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)

    asyncio.run(_run())

    assert len(captured) == 1
    assert captured[0].reply_expected is False
