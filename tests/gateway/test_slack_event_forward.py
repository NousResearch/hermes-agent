"""
Tests for config-driven Slack Events API / interactivity HTTP forwarding.

Socket Mode delivers channel message events and Block Kit actions over the
gateway's WebSocket, so external local services implementing Slack's HTTP
callbacks (e.g. a profile's order/approval server) never see them. The
``slack_event_forward`` module lets a profile declare ``slack.event_forwards``
(channel → URL) and ``slack.action_forwards`` (action_id prefix → URL) in
config.yaml so the gateway relays re-signed copies in Slack's native shapes.
"""

import hashlib
import hmac
import json
import time
import urllib.parse

import pytest

from gateway.platforms.slack_event_forward import (
    build_signed_action_request,
    build_signed_event_request,
    forward_block_actions,
    forward_event,
    parse_action_forwards,
    parse_event_forwards,
)


def _verify_slack_signature(signing_secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    """Mirror of a standard Slack v0 verifier (e.g. WPC approval server)."""
    base = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# parse_event_forwards / parse_action_forwards
# ---------------------------------------------------------------------------

def test_parse_event_forwards_keeps_channel_mapping():
    raw = {"C0123456789": "http://127.0.0.1:8787/slack/events"}
    assert parse_event_forwards(raw) == {"C0123456789": "http://127.0.0.1:8787/slack/events"}


def test_parse_forwards_reject_non_dict_and_empty_entries():
    assert parse_event_forwards(None) == {}
    assert parse_event_forwards("C012") == {}
    assert parse_action_forwards(["wpc_"]) == {}
    assert parse_action_forwards({"": "http://x", "wpc_": "", "ok_": "http://y"}) == {"ok_": "http://y"}


# ---------------------------------------------------------------------------
# build_signed_event_request — Events API envelope + v0 signature
# ---------------------------------------------------------------------------

EVENT = {"type": "message", "channel": "C0123456789", "user": "U1", "ts": "1700000000.000100", "text": "거래처\n주문"}


def test_build_signed_event_request_wraps_event_callback_envelope():
    body, headers = build_signed_event_request(EVENT, "secret", timestamp="1700000000")

    assert headers["Content-Type"] == "application/json"
    assert _verify_slack_signature("secret", "1700000000", body, headers["X-Slack-Signature"])
    envelope = json.loads(body)
    assert envelope["type"] == "event_callback"
    assert envelope["event"] == EVENT


def test_build_signed_event_request_defaults_timestamp_to_now():
    body, headers = build_signed_event_request(EVENT, "s")
    ts = int(headers["X-Slack-Request-Timestamp"])
    assert abs(time.time() - ts) < 5
    assert _verify_slack_signature("s", headers["X-Slack-Request-Timestamp"], body, headers["X-Slack-Signature"])


# ---------------------------------------------------------------------------
# build_signed_action_request — form-encoded payload= like HTTP interactivity
# ---------------------------------------------------------------------------

PAYLOAD = {
    "type": "block_actions",
    "user": {"id": "U_BOSS"},
    "actions": [{"action_id": "wpc_customer_pick:C200", "value": json.dumps({"code": "C200"})}],
}


def test_build_signed_action_request_form_encodes_payload():
    body, headers = build_signed_action_request(PAYLOAD, "secret", timestamp="1700000000")

    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert _verify_slack_signature("secret", "1700000000", body, headers["X-Slack-Signature"])
    form = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode("utf-8")).items()}
    assert json.loads(form["payload"]) == PAYLOAD


# ---------------------------------------------------------------------------
# forward_event / forward_block_actions — fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200, text=""):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse, raise_exc: Exception = None):
        self._response = response
        self._raise = raise_exc
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        if self._raise is not None:
            raise self._raise
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_forward_event_posts_signed_envelope_on_200():
    session = _FakeSession(_FakeResponse(200, json.dumps({"ok": True, "reason": "enqueued"})))
    result = await forward_event(
        EVENT, "http://127.0.0.1:8787/slack/events", "secret", session_factory=lambda: session
    )
    assert result["ok"] is True
    post = session.posts[0]
    assert post["url"] == "http://127.0.0.1:8787/slack/events"
    assert _verify_slack_signature(
        "secret", post["headers"]["X-Slack-Request-Timestamp"], post["data"], post["headers"]["X-Slack-Signature"]
    )


@pytest.mark.asyncio
async def test_forward_event_reports_non_200_status():
    session = _FakeSession(_FakeResponse(401, "invalid signature"))
    result = await forward_event(
        EVENT, "http://127.0.0.1:8787/slack/events", "secret", session_factory=lambda: session
    )
    assert result["ok"] is False
    assert result["status"] == 401


@pytest.mark.asyncio
async def test_forward_event_requires_signing_secret():
    result = await forward_event(EVENT, "http://127.0.0.1:8787/slack/events", "")
    assert result["ok"] is False
    assert "SLACK_SIGNING_SECRET" in result["error"]


@pytest.mark.asyncio
async def test_forward_event_reports_connection_errors():
    session = _FakeSession(_FakeResponse(), raise_exc=ConnectionError("refused"))
    result = await forward_event(
        EVENT, "http://127.0.0.1:8787/slack/events", "secret", session_factory=lambda: session
    )
    assert result["ok"] is False
    assert "refused" in result["error"]


@pytest.mark.asyncio
async def test_forward_block_actions_posts_signed_form_on_200():
    session = _FakeSession(_FakeResponse(200, json.dumps({"handled": True})))
    result = await forward_block_actions(
        PAYLOAD, "http://127.0.0.1:8787/slack/actions", "secret", session_factory=lambda: session
    )
    assert result["ok"] is True
    post = session.posts[0]
    assert post["url"] == "http://127.0.0.1:8787/slack/actions"
    assert post["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert _verify_slack_signature(
        "secret", post["headers"]["X-Slack-Request-Timestamp"], post["data"], post["headers"]["X-Slack-Signature"]
    )
