"""An MCP event subscription made by a served (non-default) profile of a multiplexed gateway reaches
that profile: the callback URL the server is given is answered by the gateway's webhook listener, and
the refresh tick and unsubscribe keep to that profile's own route store."""
import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.waiting import async_wait_until

PUBLIC_URL = "https://hermes.example.com"
WEBHOOK_CONFIG = (f"platforms:\n  webhook:\n    enabled: true\n    extra:\n      public_url: {PUBLIC_URL}/\n")
SUBSCRIBE = {"name": "feed.emission.created", "arguments": {"feed_id": "f-1"}}


class _RecordingSession:
    """MCP session double: records each request and grants it."""

    def __init__(self):
        self.sent = []

    async def send_request(self, request, result_type):
        self.sent.append(request.model_dump(by_alias=True, mode="json", exclude_none=True))
        expires = (datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
        return {"id": "sub_1", "refreshBefore": expires, "cursor": None, "truncated": False}


def _signed(body: bytes, secret: str) -> dict:
    msg_id, timestamp = f"msg_{uuid.uuid4().hex}", str(int(time.time()))
    key = base64.b64decode(secret.removeprefix("whsec_"))
    digest = hmac.new(key, f"{msg_id}.{timestamp}.".encode() + body, hashlib.sha256).digest()
    return {"webhook-id": msg_id, "webhook-timestamp": timestamp,
            "webhook-signature": "v1," + base64.b64encode(digest).decode(), "Content-Type": "application/json"}


def _homes(tmp_path, monkeypatch) -> tuple[Path, Path]:
    home = tmp_path / ".hermes"
    sec = home / "profiles" / "sec"
    sec.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n" + WEBHOOK_CONFIG)
    (sec / "config.yaml").write_text(WEBHOOK_CONFIG)
    return home, sec


def _routes(home: Path) -> dict:
    path = home / "webhook_subscriptions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def test_served_profile_subscription_callback_is_answered(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.run import _profile_runtime_scope
    from tools import mcp_tool_events as events

    _home, sec = _homes(tmp_path, monkeypatch)
    session = _RecordingSession()
    with _profile_runtime_scope(sec):
        result = asyncio.run(events._subscribe_event(session, SUBSCRIBE, "ambush"))
    assert "subscribed" in result, result
    delivery = session.sent[0]["params"]["delivery"]

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0}))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)

    async def verify():
        body = json.dumps({"type": "verification", "challenge": "nonce-1"}).encode()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(delivery["url"].removeprefix(PUBLIC_URL), data=body,
                                  headers=_signed(body, delivery["secret"]))
            return delivery["url"], resp.status, await resp.text()

    url, status, text = asyncio.run(verify())
    assert status == 200 and json.loads(text) == {"challenge": "nonce-1"}, (url, status, text)


def test_served_profile_refresh_and_unsubscribe_keep_to_its_store(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import mcp_tool_events as events

    home, sec = _homes(tmp_path, monkeypatch)
    session = _RecordingSession()
    with _profile_runtime_scope(sec):
        asyncio.run(events._subscribe_event(session, SUBSCRIBE, "ambush"))
    granted = session.sent[0]["params"]["delivery"]["url"]
    assert granted.startswith(f"{PUBLIC_URL}/p/sec/webhooks/"), granted

    async def refresh_tick():
        # The MCP server task starts the refresh loop in the connecting profile's context.
        server = SimpleNamespace(name="ambush", session=session, _rpc_lock=asyncio.Lock(), _config={"events": True})
        task = events.start_refresh_task(server)
        await async_wait_until(lambda: len(session.sent) == 2, what="the refresh pass")
        task.cancel()

    with _profile_runtime_scope(sec):
        asyncio.run(refresh_tick())
    assert session.sent[1]["params"]["delivery"]["url"] == granted

    with _profile_runtime_scope(sec):
        result = asyncio.run(events._unsubscribe_event(session, SUBSCRIBE, "ambush"))
    assert "unsubscribed" in result, result
    assert session.sent[2]["params"]["delivery"] == {"url": granted}
    assert _routes(sec) == {} and _routes(home) == {}
