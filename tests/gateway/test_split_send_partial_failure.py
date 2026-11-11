"""A long reply whose second chunk fails is never re-sent from the start.

``BasePlatformAdapter._send_with_retry`` (the gateway's send path) retries a transient failure and
falls back to a plain-text copy on a permanent one; both re-send the whole payload unless the result
says part of it already landed (``raw_response["partial_overflow"]``). Each test drives a real adapter
through ``_send_with_retry`` with a long reply against a fake transport that accepts the first chunk
and rejects the second, and asserts that the recipient got the first chunk once and nothing else.
"""

import itertools
import json
import time
from types import SimpleNamespace

import aiohttp
import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from tests.gateway.test_discord_send import DiscordAdapter
from tests.gateway.test_google_chat import GoogleChatAdapter, _base_config, _FakeHttpError
from tests.gateway.test_matrix import _make_adapter as _make_matrix_adapter
from tests.gateway.test_slack_send_retry import _make_adapter as _make_slack_adapter, _slack_api_error
from tests.gateway.test_teams import TeamsAdapter, _make_config as _teams_config

# Over three times the largest per-message limit (Slack 39,000), so every adapter splits it.
LONG_REPLY = "\n\n".join(f"paragraph {i:04d} " + "lorem ipsum dolor sit amet " * 6 for i in range(800))


class _SecondDeliveryRejected:
    """Delivery ledger of a fake platform: the second delivery attempt is rejected, all others land."""

    def __init__(self):
        self.attempts = itertools.count(1)
        self.delivered: list[str] = []

    def accept(self, text: str) -> bool:
        if next(self.attempts) == 2:
            return False
        self.delivered.append(text)
        return True


async def _send_long_reply(adapter, chat_id: str):
    return await adapter._send_with_retry(chat_id, LONG_REPLY, base_delay=0)


def _assert_head_delivered_once(result, platform: _SecondDeliveryRejected) -> None:
    assert result.success is False
    assert len(platform.delivered) == 1, (
        f"the head of a partly delivered reply was sent again: {len(platform.delivered)} deliveries")


async def _http_stand_in(handler) -> TestServer:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.asyncio
async def test_teams():
    platform = _SecondDeliveryRejected()

    class _TeamsApp:
        async def send(self, chat_id, text):
            if not platform.accept(text):
                raise RuntimeError("Service Unavailable")
            return SimpleNamespace(id=f"activity-{len(platform.delivered)}")

    adapter = TeamsAdapter(_teams_config(client_id="id", client_secret="s", tenant_id="t"))
    adapter._app = _TeamsApp()
    _assert_head_delivered_once(await _send_long_reply(adapter, "conv-1"), platform)


@pytest.mark.asyncio
async def test_mattermost():
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    platform = _SecondDeliveryRejected()

    async def posts(request):
        body = await request.json()
        if not platform.accept(body["message"]):
            return web.Response(status=500, text="internal error")
        return web.json_response({"id": f"post-{len(platform.delivered)}"})

    server = await _http_stand_in(posts)
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": str(server.make_url(""))}))
    adapter._session = aiohttp.ClientSession()
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "channel-1"), platform)
    finally:
        await adapter._session.close()
        await server.close()


@pytest.mark.asyncio
async def test_qqbot():
    from plugins.platforms.qqbot.adapter import QQAdapter
    platform = _SecondDeliveryRejected()

    def messages(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not platform.accept(body.get("content") or json.dumps(body.get("markdown"))):
            return httpx.Response(403, json={"message": "forbidden"})
        return httpx.Response(200, json={"id": f"msg-{len(platform.delivered)}"})

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "s"}))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    adapter._access_token, adapter._token_expires_at = "token", time.time() + 3600
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(messages))
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "user-openid"), platform)
    finally:
        await adapter._http_client.aclose()


@pytest.mark.asyncio
async def test_matrix():
    platform = _SecondDeliveryRejected()

    class _MatrixClient:
        crypto = None

        async def send_message_event(self, room_id, event_type, content):
            if not platform.accept(content["body"]):
                raise RuntimeError("M_UNKNOWN: Internal server error")
            return f"$event{len(platform.delivered)}"

    adapter = _make_matrix_adapter()
    adapter._client = _MatrixClient()
    _assert_head_delivered_once(await _send_long_reply(adapter, "!room:example.org"), platform)


@pytest.mark.asyncio
async def test_google_chat():
    platform = _SecondDeliveryRejected()

    class _ChatService:
        def spaces(self):
            return self

        def messages(self):
            return self

        def create(self, **kwargs):
            class _Request:
                def execute(self, http=None):
                    if not platform.accept(kwargs["body"]["text"]):
                        raise _FakeHttpError(status=404, reason="Not found")
                    return {"name": f"spaces/A/messages/{len(platform.delivered)}"}
            return _Request()

    adapter = GoogleChatAdapter(_base_config())
    adapter._chat_api = _ChatService()
    adapter._new_authed_http = lambda: None
    _assert_head_delivered_once(await _send_long_reply(adapter, "spaces/A"), platform)


@pytest.mark.asyncio
async def test_whatsapp():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    platform = _SecondDeliveryRejected()

    async def bridge(request):
        body = await request.json()
        if not platform.accept(body["message"]):
            return web.Response(status=500, text="bridge send failed")
        return web.json_response({"messageId": f"wamid-{len(platform.delivered)}"})

    server = await _http_stand_in(bridge)
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._bridge_port = server.port
    adapter._running = True
    adapter._http_session = aiohttp.ClientSession()
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "15550001111"), platform)
    finally:
        await adapter._http_session.close()
        await server.close()


@pytest.mark.asyncio
async def test_whatsapp_cloud():
    from plugins.platforms.whatsapp_cloud.adapter import WhatsAppCloudAdapter
    platform = _SecondDeliveryRejected()

    def graph(request: httpx.Request) -> httpx.Response:
        if not platform.accept(json.loads(request.content)["text"]["body"]):
            return httpx.Response(500, json={"error": {"message": "Service temporarily unavailable"}})
        return httpx.Response(200, json={"messages": [{"id": f"wamid.{len(platform.delivered)}"}]})

    adapter = WhatsAppCloudAdapter(PlatformConfig(enabled=True, extra={"phone_number_id": "1", "access_token": "t"}))
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(graph))
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "15550001111"), platform)
    finally:
        await adapter._http_client.aclose()


@pytest.mark.asyncio
async def test_weixin():
    from plugins.platforms.weixin.adapter import WeixinAdapter
    platform = _SecondDeliveryRejected()

    async def ilink(request):
        body = await request.json()
        text = body["msg"]["item_list"][0]["text_item"]["text"]
        if not platform.accept(text):
            return web.json_response({"ret": 5, "errmsg": "system error"})
        return web.json_response({"ret": 0})

    server = await _http_stand_in(ilink)
    adapter = WeixinAdapter(PlatformConfig(enabled=True, token="t", extra={
        "account_id": "acct", "send_chunk_delay_seconds": "0", "send_chunk_retries": "0"}))
    adapter._base_url = str(server.make_url("")).rstrip("/")
    adapter._token = "t"
    adapter._send_session = aiohttp.ClientSession()
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "wxid_peer"), platform)
    finally:
        await adapter._send_session.close()
        await server.close()


@pytest.mark.asyncio
async def test_bluebubbles():
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter
    platform = _SecondDeliveryRejected()

    def server(request: httpx.Request) -> httpx.Response:
        if not platform.accept(json.loads(request.content)["message"]):
            return httpx.Response(500, json={"status": 500, "message": "Message failed to send"})
        return httpx.Response(200, json={"status": 200, "data": {"guid": f"bubble-{len(platform.delivered)}"}})

    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://bluebubbles.test", "password": "p"}))
    adapter.client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    try:
        _assert_head_delivered_once(await _send_long_reply(adapter, "iMessage;-;+15550001111"), platform)
    finally:
        await adapter.client.aclose()


@pytest.mark.asyncio
async def test_discord():
    platform = _SecondDeliveryRejected()

    class _TextChannel:
        id = 555

        async def send(self, content, reference=None):
            if not platform.accept(content):
                raise RuntimeError("400 Bad Request (error code: 50035): Invalid Form Body")
            return SimpleNamespace(id=1000 + len(platform.delivered))

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))
    channel = _TextChannel()
    adapter._client = SimpleNamespace(get_channel=lambda channel_id: channel, fetch_channel=None)
    _assert_head_delivered_once(await _send_long_reply(adapter, "555"), platform)


@pytest.mark.asyncio
async def test_slack():
    platform = _SecondDeliveryRejected()

    class _SlackWebClient:
        async def chat_postMessage(self, **kwargs):
            if not platform.accept(kwargs["text"]):
                raise _slack_api_error(500)
            return {"ok": True, "ts": f"1700000000.{len(platform.delivered):06d}"}

    adapter = _make_slack_adapter()
    client = _SlackWebClient()
    adapter._get_client = lambda chat_id, team_id="": client
    _assert_head_delivered_once(await _send_long_reply(adapter, "C123"), platform)

