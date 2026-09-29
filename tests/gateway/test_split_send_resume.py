"""A split reply whose second chunk fails reaches the screen once, never twice and never cut short
behind a success (the partial_overflow contract, ``BasePlatformAdapter._with_partial_send``).

Regression for #68713 (WhatsApp) and the same send loop in the Mattermost, Slack, Teams, Matrix, SMS,
WhatsApp Cloud, Google Chat, BlueBubbles, Signal, Weixin, QQBot, Feishu and Yuanbao adapters.
"""

import asyncio
import json
import os
import re
import urllib.parse
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_google_chat import GoogleChatAdapter, _FakeHttpError, _base_config as _gchat_config
from tests.gateway.test_matrix import _make_adapter as _matrix_adapter
from tests.gateway.test_mattermost import _make_adapter as _mattermost_adapter
from tests.gateway.test_slack_send_retry import _make_adapter as _slack_adapter, _slack_api_error
from tests.gateway.test_teams import TeamsAdapter, _make_config as _teams_config
from tests.gateway.test_whatsapp_cloud import _make_adapter as _whatsapp_cloud_adapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter as _whatsapp_adapter


def _resp(status, body=None, text=""):
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value=body or {})
    resp.text = AsyncMock(return_value=text)
    return resp


class _Raising:
    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, *exc):
        return False


def _connect_refused():
    return aiohttp.ClientConnectorError(
        SimpleNamespace(host="h", port=1, ssl=None), OSError(61, "Connection refused"))


def _http_fake(adapter, attr, ok, failure, screen):
    """``session.post`` that lands every call except the second, which returns ``failure()``."""
    calls = []

    def post(url, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(kwargs["json"]["message"])
        return _AsyncCM(ok(len(calls)))
    session = MagicMock(closed=False)
    session.post = MagicMock(side_effect=post)
    setattr(adapter, attr, session)


def _mattermost(failure, screen):
    adapter = _mattermost_adapter()
    _http_fake(adapter, "_session", lambda n: _resp(201, {"id": f"post{n}"}), failure, screen)
    return adapter, 4000


def _whatsapp(failure, screen):
    adapter = _whatsapp_adapter()
    _http_fake(adapter, "_http_session", lambda n: _resp(200, {"messageId": f"wa{n}"}), failure, screen)
    return adapter, 4096


def _slack(failure, screen):
    adapter = _slack_adapter()
    calls = []

    async def post_message(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise failure()
        screen.append(kwargs["text"])
        return {"ok": True, "ts": f"1700.{len(calls)}"}
    client = MagicMock()
    client.chat_postMessage = AsyncMock(side_effect=post_message)
    adapter._get_client = lambda *_a, **_k: client
    return adapter, 39000


def _teams(failure, screen):
    adapter = TeamsAdapter(_teams_config(client_id="id", client_secret="secret", tenant_id="tenant"))
    calls = []

    async def send(chat_id, chunk):
        calls.append(1)
        if len(calls) == 2:
            raise failure()
        screen.append(chunk)
        return SimpleNamespace(id=f"act{len(calls)}")
    adapter._app = MagicMock()
    adapter._app.send = AsyncMock(side_effect=send)
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _matrix(failure, screen):
    adapter = _matrix_adapter()
    adapter._client, adapter._encryption = MagicMock(), False
    calls = []

    async def send_room_message(chat_id, content):
        calls.append(1)
        if len(calls) == 2:
            raise failure()
        screen.append(content["body"])
        return f"$event{len(calls)}"
    adapter._send_room_message = send_room_message
    return adapter, adapter.max_message_length


def _sms(failure, screen):
    from plugins.platforms.sms.adapter import SmsAdapter
    with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "ACfake", "TWILIO_AUTH_TOKEN": "fake", "TWILIO_PHONE_NUMBER": "+15550001111"}):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="fake"))
    calls = []

    def post(url, data=None, headers=None):
        body = urllib.parse.parse_qs(data().decode())["Body"][0]
        if len(body) > 1600:  # Twilio error 21617
            return _AsyncCM(_resp(400, {"code": 21617, "message": "body exceeds the 1600 character limit"}))
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(body)
        return _AsyncCM(_resp(201, {"sid": f"SM{len(calls)}"}))
    adapter._http_session = MagicMock()
    adapter._http_session.post = MagicMock(side_effect=post)
    return adapter, 1600


def _whatsapp_cloud(failure, screen):
    adapter = _whatsapp_cloud_adapter()
    calls = []

    async def post(url, headers=None, json=None):
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(json["text"]["body"])
        return _graph_response(200, {"messages": [{"id": f"wamid.{len(calls)}"}]})
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(side_effect=post)
    return adapter, adapter._outgoing_chunk_limit()


def _google_chat(failure, screen):
    adapter = GoogleChatAdapter(_gchat_config())
    adapter._chat_api, adapter._new_authed_http = MagicMock(), MagicMock()
    failed = []

    def create(parent, body, **kwargs):
        def execute(http=None):
            if len(screen) == 1 and len(failed) < 3:  # every in-adapter retry of chunk 2's first send
                failed.append(1)
                raise failure()
            screen.append(body["text"])
            return {"name": f"spaces/S/messages/{len(screen)}"}
        return SimpleNamespace(execute=execute)
    adapter._chat_api.spaces.return_value.messages.return_value.create = create
    return adapter, 4000


def _bluebubbles(failure, screen):
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={"server_url": "http://bb.invalid", "password": "pw"}))
    calls = []

    async def post(url, json=None, **kwargs):
        request = httpx.Request("POST", url)
        if "/chat/query" in url:
            return httpx.Response(200, json={"data": [{"chatIdentifier": "chat", "guid": "iMessage;-;chat"}]},
                                  request=request)
        calls.append(1)
        if len(calls) == 2:
            return failure(request)
        screen.append(json["message"])
        return httpx.Response(200, json={"status": 200, "data": {"guid": f"g{len(calls)}"}}, request=request)
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(side_effect=post)
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _signal(failure, screen):
    from gateway.platforms.signal import SignalAdapter
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": "http://signal.invalid", "account": "+15550009999"}))
    calls = []

    async def rpc(method, params, *args, **kwargs):
        if method != "send":
            return {}
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(params["message"])
        return {"timestamp": 1700000000000 + len(calls), "results": [{"type": "SUCCESS"}]}
    adapter._rpc = rpc
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _signal_first_send(failure, screen):
    """The FIRST chunk fails at the transport. A lost response means signal-cli got the request and may
    have sent it; a refused connect means nothing left."""
    from gateway.platforms.signal import SignalAdapter
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": "http://signal.invalid", "account": "+15550009999"}))
    sends = []

    async def post(url, *, json, timeout):
        request = httpx.Request("POST", url)
        if json["method"] != "send":
            return httpx.Response(200, json={"result": {}}, request=request)
        sends.append(1)
        if len(sends) == 1:
            exc = failure(request)
            if isinstance(exc, httpx.RemoteProtocolError):
                screen.append(json["params"]["message"])
            raise exc
        screen.append(json["params"]["message"])
        return httpx.Response(200, json={"result": {"timestamp": 1700000000000 + len(sends)}}, request=request)
    adapter.client = MagicMock(post=AsyncMock(side_effect=post))
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _weixin(failure, screen):
    from tests.gateway.test_weixin import _make_adapter as _weixin_adapter
    adapter = _weixin_adapter()
    adapter._send_session, adapter._token = MagicMock(), "test-token"
    calls = []

    async def send_text_chunk(*, chat_id, chunk, context_token, client_id):
        calls.append(1)
        if len(calls) == 2:
            raise failure()
        screen.append(chunk)
    adapter._send_text_chunk = send_text_chunk
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _qqbot(failure, screen):
    from gateway.platforms.qqbot import QQAdapter
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "fake", "client_secret": "fake"}))
    adapter._running, adapter._ws = True, SimpleNamespace(closed=False)
    adapter._access_token, adapter._token_expires_at = "fake-token", float("inf")
    failed = []

    async def request(method, url, json=None, **kwargs):
        request = httpx.Request(method, url)
        if len(screen) == 1 and len(failed) < 3:  # every in-adapter retry of chunk 2's first send
            failed.append(1)
            return failure(request)
        screen.append((json.get("markdown") or {}).get("content") or json["content"])
        return httpx.Response(200, json={"id": f"qq{len(screen)}"}, request=request)
    adapter._http_client = MagicMock()
    adapter._http_client.request = AsyncMock(side_effect=request)
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _feishu_response(code, status, message_id=None):
    return SimpleNamespace(success=lambda: code == 0, code=code, msg="" if code == 0 else "failed",
                           data=SimpleNamespace(message_id=message_id), raw=SimpleNamespace(status_code=status))


def _feishu(failure, screen):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = MagicMock()
    calls = []

    async def send_raw_message(*, chat_id, msg_type, payload, reply_to, metadata):
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(json.loads(payload)["text"])
        return _feishu_response(0, 200, f"om_{len(calls)}")
    adapter._send_raw_message = send_raw_message
    return adapter, adapter.MAX_MESSAGE_LENGTH


def _yuanbao(failure, screen):
    from gateway.platforms.yuanbao import YuanbaoAdapter
    adapter = YuanbaoAdapter(PlatformConfig(extra={
        "app_id": "fake", "app_secret": "fake", "ws_url": "wss://yuanbao.invalid/ws", "api_domain": "https://yuanbao.invalid"}))
    adapter._connection._ws = MagicMock()

    async def send_msg_body(chat_id, msg_body, reply_to, group_code):
        if len(screen) == 1:  # every in-adapter retry of chunk 2
            return failure()
        screen.append(msg_body[0]["msg_content"]["text"])
        return {"success": True, "msg_key": f"yb{len(screen)}"}
    adapter._outbound.sender._send_msg_body = send_msg_body
    return adapter, adapter.MAX_TEXT_CHUNK


def _graph_response(status, body):
    return SimpleNamespace(status_code=status, json=lambda: body, text="")


def _raise(exc):
    raise exc


def _wrapped_connect_refused():
    """An SDK error raised from a failed connect (mautrix raises MatrixConnectionError from it)."""
    try:
        raise _connect_refused()
    except aiohttp.ClientConnectorError as exc:
        try:
            raise RuntimeError(str(exc)) from exc
        except RuntimeError as wrapped:
            return wrapped


def _httpx_status_error(status):
    request = httpx.Request("POST", "https://smba.invalid/v3/conversations/c/activities")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


_UNSENT, _MAYBE_SENT = True, False
CASES = [
    pytest.param(_mattermost, lambda: _AsyncCM(_resp(429, text="rate limited")), _UNSENT, id="mattermost-429"),
    pytest.param(_mattermost, lambda: _Raising(_connect_refused()), _UNSENT, id="mattermost-connect"),
    pytest.param(_mattermost, lambda: _AsyncCM(_resp(502, text="bad gateway")), _MAYBE_SENT, id="mattermost-502"),
    pytest.param(_whatsapp, lambda: _AsyncCM(_resp(503, text='{"error":"Not connected to WhatsApp"}')), _UNSENT,
                 id="whatsapp-bridge-disconnected"),
    pytest.param(_whatsapp, lambda: _Raising(_connect_refused()), _UNSENT, id="whatsapp-bridge-down"),
    pytest.param(_whatsapp, lambda: _AsyncCM(_resp(500, text='{"error":"Network is unreachable"}')), _MAYBE_SENT,
                 id="whatsapp-bridge-500"),
    pytest.param(_slack, lambda: _slack_api_error(429), _UNSENT, id="slack-429"),
    pytest.param(_slack, lambda: _slack_api_error(500), _MAYBE_SENT, id="slack-500"),
    pytest.param(_teams, lambda: httpx.ConnectError("Connection refused"), _UNSENT, id="teams-connect"),
    pytest.param(_teams, lambda: _httpx_status_error(502), _MAYBE_SENT, id="teams-502"),
    pytest.param(_matrix, _wrapped_connect_refused, _UNSENT, id="matrix-connect"),
    pytest.param(_matrix, asyncio.TimeoutError, _MAYBE_SENT, id="matrix-timeout"),
    pytest.param(_sms, lambda: _AsyncCM(_resp(429, {"code": 20429, "message": "Too Many Requests"})), _UNSENT,
                 id="sms-429"),
    pytest.param(_sms, lambda: _AsyncCM(_resp(500, {"message": "Internal Server Error"})), _MAYBE_SENT, id="sms-500"),
    pytest.param(_whatsapp_cloud, lambda: _raise(httpx.ConnectError("Connection refused")), _UNSENT,
                 id="whatsapp-cloud-connect"),
    pytest.param(_whatsapp_cloud, lambda: _graph_response(500, {"error": {"code": 2, "message": "unavailable"}}),
                 _MAYBE_SENT, id="whatsapp-cloud-500"),
    pytest.param(_google_chat, lambda: _FakeHttpError(429, reason="Too Many Requests"), _UNSENT, id="google-chat-429"),
    pytest.param(_google_chat, lambda: _FakeHttpError(503, reason="Service unavailable"), _MAYBE_SENT,
                 id="google-chat-503"),
    pytest.param(_bluebubbles, lambda request: _raise(httpx.ConnectError("Connection refused", request=request)),
                 _UNSENT, id="bluebubbles-connect"),
    pytest.param(_bluebubbles, lambda request: httpx.Response(429, request=request), _UNSENT, id="bluebubbles-429"),
    pytest.param(_bluebubbles, lambda request: httpx.Response(500, request=request), _MAYBE_SENT, id="bluebubbles-500"),
    pytest.param(_signal, lambda: None, _MAYBE_SENT, id="signal-rpc-failure"),
    pytest.param(_signal_first_send, lambda request: httpx.ConnectError("Connection refused", request=request),
                 _UNSENT, id="signal-first-chunk-connect"),
    pytest.param(_signal_first_send, lambda request: httpx.RemoteProtocolError(
        "Server disconnected without sending a response.", request=request), _MAYBE_SENT,
        id="signal-first-chunk-response-lost"),
    pytest.param(_weixin, _connect_refused, _UNSENT, id="weixin-connect"),
    pytest.param(_weixin, lambda: RuntimeError("iLink sendmessage error: ret=-1 errcode=-1 errmsg=system error"),
                 _MAYBE_SENT, id="weixin-ilink-error"),
    pytest.param(_qqbot, lambda request: httpx.Response(429, json={"message": "rate limited"}, request=request), _UNSENT,
                 id="qqbot-429"),
    pytest.param(_qqbot, lambda request: httpx.Response(502, json={"message": "bad gateway"}, request=request),
                 _MAYBE_SENT, id="qqbot-502"),
    pytest.param(_feishu, lambda: _feishu_response(99991400, 429), _UNSENT, id="feishu-429"),
    pytest.param(_feishu, lambda: _feishu_response(1500, 500), _MAYBE_SENT, id="feishu-500"),
    pytest.param(_yuanbao, lambda: {"success": False, "error": "Request timeout after 30s"}, _MAYBE_SENT,
                 id="yuanbao-ack-timeout"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("make, failure, unsent", CASES)
async def test_mid_split_failure_never_duplicates_the_head_or_hides_the_tail(make, failure, unsent):
    screen = []
    adapter, limit = make(failure, screen)
    n_words = (limit * 5) // (2 * 7)  # 7-char words: 2.5 chunks' worth, so the reply splits in three
    content = " ".join(f"w{i:05d}" for i in range(n_words))

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await adapter._send_with_retry("chat", content, max_retries=2, base_delay=5)

    # Signal cuts at a length, not a word, and suffixes " (i/n)": rejoin the chunks as they read.
    words = re.findall(r"w\d{5}", "".join(re.sub(r" \(\d+/\d+\)$", "", chunk) for chunk in screen))
    assert len(words) == len(set(words)), "a delivered chunk was sent again"
    if unsent:
        # The refused chunk never reached the server: resume from it and finish the reply.
        assert result.success and len(set(words)) == n_words
    else:
        # It may have been posted: no retry or fallback may repeat it, and the reply is not reported delivered.
        assert not result.success and len(screen) == 1
        # A lost first chunk has no delivered head to mark partial; it is final instead.
        assert adapter._is_partial_delivery(result) or adapter._send_retry_is_final(result)
        assert not [c for c in sleep.await_args_list if c.args[0] >= 5], "backed off with nothing to resume"
