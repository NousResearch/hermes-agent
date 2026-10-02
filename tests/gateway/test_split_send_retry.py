"""A retry after a later chunk of a split reply failed must never re-post the chunks that landed.

Adapters whose ``send()`` splits a long reply used to return a plain failure when chunk N failed, so
``BasePlatformAdapter._send_with_retry`` re-sent the WHOLE payload (transient failure) or its head as
plain text (anything else): the user read the delivered chunks twice (Feishu instead skipped the
failed chunk and reported success). Each case drives the adapter's real ``send()`` through
``_send_with_retry`` with a fake transport, faked at the adapter's own HTTP/SDK seam, that rejects
the first attempt(s) at the second message.
"""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult

_TOKEN = re.compile(r"item\d{5}")


class _Transport:
    """Accepts every message except the first ``fail_times`` attempts at the second chunk."""

    def __init__(self, fail_times: int = 1):
        self.accepted: list = []
        self.rejected = 0
        self._fail_times = fail_times

    def deliver(self, body) -> bool:
        if len(self.accepted) == 1 and self.rejected < self._fail_times:
            self.rejected += 1
            return False
        self.accepted.append(str(body))
        return True


class _BridgeResponse:
    def __init__(self, status, data, text=""):
        self.status, self._data, self._text = status, data, text

    async def json(self):
        return self._data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _teams(monkeypatch, transport):
    from plugins.platforms.teams.adapter import TeamsAdapter
    adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={"client_id": "c", "client_secret": "s", "tenant_id": "t"}))

    async def send(_chat_id, text):
        if not transport.deliver(text):
            raise RuntimeError("429 Too Many Requests")
        return SimpleNamespace(id=f"activity-{len(transport.accepted)}")

    adapter._app = SimpleNamespace(send=send, reply=send)
    return adapter, "19:chat@thread.v2", 4096


def _mattermost(monkeypatch, transport):
    from plugins.platforms.mattermost.adapter import MAX_POST_LENGTH, MattermostAdapter
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": "https://mm.example.com"}))

    async def api_post(_path, payload):
        return {"id": f"post-{len(transport.accepted) + 1}"} if transport.deliver(payload["message"]) else {}

    adapter._api_post = api_post
    return adapter, "channel-1", MAX_POST_LENGTH


def _feishu(monkeypatch, transport):
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.feishu.adapter import FeishuAdapter
    adapter = object.__new__(FeishuAdapter)
    BasePlatformAdapter.__init__(adapter, PlatformConfig(enabled=True), Platform.FEISHU)
    adapter._client = object()

    async def send_raw(*, chat_id, msg_type, payload, reply_to, metadata):
        if not transport.deliver(payload):
            return SimpleNamespace(success=lambda: False, code=99991400, msg="too many requests")
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=f"om_{len(transport.accepted)}"))

    adapter._send_raw_message = send_raw
    return adapter, "oc_chat", adapter.MAX_MESSAGE_LENGTH


def _whatsapp(monkeypatch, transport):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._bridge_unavailable = AsyncMock(return_value=None)

    def bridge_req(_method, _path, _timeout, json=None):
        if transport.deliver(json["message"]):
            return _BridgeResponse(200, {"messageId": f"wa-{len(transport.accepted)}"})
        return _BridgeResponse(429, {}, "rate limit exceeded")

    adapter._bridge_req = bridge_req
    return adapter, "15551234567@s.whatsapp.net", adapter._outgoing_chunk_limit()


def _matrix(monkeypatch, transport):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="t", extra={"homeserver": "https://m.example.org", "user_id": "@bot:example.org"}))

    async def send_message_event(_room, _event_type, content):
        if not transport.deliver(content["body"]):
            raise RuntimeError("M_LIMIT_EXCEEDED: Too Many Requests")
        return f"$event-{len(transport.accepted)}"

    adapter._client = SimpleNamespace(send_message_event=send_message_event)
    return adapter, "!room:example.org", adapter.max_message_length


def _slack(monkeypatch, transport):
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = object()
    adapter._bot_user_id = "U_BOT"

    async def post_message(**kwargs):
        if not transport.deliver(kwargs["text"]):
            exc = RuntimeError("The request to the Slack API failed. (status: 429)")
            exc.response = SimpleNamespace(status_code=429, headers={})
            raise exc
        return {"ok": True, "ts": f"1700000000.{len(transport.accepted):06d}"}

    client = SimpleNamespace(chat_postMessage=post_message)
    adapter._get_client = lambda *_args, **_kwargs: client
    return adapter, "C123", adapter.MAX_MESSAGE_LENGTH


def _yuanbao(monkeypatch, transport):
    from gateway.platforms.yuanbao import YuanbaoAdapter
    adapter = YuanbaoAdapter(PlatformConfig(
        enabled=True, extra={"app_id": "k", "app_secret": "s", "ws_url": "wss://t/ws", "api_domain": "https://t"}))
    adapter._connection._ws = SimpleNamespace(send=AsyncMock())

    async def send_msg_body(_chat_id, msg_body, _reply_to, _group_code):
        if not transport.deliver(msg_body):
            return {"success": False, "error": "too many requests"}
        return {"success": True, "msg_key": f"key-{len(transport.accepted)}"}

    adapter._outbound.sender._send_msg_body = send_msg_body
    return adapter, "direct:user1", adapter.MAX_TEXT_CHUNK


def _whatsapp_cloud(monkeypatch, transport):
    from gateway import rich_sent_store
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
    adapter = WhatsAppCloudAdapter(PlatformConfig(enabled=True, extra={"phone_number_id": "1", "access_token": "t"}))
    monkeypatch.setattr(rich_sent_store, "record_async", AsyncMock())

    async def post(_url, headers=None, json=None):
        if not transport.deliver(json["text"]["body"]):
            return SimpleNamespace(status_code=429, text="", json=lambda: {
                "error": {"message": "(#130429) Rate limit hit", "code": 130429}})
        return SimpleNamespace(status_code=200, json=lambda: {"messages": [{"id": f"wamid.{len(transport.accepted)}"}]})

    adapter._http_client = SimpleNamespace(post=post)
    return adapter, "15551112222", adapter._outgoing_chunk_limit()


def _qqbot(monkeypatch, transport):
    from gateway.platforms.qqbot import QQAdapter
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    adapter._ensure_connected = AsyncMock(return_value=True)

    async def api_request(_method, _path, body=None, **_kwargs):
        if not transport.deliver(body.get("content") or body.get("markdown")):
            raise RuntimeError("network error: connection reset")
        return {"id": f"msg-{len(transport.accepted)}"}

    adapter._api_request = api_request
    return adapter, "user-openid", adapter.MAX_MESSAGE_LENGTH


def _signal(monkeypatch, transport):
    from gateway.platforms.signal import SignalAdapter
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    adapter = SignalAdapter(PlatformConfig(
        enabled=True, extra={"http_url": "http://localhost:8080", "account": "+15550000000"}))

    async def rpc(method, params, rpc_id=None, **_kwargs):
        if method != "send":
            return {}
        return {"timestamp": 1000 + len(transport.accepted)} if transport.deliver(params["message"]) else None

    adapter._rpc = rpc
    return adapter, "+15551112222", adapter.MAX_MESSAGE_LENGTH


def _bluebubbles(monkeypatch, transport):
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    adapter = BlueBubblesAdapter(PlatformConfig(
        enabled=True, extra={"server_url": "http://localhost:1234", "password": "secret"}))

    async def api_post(_path, payload):
        if not transport.deliver(payload["message"]):
            raise RuntimeError("network unreachable")
        return {"data": {"guid": f"guid-{len(transport.accepted)}"}}

    adapter._api_post = api_post
    return adapter, "iMessage;-;+15551112222", adapter.MAX_MESSAGE_LENGTH


def _discord(monkeypatch, transport):
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="discord-token"))

    async def send(content, reference=None):
        if not transport.deliver(content):
            raise ConnectionResetError("Connection reset by peer")  # a transport error: send_path_degraded
        return SimpleNamespace(id=1000 + len(transport.accepted))

    channel = SimpleNamespace(id=555, send=send)
    adapter._client = SimpleNamespace(get_channel=lambda _channel_id: channel, fetch_channel=AsyncMock())
    return adapter, "555", adapter.MAX_MESSAGE_LENGTH


def _weixin(monkeypatch, transport):
    from gateway.platforms import weixin
    adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="wx-token", extra={
        "account_id": "bot", "send_chunk_delay_seconds": "0", "send_chunk_retries": "0"}))
    adapter._send_session = object()

    async def send_message(_session, *, text, **_kwargs):
        if not transport.deliver(text):
            raise RuntimeError("network error")
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_message", send_message)
    return adapter, "wxid_user", adapter.MAX_MESSAGE_LENGTH


def _weixin_media_first(monkeypatch, transport):
    """Weixin posts attachments before the text: the photo is the message that lands first."""
    adapter, chat_id, chunk_limit = _weixin(monkeypatch, transport)
    adapter.extract_media = lambda content: ([("photo.png", False)], content)
    adapter.filter_media_delivery_paths = lambda media, *args, **kwargs: list(media)

    async def send_image_file(chat_id, image_path, **_kwargs):
        transport.deliver(f"<attachment {image_path}>")
        return SendResult(success=True, message_id="media-1")

    adapter.send_image_file = send_image_file
    return adapter, chat_id, chunk_limit


# (builder, attempts the fake rejects at chunk 2, whether that failure is one _send_with_retry
# retries). Yuanbao and QQBot retry a chunk internally three times before reporting it failed.
_CASES = {
    "teams": (_teams, 1, True),
    "mattermost": (_mattermost, 1, False),  # "Failed to create post" is not transient: no retry
    "feishu": (_feishu, 1, True),
    "whatsapp": (_whatsapp, 1, True),
    "matrix": (_matrix, 1, True),
    "slack": (_slack, 1, True),
    "yuanbao": (_yuanbao, 3, True),
    "whatsapp_cloud": (_whatsapp_cloud, 1, True),
    "qqbot": (_qqbot, 3, True),
    "signal": (_signal, 1, False),  # "RPC send failed" is not transient: no retry
    "bluebubbles": (_bluebubbles, 1, True),
    "discord": (_discord, 1, True),
    "weixin": (_weixin, 1, True),
    "weixin_media_first": (_weixin_media_first, 1, True),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", sorted(_CASES))
async def test_retry_after_a_later_chunk_fails_never_resends_delivered_chunks(platform, monkeypatch):
    """The second message of a split reply fails after the first landed. Whatever ``_send_with_retry``
    does next, nothing reaches the user twice or out of order, and success means every chunk landed.
    A failure it retries resumes at the failed message, so the whole reply arrives exactly once."""
    build, fail_times, retried = _CASES[platform]
    transport = _Transport(fail_times)
    adapter, chat_id, chunk_limit = build(monkeypatch, transport)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    tokens = [f"item{i:05d}" for i in range(chunk_limit // 4)]  # 10 chars per token → ~2.5 chunks

    result = await adapter._send_with_retry(chat_id=chat_id, content=" ".join(tokens))

    delivered = [token for body in transport.accepted for token in _TOKEN.findall(body)]
    assert transport.rejected == fail_times  # the first message landed, then the second was rejected
    assert len(set(transport.accepted)) == len(transport.accepted), "a delivered message was sent again"
    assert delivered == tokens[:len(delivered)], "a delivered chunk was sent again (or one was skipped)"
    assert result.success == (delivered == tokens)
    assert result.success is retried


@pytest.mark.asyncio
async def test_yuanbao_resumed_tail_keeps_the_per_chat_order(monkeypatch):
    """``send_text`` holds a per-chat lock so one reply's chunks are never interleaved with another
    send to that chat. The retry of a partly delivered reply must hold it too: a send that starts
    while the tail is going out waits for the whole tail."""
    real_sleep = asyncio.sleep

    async def yield_once(*_args, **_kwargs):
        await real_sleep(0)

    transport = _Transport(fail_times=3)
    adapter, chat_id, chunk_limit = _yuanbao(monkeypatch, transport)
    send_msg_body = adapter._outbound.sender._send_msg_body
    other_send = []

    async def send_msg_body_starting_another_send(chat, msg_body, reply_to, group_code):
        if transport.rejected == 3 and not other_send:  # the retry is sending the tail now
            other_send.append(asyncio.ensure_future(adapter.send(chat_id, "interjection")))
            for _ in range(5):
                await real_sleep(0)
        return await send_msg_body(chat, msg_body, reply_to, group_code)

    adapter._outbound.sender._send_msg_body = send_msg_body_starting_another_send
    monkeypatch.setattr(asyncio, "sleep", yield_once)
    tokens = [f"item{i:05d}" for i in range(chunk_limit // 4)]

    result = await adapter._send_with_retry(chat_id=chat_id, content=" ".join(tokens))
    await other_send[0]

    assert result.success
    assert "interjection" in transport.accepted[-1], "another send landed inside the resumed tail"


def _ledger_runner(platform, adapter):
    """A bare ``GatewayRunner`` whose registry serves ``adapter``, enough for the ledger's redelivery."""
    from unittest.mock import MagicMock

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: adapter}
    runner._profile_adapters = {}
    runner.session_store = None
    runner._async_session_store = MagicMock(_store=None, clear_resume_pending=AsyncMock())
    adapter.gateway_runner = runner
    return runner


async def _send_final_ledgered(adapter, chat_id, content):
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    source = SessionSource(platform=adapter.platform, chat_id=chat_id, chat_type="dm", user_id="u1")
    return await adapter.send_final_ledgered(
        MessageEvent(text="write it up", source=source, message_id="m1"),
        f"agent:main:{adapter.platform.value}:dm:{chat_id}", content, {}, reply_to=None)


@pytest.mark.asyncio
async def test_ledger_redelivery_of_a_partial_split_final_sends_only_the_tail(monkeypatch):
    """A split final whose retries all fail after chunk 1 landed goes to the delivery ledger. Its
    redelivery resumes at the failed chunk through the same ``resume``, so every token reaches the
    chat exactly once."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    transport = _Transport(fail_times=3)  # the send and both inline retries are refused at chunk 2
    adapter, chat_id, chunk_limit = _discord(monkeypatch, transport)
    runner = _ledger_runner(Platform.DISCORD, adapter)
    tokens = [f"item{i:05d}" for i in range(chunk_limit // 4)]

    result, _ = await _send_final_ledgered(adapter, chat_id, " ".join(tokens))
    assert result.success is False
    assert await runner._redeliver_failed_obligations_for_platform(Platform.DISCORD) == 1

    assert [token for body in transport.accepted for token in _TOKEN.findall(body)] == tokens


@pytest.mark.asyncio
async def test_ledger_redelivery_after_a_reconnect_does_not_resume_through_the_replaced_adapter(monkeypatch):
    """The ``resume`` a partial split send carries is bound to the adapter that sent the head. When a
    reconnect has replaced that adapter before the ledger redelivers, the replacement sends the whole
    reply; nothing goes out through the old adapter's dead connection."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    old_transport = _Transport(fail_times=10**6)  # the old connection accepts chunk 1, then nothing
    old, chat_id, chunk_limit = _discord(monkeypatch, old_transport)
    new_transport = _Transport(fail_times=0)
    new, _, _ = _discord(monkeypatch, new_transport)
    runner = _ledger_runner(Platform.DISCORD, old)
    new.gateway_runner = runner
    sent_after_reconnect = []
    deliver = old_transport.deliver

    def deliver_on_old(body):
        if runner.adapters[Platform.DISCORD] is new:
            sent_after_reconnect.append(body)
        return deliver(body)

    old_transport.deliver = deliver_on_old
    finalize = type(old)._finalize_delivery_obligation

    async def reconnect_then_finalize(self, *args, **kwargs):
        runner.adapters[Platform.DISCORD] = new  # the reconnect installed the replacement meanwhile
        return await finalize(self, *args, **kwargs)

    monkeypatch.setattr(type(old), "_finalize_delivery_obligation", reconnect_then_finalize)
    tokens = [f"item{i:05d}" for i in range(chunk_limit // 4)]

    # send_path_degraded with a replacement live: finalizing runs the replacement's redelivery sweep.
    result, sent_by = await _send_final_ledgered(old, chat_id, " ".join(tokens))

    assert sent_by is old and result.success is False
    assert not sent_after_reconnect, "the redelivery resumed through the replaced adapter"
    assert [token for body in new_transport.accepted for token in _TOKEN.findall(body)] == tokens
