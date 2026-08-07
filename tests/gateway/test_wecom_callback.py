"""Tests for the WeCom callback-mode adapter."""

import asyncio
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
from plugins.platforms.wecom.wecom_crypto import WXBizMsgCrypt


def _app(name="test-app", corp_id="ww1234567890", agent_id="1000002"):
    return {
        "name": name,
        "corp_id": corp_id,
        "corp_secret": "test-secret",
        "agent_id": agent_id,
        "token": "test-callback-token",
        "encoding_aes_key": "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",
    }


def _config(apps=None):
    return PlatformConfig(
        enabled=True,
        extra={"mode": "callback", "host": "127.0.0.1", "port": 0, "apps": apps or [_app()]},
    )


class TestWecomCrypto:
    def test_roundtrip_encrypt_decrypt(self):
        app = _app()
        crypt = WXBizMsgCrypt(app["token"], app["encoding_aes_key"], app["corp_id"])
        encrypted_xml = crypt.encrypt(
            "<xml><Content>hello</Content></xml>", nonce="nonce123", timestamp="123456",
        )
        root = ET.fromstring(encrypted_xml)
        decrypted = crypt.decrypt(
            root.findtext("MsgSignature", default=""),
            root.findtext("TimeStamp", default=""),
            root.findtext("Nonce", default=""),
            root.findtext("Encrypt", default=""),
        )
        assert b"<Content>hello</Content>" in decrypted


class TestWecomCallbackEventConstruction:
    def test_build_event_extracts_text_message(self):
        adapter = WecomCallbackAdapter(_config())
        xml_text = """
        <xml>
          <ToUserName>ww1234567890</ToUserName>
          <FromUserName>zhangsan</FromUserName>
          <CreateTime>1710000000</CreateTime>
          <MsgType>text</MsgType>
          <Content>\u4f60\u597d</Content>
          <MsgId>123456789</MsgId>
        </xml>
        """
        event = adapter._build_event(_app(), xml_text)
        assert event is not None
        assert event.source is not None
        assert event.source.user_id == "zhangsan"
        assert event.source.chat_id == "ww1234567890:zhangsan"
        assert event.message_id == "123456789"
        assert event.text == "\u4f60\u597d"


class TestWecomCallbackRouting:

    @pytest.mark.asyncio
    async def test_send_selects_correct_app_for_scoped_chat_id(self):
        apps = [
            _app(name="corp-a", corp_id="corpA", agent_id="1001"),
            _app(name="corp-b", corp_id="corpB", agent_id="2002"),
        ]
        adapter = WecomCallbackAdapter(_config(apps=apps))
        adapter._user_app_map["corpB:alice"] = "corp-b"
        adapter._access_tokens["corp-b"] = {"token": "tok-b", "expires_at": 9999999999}

        calls = {}

        class FakeResponse:
            def json(self):
                return {"errcode": 0, "msgid": "ok1"}

        class FakeClient:
            async def post(self, url, json):
                calls["url"] = url
                calls["json"] = json
                return FakeResponse()

        adapter._http_client = FakeClient()
        result = await adapter.send("corpB:alice", "hello")

        assert result.success is True
        assert calls["json"]["touser"] == "alice"
        assert calls["json"]["agentid"] == 2002
        assert "tok-b" in calls["url"]


class TestWecomCallbackSendTokenRefresh:
    @pytest.mark.asyncio
    async def test_send_retries_with_fresh_token_on_errcode_40001(self):
        """errcode=40001 must evict the cached token, refresh, and retry once."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "stale", "expires_at": 9999999999}
        adapter._user_app_map["ww1234567890:alice"] = "test-app"

        responses = [
            {"errcode": 40001, "errmsg": "invalid credential"},
            {"errcode": 0, "msgid": "msg-ok"},
        ]
        post_calls = []

        class FakeClient:
            async def post(self, url, json=None, **kw):
                post_calls.append(url)

                class R:
                    def json(inner):
                        return responses[len(post_calls) - 1]
                return R()

            async def get(self, url, params=None, **kw):
                class R:
                    def json(inner):
                        return {"errcode": 0, "access_token": "fresh", "expires_in": 7200}
                return R()

        adapter._http_client = FakeClient()
        result = await adapter.send("ww1234567890:alice", "hello")

        assert result.success is True
        assert result.message_id == "msg-ok"
        assert len(post_calls) == 2
        assert "fresh" in post_calls[1]
        assert adapter._access_tokens["test-app"]["token"] == "fresh"


class TestWecomCallbackPollLoop:
    @pytest.mark.asyncio
    async def test_poll_loop_dispatches_handle_message(self, monkeypatch):
        adapter = WecomCallbackAdapter(_config())
        calls = []

        async def fake_handle_message(event):
            calls.append(event.text)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        event = adapter._build_event(
            _app(),
            """
            <xml>
              <ToUserName>ww1234567890</ToUserName>
              <FromUserName>lisi</FromUserName>
              <CreateTime>1710000000</CreateTime>
              <MsgType>text</MsgType>
              <Content>test</Content>
              <MsgId>m2</MsgId>
            </xml>
            """,
        )
        task = asyncio.create_task(adapter._poll_loop())
        await adapter._message_queue.put(event)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["test"]


class TestWecomCallbackBodySizeLimit:
    """Pre-auth oversized-body rejection (DoS hardening, PR #10192)."""

    def _request(self, body_bytes):
        from unittest.mock import Mock

        from aiohttp import StreamReader
        from aiohttp.test_utils import make_mocked_request

        protocol = Mock(_reading_paused=False)
        reader = StreamReader(protocol=protocol, limit=2 ** 20)
        reader.feed_data(body_bytes)
        reader.feed_eof()
        return make_mocked_request(
            "POST", "/wecom/callback?msg_signature=s&timestamp=1&nonce=n",
            payload=reader,
        )

    @pytest.mark.asyncio
    async def test_oversized_body_rejected_with_413(self):
        from plugins.platforms.wecom.callback_adapter import _MAX_BODY

        adapter = WecomCallbackAdapter(_config())
        oversized = b"<xml>" + b"A" * (_MAX_BODY + 1) + b"</xml>"
        response = await adapter._handle_callback(self._request(oversized))
        assert response.status == 413




class TestStandaloneSend:
    """Out-of-process delivery for ``wecom_callback``.

    ``send_message(platform="wecom_callback")`` and ``deliver=wecom_callback``
    cron jobs route through the registry's ``standalone_sender_fn``. The
    callback platform registered none, so those sends had nothing to call.

    The sender must NOT go through ``connect()``: callback delivery is
    outbound-only, while ``connect()`` binds the callback port — and refuses
    outright when the gateway already holds it, which is precisely when an
    out-of-process send happens.
    """

    def test_registered_on_the_callback_platform(self):
        import plugins.platforms.wecom.adapter as wecom_adapter

        registered = {}

        class _Ctx:
            def register_platform(self, **kwargs):
                registered[kwargs["name"]] = kwargs

        wecom_adapter.register(_Ctx())

        assert registered["wecom"].get("standalone_sender_fn") is not None
        assert registered["wecom_callback"].get("standalone_sender_fn") is not None, (
            "deliver=wecom_callback has no sender to call"
        )

    def test_send_succeeds_without_binding_the_callback_port(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_adapter
        from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

        sent = {}

        class _Result:
            success = True
            message_id = "msg-1"
            error = None

        async def _fake_send(self, chat_id, content, **_kw):
            sent["chat_id"] = chat_id
            sent["content"] = content
            return _Result()

        def _explode(self, *a, **k):
            raise AssertionError(
                "connect() must not run — it binds the callback port"
            )

        monkeypatch.setattr(WecomCallbackAdapter, "send", _fake_send)
        monkeypatch.setattr(WecomCallbackAdapter, "connect", _explode)
        monkeypatch.setattr(
            wecom_adapter, "_build_callback_adapter",
            lambda cfg: WecomCallbackAdapter(cfg),
        )
        monkeypatch.setattr(
            "plugins.platforms.wecom.callback_adapter.check_wecom_callback_requirements",
            lambda: True,
        )

        result = asyncio.run(
            wecom_adapter._callback_standalone_send(_config(), "app:user1", "hello")
        )

        assert result == {
            "success": True,
            "platform": "wecom_callback",
            "chat_id": "app:user1",
            "message_id": "msg-1",
        }
        assert sent == {"chat_id": "app:user1", "content": "hello"}

    def test_http_client_is_opened_and_closed(self, monkeypatch):
        """The outbound client is the only resource the sender may hold."""
        import plugins.platforms.wecom.adapter as wecom_adapter
        from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

        seen = {}

        class _Result:
            success = True
            message_id = "m"
            error = None

        async def _fake_send(self, chat_id, content, **_kw):
            seen["client_during_send"] = self._http_client is not None
            return _Result()

        adapter_holder = {}

        def _build(cfg):
            adapter = WecomCallbackAdapter(cfg)
            adapter_holder["a"] = adapter
            return adapter

        monkeypatch.setattr(WecomCallbackAdapter, "send", _fake_send)
        monkeypatch.setattr(wecom_adapter, "_build_callback_adapter", _build)
        monkeypatch.setattr(
            "plugins.platforms.wecom.callback_adapter.check_wecom_callback_requirements",
            lambda: True,
        )

        asyncio.run(
            wecom_adapter._callback_standalone_send(_config(), "app:user1", "hi")
        )

        assert seen["client_during_send"] is True
        assert adapter_holder["a"]._http_client is None, "client was left open"

    def test_send_failure_is_reported(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_adapter
        from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

        class _Result:
            success = False
            message_id = None
            error = "errcode 40013"

        async def _fake_send(self, chat_id, content, **_kw):
            return _Result()

        monkeypatch.setattr(WecomCallbackAdapter, "send", _fake_send)
        monkeypatch.setattr(
            wecom_adapter, "_build_callback_adapter",
            lambda cfg: WecomCallbackAdapter(cfg),
        )
        monkeypatch.setattr(
            "plugins.platforms.wecom.callback_adapter.check_wecom_callback_requirements",
            lambda: True,
        )

        result = asyncio.run(
            wecom_adapter._callback_standalone_send(_config(), "app:user1", "hi")
        )

        assert "errcode 40013" in result["error"]
        assert "success" not in result

    def test_missing_requirements_reported_not_raised(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_adapter

        monkeypatch.setattr(
            "plugins.platforms.wecom.callback_adapter.check_wecom_callback_requirements",
            lambda: False,
        )

        result = asyncio.run(
            wecom_adapter._callback_standalone_send(_config(), "app:user1", "hi")
        )

        assert "requirements not met" in result["error"].lower()


class TestEnsureHttpClientSeam:
    def test_idempotent_and_closes(self):
        adapter = WecomCallbackAdapter(_config())
        adapter._ensure_http_client()
        first = adapter._http_client
        adapter._ensure_http_client()

        assert adapter._http_client is first, "must not replace a live client"

        asyncio.run(adapter.aclose_http_client())
        assert adapter._http_client is None
        asyncio.run(adapter.aclose_http_client())  # second close is harmless
