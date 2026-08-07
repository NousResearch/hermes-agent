"""Tests for the WeCom callback-mode adapter."""

import asyncio
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
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
    @pytest.mark.asyncio
    async def test_build_event_extracts_text_message(self):
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
        event = await adapter._build_event(_app(), xml_text)
        assert event is not None
        assert event.source is not None
        assert event.source.user_id == "zhangsan"
        assert event.source.chat_id == "ww1234567890:zhangsan"
        assert event.message_id == "123456789"
        assert event.text == "\u4f60\u597d"

    @pytest.mark.asyncio
    async def test_build_event_image_returns_none_no_double_event(self):
        """Inbound image must NOT produce a placeholder event — only the
        background download task queues a single PHOTO event later.
        (sweeper review: double-event bug)
        """
        adapter = WecomCallbackAdapter(_config())
        xml_text = """
        <xml>
          <ToUserName>ww1234567890</ToUserName>
          <FromUserName>zhangsan</FromUserName>
          <CreateTime>1710000000</CreateTime>
          <MsgType>image</MsgType>
          <PicUrl>https://example.com/photo.jpg</PicUrl>
          <MediaId>MEDIA123</MediaId>
          <MsgId>img001</MsgId>
        </xml>
        """
        # _build_event must return None for image — no placeholder event
        event = await adapter._build_event(_app(), xml_text)
        assert event is None
        # The background download task should be tracked in _background_tasks
        assert len(adapter._background_tasks) > 0


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
        event = await adapter._build_event(
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


class TestWecomCallbackImageSend:
    """Regression coverage for send_image_file (PR #75341)."""

    @pytest.mark.asyncio
    async def test_send_image_file_uploads_then_sends(self, tmp_path):
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "tok", "expires_at": 9999999999}
        adapter._user_app_map["ww1234567890:alice"] = "test-app"

        # Create a small test image
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake JPEG data")

        post_calls = []

        class FakeResponse:
            def __init__(self, data):
                self._data = data
            def json(self):
                return self._data

        class FakeClient:
            async def post(self, url, json=None, files=None, **kw):
                post_calls.append({"url": url, "json": json, "files": files})
                if "media/upload" in url:
                    return FakeResponse({"errcode": 0, "media_id": "media-123", "type": "image"})
                return FakeResponse({"errcode": 0, "msgid": "img-msg-1"})

        adapter._http_client = FakeClient()
        result = await adapter.send_image_file("ww1234567890:alice", str(img))

        assert result.success is True
        assert result.message_id == "img-msg-1"
        assert len(post_calls) == 2  # upload + send
        assert "media/upload" in post_calls[0]["url"]
        assert "message/send" in post_calls[1]["url"]
        assert post_calls[1]["json"]["msgtype"] == "image"
        assert post_calls[1]["json"]["image"]["media_id"] == "media-123"

    @pytest.mark.asyncio
    async def test_send_image_file_token_refresh_on_40001(self, tmp_path):
        """Token refresh must work for image upload path too."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "stale", "expires_at": 9999999999}
        adapter._user_app_map["ww1234567890:alice"] = "test-app"

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake JPEG data")

        upload_responses = [
            {"errcode": 40001, "errmsg": "invalid credential"},
            {"errcode": 0, "media_id": "fresh-media", "type": "image"},
        ]
        send_responses = [{"errcode": 0, "msgid": "img-ok"}]
        upload_idx = [0]
        send_idx = [0]

        class FakeResponse:
            def __init__(self, data):
                self._data = data
            def json(self):
                return self._data

        class FakeClient:
            async def post(self, url, json=None, files=None, **kw):
                if "media/upload" in url:
                    resp = upload_responses[upload_idx[0]]
                    upload_idx[0] += 1
                    return FakeResponse(resp)
                resp = send_responses[send_idx[0]]
                send_idx[0] += 1
                return FakeResponse(resp)

            async def get(self, url, params=None, **kw):
                return FakeResponse({"errcode": 0, "access_token": "fresh-tok", "expires_in": 7200})

        adapter._http_client = FakeClient()
        result = await adapter.send_image_file("ww1234567890:alice", str(img))

        assert result.success is True
        assert adapter._access_tokens["test-app"]["token"] == "fresh-tok"

    @pytest.mark.asyncio
    async def test_send_image_file_rejects_unsafe_path(self):
        """Path traversal must be rejected by validate_media_delivery_path."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "tok", "expires_at": 9999999999}
        adapter._user_app_map["ww1234567890:alice"] = "test-app"
        result = await adapter.send_image_file("ww1234567890:alice", "../../etc/passwd")
        assert result.success is False


class TestWecomCallbackInboundImageFailure:
    """Regression: cache failure must queue a degraded event, not lose the message."""

    @pytest.mark.asyncio
    async def test_cache_and_queue_image_failure_still_queues_event(self, monkeypatch):
        adapter = WecomCallbackAdapter(_config())
        # Monkey-patch the shared helper to raise (simulates download failure)
        from gateway.platforms import base as base_mod
        async def _fail(url, ext=".jpg"):
            raise RuntimeError("network unreachable")
        monkeypatch.setattr(base_mod, "cache_image_from_url", _fail)

        from gateway.platforms.base import MessageEvent, MessageType
        source = adapter.build_source(
            chat_id="ww1234567890:zhangsan", chat_name="zhangsan",
            chat_type="dm", user_id="zhangsan", user_name="zhangsan",
        )
        await adapter._cache_and_queue_image(
            "https://example.com/photo.jpg", "MEDIA123",
            source, "img001", "<xml/>",
        )
        # Must have queued exactly one event
        assert not adapter._message_queue.empty()
        event = await adapter._message_queue.get()
        assert event.message_type == MessageType.PHOTO
        # No cached file → degraded text placeholder
        assert event.text == "[图片]"
        assert event.media_urls == []


