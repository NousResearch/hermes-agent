"""Tests for the WeCom callback-mode adapter."""

import asyncio
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.wecom_callback import WecomCallbackAdapter
from gateway.platforms.wecom_crypto import WXBizMsgCrypt


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

    def test_signature_mismatch_raises(self):
        app = _app()
        crypt = WXBizMsgCrypt(app["token"], app["encoding_aes_key"], app["corp_id"])
        encrypted_xml = crypt.encrypt("<xml/>", nonce="n", timestamp="1")
        root = ET.fromstring(encrypted_xml)
        from gateway.platforms.wecom_crypto import SignatureError
        with pytest.raises(SignatureError):
            crypt.decrypt("bad-sig", "1", "n", root.findtext("Encrypt", default=""))


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

    def test_build_event_returns_none_for_subscribe(self):
        adapter = WecomCallbackAdapter(_config())
        xml_text = """
        <xml>
          <ToUserName>ww1234567890</ToUserName>
          <FromUserName>zhangsan</FromUserName>
          <CreateTime>1710000000</CreateTime>
          <MsgType>event</MsgType>
          <Event>subscribe</Event>
        </xml>
        """
        event = adapter._build_event(_app(), xml_text)
        assert event is None


class TestWecomCallbackRouting:
    def test_user_app_key_scopes_across_corps(self):
        adapter = WecomCallbackAdapter(_config())
        assert adapter._user_app_key("corpA", "alice") == "corpA:alice"
        assert adapter._user_app_key("corpB", "alice") == "corpB:alice"
        assert adapter._user_app_key("corpA", "alice") != adapter._user_app_key("corpB", "alice")

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

    @pytest.mark.asyncio
    async def test_send_falls_back_from_bare_user_id_when_unique(self):
        apps = [_app(name="corp-a", corp_id="corpA", agent_id="1001")]
        adapter = WecomCallbackAdapter(_config(apps=apps))
        adapter._user_app_map["corpA:alice"] = "corp-a"
        adapter._access_tokens["corp-a"] = {"token": "tok-a", "expires_at": 9999999999}

        calls = {}

        class FakeResponse:
            def json(self):
                return {"errcode": 0, "msgid": "ok2"}

        class FakeClient:
            async def post(self, url, json):
                calls["url"] = url
                calls["json"] = json
                return FakeResponse()

        adapter._http_client = FakeClient()
        result = await adapter.send("alice", "hello")

        assert result.success is True
        assert calls["json"]["agentid"] == 1001


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

    @pytest.mark.asyncio
    async def test_send_retries_with_fresh_token_on_errcode_42001(self):
        """errcode=42001 (token expired) must also trigger the refresh-retry path."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "expired", "expires_at": 9999999999}

        responses = [
            {"errcode": 42001, "errmsg": "access_token expired"},
            {"errcode": 0, "msgid": "msg-42"},
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
                        return {"errcode": 0, "access_token": "renewed", "expires_in": 7200}
                return R()

        adapter._http_client = FakeClient()
        result = await adapter.send("alice", "hello")

        assert result.success is True
        assert len(post_calls) == 2

    @pytest.mark.asyncio
    async def test_send_does_not_retry_on_non_token_errcode(self):
        """Errors unrelated to token validity must fail immediately without retrying."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "good", "expires_at": 9999999999}

        post_calls = []

        class FakeClient:
            async def post(self, url, json=None, **kw):
                post_calls.append(url)

                class R:
                    def json(inner):
                        return {"errcode": 60020, "errmsg": "not allow to access"}
                return R()

        adapter._http_client = FakeClient()
        result = await adapter.send("alice", "hello")

        assert result.success is False
        assert len(post_calls) == 1

    @pytest.mark.asyncio
    async def test_send_fails_cleanly_when_retry_also_fails(self):
        """If the refreshed token is also rejected, return failure without looping further."""
        adapter = WecomCallbackAdapter(_config())
        adapter._access_tokens["test-app"] = {"token": "bad1", "expires_at": 9999999999}

        post_calls = []

        class FakeClient:
            async def post(self, url, json=None, **kw):
                post_calls.append(url)

                class R:
                    def json(inner):
                        return {"errcode": 42001, "errmsg": "access_token expired"}
                return R()

            async def get(self, url, params=None, **kw):
                class R:
                    def json(inner):
                        return {"errcode": 0, "access_token": "bad2", "expires_in": 7200}
                return R()

        adapter._http_client = FakeClient()
        result = await adapter.send("alice", "hello")

        assert result.success is False
        assert len(post_calls) == 2


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


class TestWecomCallbackAccessLogRedaction:
    """The WeCom URL-verification handshake and inbound callbacks carry the
    ``msg_signature`` HMAC (plus ``echostr``/``timestamp``/``nonce``) in the
    request *query string* (see ``_handle_verify`` / ``_handle_callback``).
    aiohttp's default access logger would write that full request target to
    agent.log verbatim, leaking the signature. ``connect()`` must install a
    redacting access-log class that keeps request logging (so ``/health`` and
    traffic stay observable) while replacing the sensitive query values with
    ``REDACTED`` -- it must NOT disable access logging wholesale.
    """

    def test_redact_request_target_redacts_sensitive_query_params(self):
        from gateway.platforms.wecom_callback import _redact_request_target

        target = (
            "/wecom/callback?msg_signature=abc123&timestamp=1700000000"
            "&nonce=xyz&echostr=secret-blob&foo=keep"
        )
        redacted = _redact_request_target(target)

        # sensitive values gone
        assert "abc123" not in redacted
        assert "secret-blob" not in redacted
        assert "1700000000" not in redacted
        assert "xyz" not in redacted
        assert redacted.count("REDACTED") == 4
        # path and non-sensitive params preserved for observability
        assert redacted.startswith("/wecom/callback?")
        assert "foo=keep" in redacted

    def test_redact_request_target_leaves_plain_path_untouched(self):
        from gateway.platforms.wecom_callback import _redact_request_target

        assert _redact_request_target("/health") == "/health"

    def test_redacting_access_logger_logs_redacted_target(self, caplog):
        import logging

        from gateway.platforms.wecom_callback import _build_redacting_access_logger

        logger_class = _build_redacting_access_logger()
        assert logger_class is not None

        captured_logger = logging.getLogger("wecom.test.access")

        class _Req:
            method = "POST"
            path_qs = "/wecom/callback?msg_signature=topsecret&foo=keep"

        class _Resp:
            status = 200
            body_length = 12

        access_logger = logger_class(captured_logger, log_format="")
        with caplog.at_level(logging.INFO, logger="wecom.test.access"):
            access_logger.log(_Req(), _Resp(), 0.001)

        record_text = "\n".join(r.getMessage() for r in caplog.records)
        # the signature value must never be persisted; the path stays for ops
        assert "topsecret" not in record_text
        assert "REDACTED" in record_text
        assert "/wecom/callback" in record_text
        assert "foo=keep" in record_text

    @pytest.mark.asyncio
    async def test_connect_installs_redacting_access_log_class(self, monkeypatch):
        from gateway.platforms import wecom_callback as mod

        captured: dict = {}

        class FakeRunner:
            def __init__(self, app, **kwargs):
                captured["kwargs"] = kwargs

            async def setup(self):
                return None

            async def cleanup(self):
                return None

        class FakeSite:
            def __init__(self, *args, **kwargs):
                pass

            async def start(self):
                return None

            async def stop(self):
                return None

        class FakeHttpClient:
            def __init__(self, *args, **kwargs):
                pass

            async def aclose(self):
                return None

        # The port-in-use probe connects to 127.0.0.1:<port>; force the
        # "port free" branch deterministically.
        class FakeSocket:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def settimeout(self, _):
                return None

            def connect(self, _):
                raise ConnectionRefusedError

        monkeypatch.setattr(mod.web, "AppRunner", FakeRunner)
        monkeypatch.setattr(mod.web, "TCPSite", FakeSite)
        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeHttpClient)
        monkeypatch.setattr(mod._socket, "socket", lambda *a, **k: FakeSocket())

        adapter = WecomCallbackAdapter(_config())

        # Avoid the real poll loop and token-refresh network calls.
        async def _noop_refresh(app):
            return "tok"

        monkeypatch.setattr(adapter, "_refresh_access_token", _noop_refresh)

        async def _noop_poll():
            return None

        monkeypatch.setattr(adapter, "_poll_loop", _noop_poll)

        result = await adapter.connect()
        await adapter.disconnect()

        assert result is True
        assert "access_log_class" in captured["kwargs"], (
            "AppRunner must be constructed with an explicit access_log_class "
            "so request logging stays enabled with redaction"
        )
        access_log_class = captured["kwargs"]["access_log_class"]
        # The installed class must be a real aiohttp access logger (logging is
        # NOT disabled) and must redact the sensitive query params.
        from aiohttp.abc import AbstractAccessLogger

        assert access_log_class is not None
        assert issubclass(access_log_class, AbstractAccessLogger)
        # access_log must NOT be set to None — that would kill /health logging.
        assert captured["kwargs"].get("access_log", "default") is not None
