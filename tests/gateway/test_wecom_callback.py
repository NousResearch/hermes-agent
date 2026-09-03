"""Tests for the WeCom callback-mode adapter."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import time
from unittest.mock import Mock
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.wecom.callback_adapter import (
    MESSAGE_DEDUP_TTL_SECONDS,
    WecomCallbackAdapter,
)
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


@pytest.fixture
def hermes_home(tmp_path):
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _callback_request(
    app, *, msg_id="callback-1", msg_type="text", valid_signature=True,
):
    """Build the real signed/encrypted HTTP request WeCom sends."""
    from aiohttp import StreamReader
    from aiohttp.test_utils import make_mocked_request

    plaintext = f"""
    <xml>
      <ToUserName>{app['corp_id']}</ToUserName>
      <FromUserName>alice</FromUserName>
      <CreateTime>1710000000</CreateTime>
      <MsgType>{msg_type}</MsgType>
      <Content>hello</Content>
      <MsgId>{msg_id}</MsgId>
    </xml>
    """
    crypt = WXBizMsgCrypt(app["token"], app["encoding_aes_key"], app["corp_id"])
    envelope = crypt.encrypt(plaintext, nonce="nonce123", timestamp="1710000000")
    root = ET.fromstring(envelope)
    signature = root.findtext("MsgSignature") if valid_signature else "invalid"
    query = (
        f"msg_signature={signature}"
        f"&timestamp={root.findtext('TimeStamp')}&nonce={root.findtext('Nonce')}"
    )
    reader = StreamReader(protocol=Mock(_reading_paused=False), limit=2**20)
    reader.feed_data(envelope.encode())
    reader.feed_eof()
    return make_mocked_request("POST", f"/wecom/callback?{query}", payload=reader)


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


class TestWecomCallbackRetryDeduplication:
    @pytest.mark.asyncio
    async def test_slow_claim_does_not_block_event_loop(self, hermes_home, monkeypatch):
        app = _app()
        adapter = WecomCallbackAdapter(_config([app]))
        adapter._init_replay_store()
        claim_started = threading.Event()
        release_claim = threading.Event()

        def slow_claim(*args, **kwargs):
            claim_started.set()
            assert release_claim.wait(timeout=1)
            return True

        monkeypatch.setattr(adapter, "_claim_message", slow_claim)
        callback = asyncio.create_task(adapter._handle_callback(_callback_request(app)))
        assert await asyncio.to_thread(claim_started.wait, 1)

        sentinel = asyncio.Event()
        asyncio.get_running_loop().call_soon(sentinel.set)
        await asyncio.wait_for(sentinel.wait(), timeout=0.05)
        assert not callback.done()

        release_claim.set()
        response = await callback
        assert response.status == 200
        assert adapter._message_queue.qsize() == 1
        await adapter._cleanup()

    @pytest.mark.asyncio
    async def test_lock_contention_fails_quickly_without_queueing(self, hermes_home):
        app = _app()
        adapter = WecomCallbackAdapter(_config([app]))
        adapter._init_replay_store()
        db_path = adapter._replay_store.execute("PRAGMA database_list").fetchone()[2]
        blocker = sqlite3.connect(db_path)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            response = await adapter._handle_callback(_callback_request(app))
            elapsed = time.monotonic() - started
        finally:
            blocker.rollback()
            blocker.close()

        assert response.status == 500
        assert elapsed < 1.0
        assert adapter._message_queue.empty()
        await adapter._cleanup()

    @pytest.mark.asyncio
    async def test_duplicate_survives_adapter_restart(self, hermes_home):
        app = _app()
        first = WecomCallbackAdapter(_config([app]))
        first._init_replay_store()

        response = await first._handle_callback(_callback_request(app, msg_id="same-id"))
        assert response.status == 200
        assert response.text == "success"
        assert first._message_queue.qsize() == 1
        await first._cleanup()

        restarted = WecomCallbackAdapter(_config([app]))
        restarted._init_replay_store()
        response = await restarted._handle_callback(_callback_request(app, msg_id="same-id"))
        assert response.status == 200
        assert response.text == "success"
        assert restarted._message_queue.empty()
        await restarted._cleanup()

    @pytest.mark.asyncio
    async def test_same_message_id_is_scoped_by_corp_and_agent(self, hermes_home):
        apps = [
            _app(name="corp-a", corp_id="corp-a", agent_id="1001"),
            _app(name="corp-b", corp_id="corp-b", agent_id="2002"),
        ]
        adapter = WecomCallbackAdapter(_config(apps))
        adapter._init_replay_store()

        for app in apps:
            response = await adapter._handle_callback(_callback_request(app, msg_id="shared"))
            assert response.status == 200
        assert adapter._message_queue.qsize() == 2

        # Agent ID is part of the durable key too, independently of callback routing.
        assert adapter._claim_message(apps[0], "agent-scoped") is True
        other_agent = {**apps[0], "agent_id": "9999"}
        assert adapter._claim_message(other_agent, "agent-scoped") is True
        await adapter._cleanup()

    def test_claim_is_atomic_across_connections(self, hermes_home):
        first = WecomCallbackAdapter(_config())
        second = WecomCallbackAdapter(_config())
        first._init_replay_store()
        second._init_replay_store()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda adapter: adapter._claim_message(_app(), "race"), [first, second]))

        assert sorted(claims) == [False, True]
        first._close_replay_store()
        second._close_replay_store()

    def test_ttl_is_strict_and_expired_rows_are_pruned(self, hermes_home, monkeypatch):
        from plugins.platforms.wecom import callback_adapter as callback_module

        adapter = WecomCallbackAdapter(_config())
        adapter._init_replay_store()
        monkeypatch.setattr(callback_module.time, "time", lambda: 1_000.0)
        assert adapter._claim_message(_app(), "boundary") is True

        monkeypatch.setattr(
            callback_module.time,
            "time",
            lambda: 1_000.0 + MESSAGE_DEDUP_TTL_SECONDS - 0.001,
        )
        assert adapter._claim_message(_app(), "boundary") is False

        monkeypatch.setattr(
            callback_module.time,
            "time",
            lambda: 1_000.0 + MESSAGE_DEDUP_TTL_SECONDS,
        )
        assert adapter._claim_message(_app(), "boundary") is True
        rows = adapter._replay_store.execute(
            "SELECT COUNT(*) FROM callback_replays WHERE claimed_at <= ?",
            (1_000.0,),
        ).fetchone()[0]
        assert rows == 0
        adapter._close_replay_store()

    def test_initialization_prunes_expired_metadata(self, hermes_home, monkeypatch):
        from plugins.platforms.wecom import callback_adapter as callback_module

        adapter = WecomCallbackAdapter(_config())
        adapter._init_replay_store()
        adapter._replay_store.executemany(
            "INSERT INTO callback_replays VALUES (?, ?, ?, ?)",
            [
                ("corp", "agent", "expired", 699.0),
                ("corp", "agent", "fresh", 701.0),
            ],
        )
        adapter._replay_store.commit()
        adapter._close_replay_store()

        monkeypatch.setattr(callback_module.time, "time", lambda: 1_000.0)
        adapter._init_replay_store()
        message_ids = {
            row[0]
            for row in adapter._replay_store.execute(
                "SELECT message_id FROM callback_replays"
            )
        }
        assert message_ids == {"fresh"}
        adapter._close_replay_store()

    def test_reinitialization_closes_previous_connection(self, hermes_home):
        adapter = WecomCallbackAdapter(_config())
        adapter._init_replay_store()
        previous = adapter._replay_store

        adapter._init_replay_store()

        with pytest.raises(sqlite3.ProgrammingError):
            previous.execute("SELECT 1")
        assert adapter._replay_store is not previous
        adapter._close_replay_store()

    def test_retry_metadata_is_profile_isolated(self, tmp_path):
        profile_a = tmp_path / "profile-a"
        profile_b = tmp_path / "profile-b"
        token_a = set_hermes_home_override(str(profile_a))
        try:
            first = WecomCallbackAdapter(_config())
            first._init_replay_store()
            assert first._claim_message(_app(), "same-id") is True
            first._close_replay_store()

            token_b = set_hermes_home_override(str(profile_b))
            try:
                second = WecomCallbackAdapter(_config())
                second._init_replay_store()
                assert second._claim_message(_app(), "same-id") is True
                second._close_replay_store()
            finally:
                reset_hermes_home_override(token_b)
        finally:
            reset_hermes_home_override(token_a)

    @pytest.mark.asyncio
    async def test_store_failure_returns_500_without_queueing(self, hermes_home):
        app = _app()
        adapter = WecomCallbackAdapter(_config([app]))
        adapter._init_replay_store()
        adapter._replay_store.close()

        response = await adapter._handle_callback(_callback_request(app))

        assert response.status == 500
        assert adapter._message_queue.empty()
        await adapter._cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_kwargs", "expected_status"),
        [
            ({"msg_type": "image"}, 200),
            ({"valid_signature": False}, 400),
        ],
    )
    async def test_invalid_or_ignored_callback_does_not_claim(
        self, hermes_home, monkeypatch, request_kwargs, expected_status,
    ):
        app = _app()
        adapter = WecomCallbackAdapter(_config([app]))
        adapter._init_replay_store()

        def fail_if_called(*args, **kwargs):
            raise AssertionError("invalid/ignored callbacks must not consume a replay key")

        monkeypatch.setattr(adapter, "_claim_message", fail_if_called)
        response = await adapter._handle_callback(
            _callback_request(app, msg_id="unclaimed", **request_kwargs)
        )

        assert response.status == expected_status
        assert adapter._message_queue.empty()
        await adapter._cleanup()

    @pytest.mark.asyncio
    async def test_connect_fails_closed_before_binding_when_store_init_fails(
        self, monkeypatch,
    ):
        from plugins.platforms.wecom import callback_adapter as callback_module

        adapter = WecomCallbackAdapter(_config())
        bound = False

        def fail_store(*args, **kwargs):
            raise OSError("storage unavailable")

        class NeverBind:
            def __init__(self, *args, **kwargs):
                nonlocal bound
                bound = True

        monkeypatch.setattr(callback_module, "plugin_db", fail_store, raising=False)
        monkeypatch.setattr(callback_module.web, "TCPSite", NeverBind)

        assert await adapter.connect() is False
        assert bound is False
        assert adapter._message_queue.empty()

    @pytest.mark.asyncio
    async def test_repeated_connect_closes_previous_replay_connection(
        self, hermes_home, monkeypatch,
    ):
        from plugins.platforms.wecom import callback_adapter as callback_module

        class TokenResponse:
            def json(self):
                return {"errcode": 0, "access_token": "token", "expires_in": 7200}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def get(self, *args, **kwargs):
                return TokenResponse()

            async def aclose(self):
                self.closed = True

        monkeypatch.setattr(callback_module.httpx, "AsyncClient", FakeClient)
        adapter = WecomCallbackAdapter(_config())
        assert await adapter.connect() is True
        previous = adapter._replay_store

        assert await adapter.connect() is True

        with pytest.raises(sqlite3.ProgrammingError):
            previous.execute("SELECT 1")
        assert adapter._replay_store is not previous
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_cleanup_closes_replay_store(self, hermes_home):
        adapter = WecomCallbackAdapter(_config())
        adapter._init_replay_store()
        connection = adapter._replay_store

        await adapter._cleanup()

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_cleanup_closes_every_resource_when_other_cleanup_raises(self, hermes_home):
        adapter = WecomCallbackAdapter(_config())
        adapter._init_replay_store()
        connection = adapter._replay_store

        class BrokenRunner:
            async def cleanup(self):
                raise RuntimeError("runner cleanup failed")

        class BrokenClient:
            async def aclose(self):
                raise RuntimeError("client cleanup failed")

        adapter._runner = BrokenRunner()
        adapter._http_client = BrokenClient()

        with pytest.raises(RuntimeError, match="runner cleanup failed"):
            await adapter._cleanup()

        assert adapter._runner is None
        assert adapter._http_client is None
        assert adapter._replay_store is None
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

        # Cleanup remains safe after a partially failing cleanup pass.
        await adapter._cleanup()


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


