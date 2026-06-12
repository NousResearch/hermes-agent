"""Tests for the Synology Chat platform adapter (bundled plugin)."""
import asyncio
import json
import os
import urllib.parse

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig

from plugins.platforms.synology_chat.adapter import (
    SynologyChatAdapter,
    _apply_yaml_config,
    _env_enablement,
    _is_connected,
    _standalone_send,
    check_synology_chat_requirements,
    interactive_setup,
)

BOT_TOKEN = "b" * 64
CHAN_A_TOKEN = "a" * 64
CHAN_B_TOKEN = "c" * 64


def _make_adapter(**extra_overrides) -> SynologyChatAdapter:
    extra = {
        "incoming_url": "https://nas.example.com:5001/webapi/entry.cgi?api=SYNO.Chat.External&method=chatbot&version=2&token=%22x%22",
        "channels": {
            "42": {"token": CHAN_A_TOKEN, "incoming_url": "https://nas.example.com:5001/incoming-42"},
            "devops": {"token": CHAN_B_TOKEN, "incoming_url": "https://nas.example.com:5001/incoming-devops"},
        },
    }
    extra.update(extra_overrides)
    config = PlatformConfig(enabled=True, token=BOT_TOKEN, extra=extra)
    return SynologyChatAdapter(config)


def _form_body(**fields) -> bytes:
    return urllib.parse.urlencode(fields).encode()


class _FakeRequest:
    def __init__(self, body: bytes, content_type="application/x-www-form-urlencoded",
                 remote="203.0.113.10", content_length="auto"):
        self._body = body
        self.content_type = content_type
        self.content_length = len(body) if content_length == "auto" else content_length
        self.remote = remote

    async def read(self):
        return self._body


async def _inbound(adapter, *, remote="203.0.113.10", content_type="application/x-www-form-urlencoded", **fields):
    captured = {}

    async def fake_handle(event):
        captured["event"] = event

    adapter.handle_message = fake_handle
    request = _FakeRequest(_form_body(**fields), content_type=content_type, remote=remote)
    response = await adapter._handle_inbound(request)
    return response, captured.get("event")


def _post_mock(status=200, body="ok"):
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


# ---------------------------------------------------------------------------
# Requirements / construction
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_requires_both_env(self, monkeypatch):
        monkeypatch.delenv("SYNOLOGY_CHAT_TOKEN", raising=False)
        monkeypatch.delenv("SYNOLOGY_CHAT_INCOMING_URL", raising=False)
        assert check_synology_chat_requirements() is False
        monkeypatch.setenv("SYNOLOGY_CHAT_TOKEN", BOT_TOKEN)
        assert check_synology_chat_requirements() is False
        monkeypatch.setenv("SYNOLOGY_CHAT_INCOMING_URL", "https://nas/webapi")
        assert check_synology_chat_requirements() is True

    def test_channels_from_config_and_env(self, monkeypatch):
        monkeypatch.setenv("SYNOLOGY_CHANNEL_TOKEN_77", "t" * 64)
        monkeypatch.setenv("SYNOLOGY_CHANNEL_WEBHOOK_77", "https://nas/incoming-77")
        adapter = _make_adapter()
        assert adapter._channels["42"]["token"] == CHAN_A_TOKEN
        assert adapter._channels["77"]["incoming_url"] == "https://nas/incoming-77"


# ---------------------------------------------------------------------------
# Inbound: multi-token binding (the core security property)
# ---------------------------------------------------------------------------

class TestTokenBinding:
    def test_bot_token_dm_ok(self):
        assert _make_adapter()._validate_inbound_token(BOT_TOKEN, "") is True

    def test_channel_token_own_channel_ok(self):
        assert _make_adapter()._validate_inbound_token(CHAN_A_TOKEN, "42") is True

    def test_channel_token_cross_channel_rejected(self):
        assert _make_adapter()._validate_inbound_token(CHAN_A_TOKEN, "devops") is False

    def test_bot_token_on_channel_rejected(self):
        assert _make_adapter()._validate_inbound_token(BOT_TOKEN, "42") is False

    def test_channel_token_on_dm_rejected(self):
        assert _make_adapter()._validate_inbound_token(CHAN_A_TOKEN, "") is False

    def test_unconfigured_channel_rejected(self):
        assert _make_adapter()._validate_inbound_token(CHAN_A_TOKEN, "unknown") is False

    def test_fail_closed_on_empty(self):
        adapter = _make_adapter()
        assert adapter._validate_inbound_token("", "") is False
        adapter._token = ""
        assert adapter._validate_inbound_token("anything", "") is False


# ---------------------------------------------------------------------------
# Inbound: pipeline end-to-end (parsing, routing, event)
# ---------------------------------------------------------------------------

class TestInboundPipeline:
    def test_dm_payload(self):
        adapter = _make_adapter()
        response, event = asyncio.run(_inbound(
            adapter, token=BOT_TOKEN, user_id="4", username="alice", text="hello",
        ))
        assert response.status == 204
        assert event.source.chat_type == "dm"
        assert event.source.chat_id == "dm:4"  # type-prefixed (no DM<->channel collision)
        assert event.source.user_id == "4"
        assert event.message_id is None
        assert event.text == "hello"

    def test_channel_payload_with_trigger_word(self):
        adapter = _make_adapter()
        response, event = asyncio.run(_inbound(
            adapter, token=CHAN_A_TOKEN, user_id="4", username="alice",
            text="Hermes what time is it", channel_id="42",
            channel_name="general", trigger_word="Hermes", post_id="123",
        ))
        assert response.status == 204
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "ch:42"
        assert event.message_id == "123"
        assert event.text == "what time is it"

    def test_dm_user_id_colliding_with_channel_id_stays_dm(self):
        # user_id "42" == configured channel "42": prefixing keeps them apart.
        adapter = _make_adapter()
        _, event = asyncio.run(_inbound(
            adapter, token=BOT_TOKEN, user_id="42", text="secret",
        ))
        assert event.source.chat_id == "dm:42"
        url, extra, err = adapter._resolve_destination(event.source.chat_id)
        assert err is None and extra == {"user_ids": [42]}
        assert "method=chatbot" in url  # NOT the channel-42 public webhook

    def test_empty_channel_id_routes_as_dm(self):
        adapter = _make_adapter()
        _, event = asyncio.run(_inbound(
            adapter, token=BOT_TOKEN, user_id="4", text="hi", channel_id="",
        ))
        assert event.source.chat_type == "dm"

    def test_channel_name_fallback_without_channel_id(self):
        adapter = _make_adapter()
        _, event = asyncio.run(_inbound(
            adapter, token=CHAN_B_TOKEN, user_id="4", text="Hermes hi",
            channel_name="devops", trigger_word="Hermes",
        ))
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "ch:devops"

    def test_invalid_token_401(self):
        adapter = _make_adapter()
        response, event = asyncio.run(_inbound(adapter, token="wrong", user_id="4", text="hi"))
        assert response.status == 401 and event is None

    def test_cross_channel_token_401(self):
        adapter = _make_adapter()
        response, _ = asyncio.run(_inbound(
            adapter, token=CHAN_A_TOKEN, user_id="4", text="x", channel_id="devops",
        ))
        assert response.status == 401

    def test_missing_text_400(self):
        adapter = _make_adapter()
        response, _ = asyncio.run(_inbound(adapter, token=BOT_TOKEN, user_id="4"))
        assert response.status == 400

    def test_missing_user_id_400(self):
        adapter = _make_adapter()
        response, _ = asyncio.run(_inbound(adapter, token=BOT_TOKEN, text="hi"))
        assert response.status == 400

    def test_json_fallback(self):
        adapter = _make_adapter()
        captured = {}
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured.__setitem__("e", e))
        body = json.dumps({"token": BOT_TOKEN, "user_id": "4", "text": "hi"}).encode()
        request = _FakeRequest(body, content_type="application/json")
        response = asyncio.run(adapter._handle_inbound(request))
        assert response.status == 204
        assert captured["e"].text == "hi"

    def test_oversized_body_413_with_content_length(self):
        adapter = _make_adapter()
        request = _FakeRequest(b"x" * 70000)
        assert asyncio.run(adapter._handle_inbound(request)).status == 413

    def test_oversized_body_413_chunked_no_content_length(self):
        # Spoofed/chunked request: content_length=None -> the post-read len()
        # guard must still reject (the aiohttp client_max_size is the primary
        # guard in prod; this covers the secondary one).
        adapter = _make_adapter()
        request = _FakeRequest(b"x" * 70000, content_length=None)
        assert asyncio.run(adapter._handle_inbound(request)).status == 413

    def test_agent_error_never_bounces_webhook(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        request = _FakeRequest(_form_body(token=BOT_TOKEN, user_id="4", text="hi"))
        assert asyncio.run(adapter._handle_inbound(request)).status == 204

    def test_ip_rate_limit(self):
        adapter = _make_adapter()
        for _ in range(120):
            assert adapter._ip_rate_limited("10.0.0.1") is False
        assert adapter._ip_rate_limited("10.0.0.1") is True
        assert adapter._ip_rate_limited("10.0.0.2") is False

    def test_ip_window_evicts_empty(self, monkeypatch):
        adapter = _make_adapter()
        t = [1000.0]
        monkeypatch.setattr("plugins.platforms.synology_chat.adapter.time.monotonic", lambda: t[0])
        adapter._ip_rate_limited("10.0.0.5")
        assert "10.0.0.5" in adapter._ip_windows
        # A new IP far in the future: the old window is empty but still keyed
        # until its own next call — bounded by LRU cap, never fails open.
        assert adapter._ip_rate_limited("10.0.0.5") is False  # same IP, purges its stale entries


class TestTriggerWordStrip:
    def test_strips_leading_trigger(self):
        s = SynologyChatAdapter._strip_trigger_word
        assert s("Hermes what time", "Hermes") == "what time"
        assert s("hermes: hello", "Hermes") == "hello"
        assert s("hello no trigger", "Hermes") == "hello no trigger"
        assert s("  Hermes,  spaced  ", "Hermes") == "spaced"
        assert s("text", "") == "text"


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------

class TestRouting:
    def test_dm_prefix(self):
        url, extra, err = _make_adapter()._resolve_destination("dm:4")
        assert err is None and "method=chatbot" in url and extra == {"user_ids": [4]}

    def test_channel_prefix(self):
        url, extra, err = _make_adapter()._resolve_destination("ch:42")
        assert err is None and url.endswith("incoming-42") and extra == {}

    def test_bare_id_channel_membership(self):
        # cron / home-channel bare ids resolve by membership
        url, extra, err = _make_adapter()._resolve_destination("devops")
        assert err is None and url.endswith("incoming-devops")

    def test_bare_numeric_id_dm(self):
        url, extra, err = _make_adapter()._resolve_destination("999")
        assert err is None and "method=chatbot" in url and extra == {"user_ids": [999]}

    def test_channel_without_incoming_url_errors(self):
        adapter = _make_adapter(channels={"88": {"token": "x" * 64}})
        url, extra, err = adapter._resolve_destination("ch:88")
        assert url is None and "incoming_url" in err

    def test_non_numeric_dm_errors(self):
        url, extra, err = _make_adapter()._resolve_destination("dm:not-a-number")
        assert url is None and "user_id" in err


class TestOutboundSend:
    def test_send_empty_content_noop(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        result = asyncio.run(adapter.send("dm:4", ""))
        assert result.success is True

    def test_send_routing_error_returns_failure(self):
        adapter = _make_adapter(channels={"88": {"token": "x" * 64}})
        adapter._http_session = MagicMock()
        result = asyncio.run(adapter.send("ch:88", "hi"))
        assert result.success is False and "incoming_url" in result.error

    def test_send_dm_payload_encoding(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = _post_mock(200, "ok")
        result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is True
        body = adapter._http_session.post.call_args.kwargs["data"]
        decoded = json.loads(urllib.parse.unquote(body[len("payload="):]))
        assert decoded == {"text": "hello", "user_ids": [4]}

    def test_send_multichunk(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = _post_mock(200, "ok")
        with patch("asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(adapter.send("dm:4", "x" * 4500))  # > 2*2000
        assert result.success is True
        assert adapter._http_session.post.call_count == 3

    def test_api_error_411_retries(self):
        adapter = _make_adapter()
        responses = [
            json.dumps({"success": False, "error": {"code": 411}}),
            json.dumps({"success": False, "error": {"code": 411}}),
            "ok",
        ]

        def post_side_effect(*a, **k):
            resp = MagicMock()
            resp.status = 200
            resp.text = AsyncMock(return_value=responses.pop(0))
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        adapter._http_session = MagicMock()
        adapter._http_session.post = MagicMock(side_effect=post_side_effect)
        with patch("asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is True
        assert adapter._http_session.post.call_count == 3

    def test_permanent_api_error_no_retry(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = _post_mock(200, json.dumps({"success": False, "error": {"code": 800}}))
        result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is False and "800" in result.error
        assert adapter._http_session.post.call_count == 1

    def test_timeout_no_retry_not_retryable(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = MagicMock(side_effect=asyncio.TimeoutError)
        result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is False
        assert result.retryable is False  # base must NOT re-send (double-delivery)
        assert "timed out" in result.error
        assert adapter._http_session.post.call_count == 1

    def test_http_4xx_direct_failure(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = _post_mock(404, "not found")
        result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is False and adapter._http_session.post.call_count == 1

    def test_http_5xx_retries(self):
        adapter = _make_adapter()
        adapter._http_session = MagicMock()
        adapter._http_session.post = _post_mock(503, "unavailable")
        with patch("asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(adapter.send("dm:4", "hello"))
        assert result.success is False and result.retryable is True
        assert adapter._http_session.post.call_count == 3


class TestFormatMessage:
    def test_strips_markdown(self):
        adapter = _make_adapter()
        assert adapter.format_message("**bold** and *italic* and `code`") == "bold and italic and code"
        assert adapter.format_message("## Title\ntext") == "Title\ntext"
        assert "```" not in adapter.format_message("```python\nx = 1\n```")

    def test_links_to_dsm_syntax(self):
        adapter = _make_adapter()
        assert adapter.format_message("see [docs](https://example.com/d)") == "see <https://example.com/d|docs>"
        out = adapter.format_message("[a](https://x.io) and [b](https://y.io)")
        assert out == "<https://x.io|a> and <https://y.io|b>"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_connect_requires_config(self):
        config = PlatformConfig(enabled=True, token="", extra={})
        adapter = SynologyChatAdapter(config)
        assert asyncio.run(adapter.connect()) is False

    def test_connect_disconnect_roundtrip(self):
        # Mock the aiohttp server so the test opens no real socket (matches the
        # repo convention, e.g. test_sms.py) — keeps CI sandboxes happy.
        async def _roundtrip():
            with patch("aiohttp.web.AppRunner") as mock_runner_cls, \
                 patch("aiohttp.web.TCPSite") as mock_site_cls, \
                 patch("aiohttp.ClientSession", return_value=MagicMock(closed=False, close=AsyncMock())):
                mock_runner_cls.return_value.setup = AsyncMock()
                mock_runner_cls.return_value.cleanup = AsyncMock()
                mock_site_cls.return_value.start = AsyncMock()
                adapter = _make_adapter()
                assert await adapter.connect() is True
                assert adapter._runner is not None
                await adapter.disconnect()
                assert adapter._runner is None
        asyncio.run(_roundtrip())

    def test_get_chat_info(self):
        adapter = _make_adapter()
        assert asyncio.run(adapter.get_chat_info("ch:42"))["type"] == "group"
        assert asyncio.run(adapter.get_chat_info("dm:4"))["type"] == "dm"


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestHooks:
    def test_is_connected_takes_config(self, monkeypatch):
        # Contract: registry calls is_connected(config) with a PlatformConfig.
        monkeypatch.setenv("SYNOLOGY_CHAT_TOKEN", BOT_TOKEN)
        monkeypatch.setenv("SYNOLOGY_CHAT_INCOMING_URL", "https://nas/webapi")
        assert _is_connected(PlatformConfig(enabled=True)) is True
        monkeypatch.delenv("SYNOLOGY_CHAT_TOKEN", raising=False)
        assert _is_connected(PlatformConfig(enabled=True)) is False

    def test_env_enablement_requires_both(self, monkeypatch):
        monkeypatch.delenv("SYNOLOGY_CHAT_TOKEN", raising=False)
        monkeypatch.delenv("SYNOLOGY_CHAT_INCOMING_URL", raising=False)
        assert _env_enablement() is None
        monkeypatch.setenv("SYNOLOGY_CHAT_TOKEN", BOT_TOKEN)
        monkeypatch.setenv("SYNOLOGY_CHAT_INCOMING_URL", "https://nas/webapi")
        assert _env_enablement()["token"] == BOT_TOKEN

    def test_env_enablement_channels_and_home_prefixed(self, monkeypatch):
        monkeypatch.setenv("SYNOLOGY_CHAT_TOKEN", BOT_TOKEN)
        monkeypatch.setenv("SYNOLOGY_CHAT_INCOMING_URL", "https://nas/webapi")
        monkeypatch.setenv("SYNOLOGY_CHANNEL_TOKEN_42", CHAN_A_TOKEN)
        monkeypatch.setenv("SYNOLOGY_CHAT_HOME_CHANNEL", "devops")
        seed = _env_enablement()
        assert seed["channels"]["42"]["token"] == CHAN_A_TOKEN
        assert seed["home_channel"]["chat_id"] == "devops"  # bare id (membership routing)

    def test_apply_yaml_config_env_precedence(self, monkeypatch):
        monkeypatch.setenv("SYNOLOGY_CHAT_WEBHOOK_PORT", "9999")
        _apply_yaml_config({}, {"webhook_port": 8645})
        assert os.environ["SYNOLOGY_CHAT_WEBHOOK_PORT"] == "9999"

    def test_apply_yaml_config_channels_to_extra(self):
        extra = _apply_yaml_config({}, {
            "channels": {"42": {"token": CHAN_A_TOKEN, "incoming_url": "https://nas/i42", "ignored": "x"}},
        })
        assert extra["channels"]["42"] == {"token": CHAN_A_TOKEN, "incoming_url": "https://nas/i42"}

    def test_interactive_setup(self, monkeypatch):
        monkeypatch.delenv("SYNOLOGY_CHAT_TOKEN", raising=False)
        monkeypatch.delenv("SYNOLOGY_CHAT_INCOMING_URL", raising=False)
        saved = {}
        monkeypatch.setattr("hermes_cli.config.save_env_value",
                            lambda k, v: saved.__setitem__(k, v))
        inputs = iter([BOT_TOKEN, "https://nas/webapi-chatbot"])
        monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
        assert interactive_setup() is True
        assert saved["SYNOLOGY_CHAT_TOKEN"] == BOT_TOKEN
        assert saved["SYNOLOGY_CHAT_INCOMING_URL"] == "https://nas/webapi-chatbot"

    def test_interactive_setup_rejects_empty(self, monkeypatch):
        monkeypatch.delenv("SYNOLOGY_CHAT_TOKEN", raising=False)
        monkeypatch.delenv("SYNOLOGY_CHAT_INCOMING_URL", raising=False)
        monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: None)
        monkeypatch.setattr("builtins.input", lambda *a: "")
        assert interactive_setup() is False


class TestStandaloneSend:
    def _patched_session(self, posted, status=200, body="ok"):
        class _Resp:
            def __init__(self):
                self.status = status
            async def text(self):
                return body

        class _CM:
            async def __aenter__(self_):
                return _Resp()
            async def __aexit__(self_, *a):
                return False

        class _Session:
            def __init__(self, *a, **k):
                pass
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *a):
                return False
            def post(self_, url, **kwargs):
                posted["url"] = url
                posted["data"] = kwargs.get("data")
                return _CM()

        return _Session

    def test_standalone_routes_channel(self):
        pconfig = MagicMock()
        pconfig.extra = {
            "incoming_url": "https://nas/chatbot",
            "channels": {"devops": {"incoming_url": "https://nas/incoming-devops"}},
        }
        posted = {}
        with patch("aiohttp.ClientSession", self._patched_session(posted)), \
             patch("aiohttp.TCPConnector", MagicMock()), patch("aiohttp.ClientTimeout", MagicMock()):
            result = asyncio.run(_standalone_send(pconfig, "ch:devops", "hello from cron"))
        assert result.get("success") is True
        assert posted["url"] == "https://nas/incoming-devops"

    def test_standalone_bare_home_channel(self):
        pconfig = MagicMock()
        pconfig.extra = {
            "incoming_url": "https://nas/chatbot",
            "channels": {"devops": {"incoming_url": "https://nas/incoming-devops"}},
        }
        posted = {}
        with patch("aiohttp.ClientSession", self._patched_session(posted)), \
             patch("aiohttp.TCPConnector", MagicMock()), patch("aiohttp.ClientTimeout", MagicMock()):
            result = asyncio.run(_standalone_send(pconfig, "devops", "ping"))
        assert result.get("success") is True
        assert posted["url"] == "https://nas/incoming-devops"

    def test_standalone_dm_user_ids(self):
        pconfig = MagicMock()
        pconfig.extra = {"incoming_url": "https://nas/chatbot"}
        posted = {}
        with patch("aiohttp.ClientSession", self._patched_session(posted)), \
             patch("aiohttp.TCPConnector", MagicMock()), patch("aiohttp.ClientTimeout", MagicMock()):
            result = asyncio.run(_standalone_send(pconfig, "dm:4", "ping"))
        assert result.get("success") is True
        decoded = json.loads(urllib.parse.unquote(posted["data"][len("payload="):]))
        assert decoded["user_ids"] == [4]

    def test_standalone_api_error_surfaced(self):
        pconfig = MagicMock()
        pconfig.extra = {"incoming_url": "https://nas/chatbot"}
        posted = {}
        body = json.dumps({"success": False, "error": {"code": 800}})
        with patch("aiohttp.ClientSession", self._patched_session(posted, 200, body)), \
             patch("aiohttp.TCPConnector", MagicMock()), patch("aiohttp.ClientTimeout", MagicMock()):
            result = asyncio.run(_standalone_send(pconfig, "dm:4", "ping"))
        assert "error" in result and "800" in result["error"]
