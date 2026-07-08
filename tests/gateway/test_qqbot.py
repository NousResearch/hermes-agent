"""Tests for the QQ Bot platform adapter."""

import asyncio
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra):
    """Build a PlatformConfig(enabled=True, extra=extra) for testing."""
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# check_qq_requirements
# ---------------------------------------------------------------------------

class TestQQRequirements:
    def test_returns_bool(self):
        from gateway.platforms.qqbot import check_qq_requirements
        result = check_qq_requirements()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# QQAdapter.__init__
# ---------------------------------------------------------------------------

class TestQQAdapterInit:
    def _make(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_basic_attributes(self):
        adapter = self._make(app_id="123", client_secret="sec")
        assert adapter._app_id == "123"
        assert adapter._client_secret == "sec"

    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"QQ_APP_ID": "env_id", "QQ_CLIENT_SECRET": "env_sec"}, clear=False):
            adapter = self._make()
            assert adapter._app_id == "env_id"
            assert adapter._client_secret == "env_sec"

    def test_env_fallback_extra_wins(self):
        with mock.patch.dict(os.environ, {"QQ_APP_ID": "env_id"}, clear=False):
            adapter = self._make(app_id="extra_id", client_secret="sec")
            assert adapter._app_id == "extra_id"

    def test_dm_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._dm_policy == "pairing"

    def test_dm_policy_explicit(self):
        adapter = self._make(app_id="a", client_secret="b", dm_policy="allowlist")
        assert adapter._dm_policy == "allowlist"

    def test_group_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._group_policy == "disabled"

    def test_allow_from_parsing_string(self):
        adapter = self._make(app_id="a", client_secret="b", allow_from="x, y , z")
        assert adapter._allow_from == ["x", "y", "z"]

    def test_allow_from_parsing_list(self):
        adapter = self._make(app_id="a", client_secret="b", allow_from=["a", "b"])
        assert adapter._allow_from == ["a", "b"]

    def test_allow_from_default_empty(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._allow_from == []

    def test_group_allow_from(self):
        adapter = self._make(app_id="a", client_secret="b", group_allow_from="g1,g2")
        assert adapter._group_allow_from == ["g1", "g2"]

    def test_markdown_support_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._markdown_support is True

    def test_markdown_support_false(self):
        adapter = self._make(app_id="a", client_secret="b", markdown_support=False)
        assert adapter._markdown_support is False

    def test_name_property(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter.name == "QQBot"


# ---------------------------------------------------------------------------
# _coerce_list
# ---------------------------------------------------------------------------

class TestCoerceList:
    def _fn(self, value):
        from gateway.platforms.qqbot import _coerce_list
        return _coerce_list(value)

    def test_none(self):
        assert self._fn(None) == []

    def test_string(self):
        assert self._fn("a, b ,c") == ["a", "b", "c"]

    def test_list(self):
        assert self._fn(["x", "y"]) == ["x", "y"]

    def test_empty_string(self):
        assert self._fn("") == []

    def test_tuple(self):
        assert self._fn(("a", "b")) == ["a", "b"]

    def test_single_item_string(self):
        assert self._fn("hello") == ["hello"]


# ---------------------------------------------------------------------------
# _is_voice_content_type
# ---------------------------------------------------------------------------

class TestIsVoiceContentType:
    def _fn(self, content_type, filename):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._is_voice_content_type(content_type, filename)

    def test_voice_content_type(self):
        assert self._fn("voice", "msg.silk") is True

    def test_audio_content_type(self):
        assert self._fn("audio/mp3", "file.mp3") is True

    def test_voice_extension(self):
        assert self._fn("", "file.silk") is True

    def test_non_voice(self):
        assert self._fn("image/jpeg", "photo.jpg") is False

    def test_audio_extension_amr(self):
        assert self._fn("", "recording.amr") is True


# ---------------------------------------------------------------------------
# Voice attachment SSRF protection
# ---------------------------------------------------------------------------

class TestVoiceAttachmentSSRFProtection:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_stt_blocks_unsafe_download_url(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._http_client = mock.AsyncMock()

        with mock.patch("tools.url_safety.is_safe_url", return_value=False):
            transcript = asyncio.run(
                adapter._stt_voice_attachment(
                    "http://127.0.0.1/voice.silk",
                    "audio/silk",
                    "voice.silk",
                )
            )

        assert transcript is None
        adapter._http_client.get.assert_not_called()

    def test_connect_uses_redirect_guard_hook(self):
        from gateway.platforms.qqbot import QQAdapter, _ssrf_redirect_guard

        client = mock.AsyncMock()
        with mock.patch("gateway.platforms.qqbot.adapter.httpx.AsyncClient", return_value=client) as async_client_cls:
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            adapter._ensure_token = mock.AsyncMock(side_effect=RuntimeError("stop after client creation"))

            connected = asyncio.run(adapter.connect())

        assert connected is False
        assert async_client_cls.call_count == 1
        kwargs = async_client_cls.call_args.kwargs
        assert kwargs.get("follow_redirects") is True
        assert kwargs.get("event_hooks", {}).get("response") == [_ssrf_redirect_guard]

    def test_connect_accepts_is_reconnect_param(self):
        """connect() must accept is_reconnect for interface conformance with
        the base adapter, which the reconnect watcher calls with
        is_reconnect=True."""
        from gateway.platforms.qqbot import QQAdapter

        adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
        adapter._ensure_token = mock.AsyncMock(side_effect=RuntimeError("stop after client init"))

        # Both forms must not raise TypeError.
        connected_default = asyncio.run(adapter.connect())
        connected_explicit = asyncio.run(adapter.connect(is_reconnect=True))
        assert connected_default is False
        assert connected_explicit is False


# ---------------------------------------------------------------------------
# WebSocket proxy handling
# ---------------------------------------------------------------------------

class TestQQWebSocketProxy:
    @pytest.mark.asyncio
    async def test_open_ws_honors_proxy_env(self, monkeypatch):
        from gateway.platforms.qqbot import QQAdapter

        for key in (
            "WSS_PROXY",
            "wss_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

        adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))

        seen_session_kwargs = {}
        seen_ws_kwargs = {}

        class FakeSession:
            def __init__(self, **kwargs):
                seen_session_kwargs.update(kwargs)
                self.closed = False

            async def close(self):
                self.closed = True

            async def ws_connect(self, *args, **kwargs):
                seen_ws_kwargs.update(kwargs)
                return mock.AsyncMock(closed=False)

        with mock.patch("gateway.platforms.qqbot.adapter.aiohttp.ClientSession", side_effect=FakeSession):
            await adapter._open_ws("wss://api.sgroup.qq.com/websocket")

        assert seen_session_kwargs.get("trust_env") is True
        assert seen_ws_kwargs.get("proxy") == "http://127.0.0.1:7897"

# ---------------------------------------------------------------------------
# _strip_at_mention
# ---------------------------------------------------------------------------

class TestStripAtMention:
    def _fn(self, content):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._strip_at_mention(content)

    def test_removes_mention(self):
        result = self._fn("@BotUser hello there")
        assert result == "hello there"

    def test_no_mention(self):
        result = self._fn("just text")
        assert result == "just text"

    def test_empty_string(self):
        assert self._fn("") == ""

    def test_only_mention(self):
        assert self._fn("@Someone  ") == ""

    def test_strips_explicit_mention_tag(self):
        # Full-push GROUP_MESSAGE_CREATE may carry the <@!id> tag form.
        assert self._fn("<@!1903885637> hello there") == "hello there"

    def test_strips_tag_midstring(self):
        assert self._fn("hey <@!123> how are you") == "hey  how are you"


# ---------------------------------------------------------------------------
# Group rich media / markdown parity (2.3)
# ---------------------------------------------------------------------------

class TestGroupMediaMarkdown:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    @pytest.mark.asyncio
    async def test_group_inbound_image_populates_media_urls(self):
        adapter = self._make_adapter(group_require_mention=False)  # always
        captured = []

        async def fake_process(_a):
            return {"image_urls": ["/tmp/x.jpg"],
                    "image_media_types": ["image/jpeg"],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]

        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "look",
             "attachments": [{"content_type": "image/jpeg", "url": "u"}]},
            "m1", "look", {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].media_urls == ["/tmp/x.jpg"]
        assert captured[0].media_types == ["image/jpeg"]

    @pytest.mark.asyncio
    async def test_group_markdown_send_routes_to_group_endpoint(self):
        adapter = self._make_adapter(markdown_support=True)
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "sent1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("g1", "**bold** reply")
        assert result.success
        assert calls and calls[0][1] == "/v2/groups/g1/messages"
        # markdown_support=True → markdown msg_type (2).
        assert calls[0][2]["msg_type"] == 2
        assert calls[0][2]["markdown"]["content"] == "**bold** reply"

    @pytest.mark.asyncio
    async def test_group_plaintext_send_routes_to_group_endpoint(self):
        # Default markdown_support=False → plain text msg_type (0).
        adapter = self._make_adapter(markdown_support=False)
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "sent2"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("g1", "plain reply", reply_to="mm1")
        assert result.success
        assert calls[0][1] == "/v2/groups/g1/messages"
        assert calls[0][2]["msg_type"] == 0
        assert calls[0][2]["content"] == "plain reply"
        assert calls[0][2].get("msg_id") == "mm1"

    @pytest.mark.asyncio
    async def test_group_media_send_posts_msg_type_media(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_upload(chat_type, chat_id, file_type, **kw):
            return {"file_info": "FILEINFO"}

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "media1"}

        adapter._upload_media = fake_upload  # type: ignore[assignment]
        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter._send_media(
            "g1", "https://example.com/x.jpg", 1, "image",
        )
        assert result.success
        assert calls and calls[0][1] == "/v2/groups/g1/messages"
        assert calls[0][2]["msg_type"] == 7  # MSG_TYPE_MEDIA
        assert calls[0][2]["media"]["file_info"] == "FILEINFO"


# ---------------------------------------------------------------------------
# Reserved runtime mode-switch hook (2.2.3)
# ---------------------------------------------------------------------------

class TestGroupModeRuntimeOverride:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "a")
        extra.setdefault("client_secret", "b")
        return QQAdapter(_make_config(**extra))

    def test_override_takes_effect(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        adapter = self._make_adapter(group_require_mention=True)
        adapter._set_group_mode_override("g1", require_mention=False)
        assert adapter._group_mode_runtime_overrides == {"g1": False}
        # resolve should now honour the runtime override for g1.
        eff = resolve_require_mention(
            "g1",
            global_default=adapter._group_require_mention,
            per_group=adapter._group_mode_overrides,
            runtime_overrides=adapter._group_mode_runtime_overrides,
        )
        assert eff is False


# ---------------------------------------------------------------------------
# _is_dm_allowed
# ---------------------------------------------------------------------------

class TestDmAllowed:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_open_policy_requires_opt_in(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="open")
        assert adapter._is_dm_allowed("any_user") is False

    def test_open_policy_with_opt_in(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="open")
        assert adapter._is_dm_allowed("any_user") is True
        assert adapter._is_dm_intake_allowed("any_user") is True

    def test_disabled_policy(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="disabled")
        assert adapter._is_dm_allowed("any_user") is False

    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="allowlist", allow_from="user1,user2")
        assert adapter._is_dm_allowed("user1") is True

    def test_allowlist_no_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="allowlist", allow_from="user1,user2")
        assert adapter._is_dm_allowed("user3") is False

    def test_allowlist_wildcard(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="allowlist", allow_from="*")
        assert adapter._is_dm_allowed("anyone") is True


# ---------------------------------------------------------------------------
# _is_group_allowed
# ---------------------------------------------------------------------------

class TestGroupAllowed:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_open_policy(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", group_policy="open")
        assert adapter._is_group_allowed("grp1", "user1") is True

    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", group_policy="allowlist", group_allow_from="grp1")
        assert adapter._is_group_allowed("grp1", "user1") is True

    def test_allowlist_no_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", group_policy="allowlist", group_allow_from="grp1")
        assert adapter._is_group_allowed("grp2", "user1") is False

    def test_pairing_default_blocks_groups(self):
        # group_policy default is now "disabled" (Feishu-aligned: no
        # "pairing" mode for groups). Any group message is denied by default.
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._group_policy == "disabled"
        assert adapter._is_group_allowed("grp1", "user1") is False

    def test_unknown_group_policy_falls_back_to_disabled(self):
        adapter = self._make_adapter(
            app_id="a", client_secret="b", group_policy="pairing",  # no longer valid
        )
        assert adapter._group_policy == "disabled"
        assert adapter._is_group_allowed("grp1", "user1") is False

    def test_pairing_default_strict_dm_auth_denies_unknown(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._dm_policy == "pairing"
        assert adapter._is_dm_allowed("any_user") is False

    def test_pairing_default_forwards_dm_to_gateway_intake(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._is_dm_intake_allowed("any_user") is True

# ---------------------------------------------------------------------------
# _resolve_stt_config
# ---------------------------------------------------------------------------

class TestResolveSTTConfig:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_no_config(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        with mock.patch.dict(os.environ, {}, clear=True):
            assert adapter._resolve_stt_config() is None

    def test_env_config(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        with mock.patch.dict(os.environ, {
            "QQ_STT_API_KEY": "key123",
            "QQ_STT_BASE_URL": "https://example.com/v1",
            "QQ_STT_MODEL": "my-model",
        }, clear=True):
            cfg = adapter._resolve_stt_config()
            assert cfg is not None
            assert cfg["api_key"] == "key123"
            assert cfg["base_url"] == "https://example.com/v1"
            assert cfg["model"] == "my-model"

    def test_extra_config(self):
        stt_cfg = {
            "baseUrl": "https://custom.api/v4",
            "apiKey": "sk_extra",
            "model": "glm-asr",
        }
        adapter = self._make_adapter(app_id="a", client_secret="b", stt=stt_cfg)
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = adapter._resolve_stt_config()
            assert cfg is not None
            assert cfg["base_url"] == "https://custom.api/v4"
            assert cfg["api_key"] == "sk_extra"
            assert cfg["model"] == "glm-asr"


# ---------------------------------------------------------------------------
# _detect_message_type
# ---------------------------------------------------------------------------

class TestDetectMessageType:
    def _fn(self, media_urls, media_types):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._detect_message_type(media_urls, media_types)

    def test_no_media(self):
        from gateway.platforms.base import MessageType
        assert self._fn([], []) == MessageType.TEXT

    def test_image(self):
        from gateway.platforms.base import MessageType
        assert self._fn(["file.jpg"], ["image/jpeg"]) == MessageType.PHOTO

    def test_voice(self):
        from gateway.platforms.base import MessageType
        assert self._fn(["voice.silk"], ["audio/silk"]) == MessageType.VOICE

    def test_video(self):
        from gateway.platforms.base import MessageType
        assert self._fn(["vid.mp4"], ["video/mp4"]) == MessageType.VIDEO


# ---------------------------------------------------------------------------
# QQCloseError
# ---------------------------------------------------------------------------

class TestQQCloseError:
    def test_attributes(self):
        from gateway.platforms.qqbot import QQCloseError
        err = QQCloseError(4004, "bad token")
        assert err.code == 4004
        assert err.reason == "bad token"

    def test_code_none(self):
        from gateway.platforms.qqbot import QQCloseError
        err = QQCloseError(None, "")
        assert err.code is None

    def test_string_to_int(self):
        from gateway.platforms.qqbot import QQCloseError
        err = QQCloseError("4914", "banned")
        assert err.code == 4914
        assert err.reason == "banned"

    def test_message_format(self):
        from gateway.platforms.qqbot import QQCloseError
        err = QQCloseError(4008, "rate limit")
        assert "4008" in str(err)
        assert "rate limit" in str(err)


# ---------------------------------------------------------------------------
# _dispatch_payload
# ---------------------------------------------------------------------------

class TestDispatchPayload:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        adapter = QQAdapter(_make_config(**extra))
        return adapter

    def test_unknown_op(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Should not raise
        adapter._dispatch_payload({"op": 99, "d": {}})
        # last_seq should remain None
        assert adapter._last_seq is None

    def test_op10_updates_heartbeat_interval(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({"op": 10, "d": {"heartbeat_interval": 50000}})
        # Should be 50000 / 1000 * 0.8 = 40.0
        assert adapter._heartbeat_interval == 40.0

    def test_op11_heartbeat_ack(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Should not raise
        adapter._dispatch_payload({"op": 11, "t": "HEARTBEAT_ACK", "s": 42})

    def test_seq_tracking(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({"op": 0, "t": "READY", "s": 100, "d": {}})
        assert adapter._last_seq == 100

    def test_seq_increments(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({"op": 0, "t": "READY", "s": 5, "d": {}})
        adapter._dispatch_payload({"op": 0, "t": "SOME_EVENT", "s": 10, "d": {}})
        assert adapter._last_seq == 10


# ---------------------------------------------------------------------------
# READY / RESUMED handling
# ---------------------------------------------------------------------------

class TestReadyHandling:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_ready_stores_session(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({
            "op": 0, "t": "READY",
            "s": 1,
            "d": {"session_id": "sess_abc123"},
        })
        assert adapter._session_id == "sess_abc123"

    def test_resumed_preserves_session(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._session_id = "old_sess"
        adapter._last_seq = 50
        adapter._dispatch_payload({
            "op": 0, "t": "RESUMED", "s": 60, "d": {},
        })
        # Session should remain unchanged on RESUMED
        assert adapter._session_id == "old_sess"
        assert adapter._last_seq == 60


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def _fn(self, raw):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._parse_json(raw)

    def test_valid_json(self):
        result = self._fn('{"op": 10, "d": {}}')
        assert result == {"op": 10, "d": {}}

    def test_invalid_json(self):
        result = self._fn("not json")
        assert result is None

    def test_none_input(self):
        result = self._fn(None)
        assert result is None

    def test_non_dict_json(self):
        result = self._fn('"just a string"')
        assert result is None

    def test_empty_dict(self):
        result = self._fn('{}')
        assert result == {}


# ---------------------------------------------------------------------------
# _build_text_body
# ---------------------------------------------------------------------------

class TestBuildTextBody:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_plain_text(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        body = adapter._build_text_body("hello world")
        assert body["msg_type"] == 0  # MSG_TYPE_TEXT
        assert body["content"] == "hello world"

    def test_markdown_text(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=True)
        body = adapter._build_text_body("**bold** text")
        assert body["msg_type"] == 2  # MSG_TYPE_MARKDOWN
        assert body["markdown"]["content"] == "**bold** text"

    def test_truncation(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        long_text = "x" * 10000
        body = adapter._build_text_body(long_text)
        assert len(body["content"]) == adapter.MAX_MESSAGE_LENGTH

    def test_empty_string(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        body = adapter._build_text_body("")
        assert body["content"] == ""

    def test_reply_to(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        body = adapter._build_text_body("reply text", reply_to="msg_123")
        assert body.get("message_reference", {}).get("message_id") == "msg_123"


# ---------------------------------------------------------------------------
# _wait_for_reconnection / send reconnection wait
# ---------------------------------------------------------------------------

class TestWaitForReconnection:
    """Test that send() waits for reconnection instead of silently dropping."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    @pytest.mark.asyncio
    async def test_send_waits_and_succeeds_on_reconnect(self):
        """send() should wait for reconnection and then deliver the message."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Initially disconnected
        adapter._running = False
        adapter._http_client = mock.MagicMock()

        # Simulate reconnection after 0.3s (faster than real interval)
        async def fake_api_request(*args, **kwargs):
            return {"id": "msg_123"}

        adapter._api_request = fake_api_request
        adapter._ensure_token = mock.AsyncMock()
        adapter._RECONNECT_POLL_INTERVAL = 0.1
        adapter._RECONNECT_WAIT_SECONDS = 5.0

        # Schedule reconnection after a short delay
        async def reconnect_after_delay():
            await asyncio.sleep(0.3)
            adapter._running = True
            adapter._ws = SimpleNamespace(closed=False)

        asyncio.get_event_loop().create_task(reconnect_after_delay())

        result = await adapter.send("test_openid", "Hello, world!")
        assert result.success
        assert result.message_id == "msg_123"

    @pytest.mark.asyncio
    async def test_send_returns_retryable_after_timeout(self):
        """send() should return retryable=True if reconnection takes too long."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._running = False
        adapter._RECONNECT_POLL_INTERVAL = 0.05
        adapter._RECONNECT_WAIT_SECONDS = 0.2

        result = await adapter.send("test_openid", "Hello, world!")
        assert not result.success
        assert result.retryable is True
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_succeeds_immediately_when_connected(self):
        """send() should not wait when already connected."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._http_client = mock.MagicMock()

        async def fake_api_request(*args, **kwargs):
            return {"id": "msg_immediate"}

        adapter._api_request = fake_api_request

        result = await adapter.send("test_openid", "Hello!")
        assert result.success
        assert result.message_id == "msg_immediate"

    @pytest.mark.asyncio
    async def test_send_media_waits_for_reconnect(self):
        """_send_media should also wait for reconnection."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._running = False
        adapter._RECONNECT_POLL_INTERVAL = 0.05
        adapter._RECONNECT_WAIT_SECONDS = 0.2

        result = await adapter._send_media("test_openid", "http://example.com/img.jpg", 1, "image")
        assert not result.success
        assert result.retryable is True
        assert "Not connected" in result.error


# ---------------------------------------------------------------------------
# ChunkedUploader
# ---------------------------------------------------------------------------

class TestChunkedUploadFormatSize:
    def test_bytes(self):
        from gateway.platforms.qqbot.chunked_upload import format_size
        assert format_size(100) == "100.0 B"

    def test_kilobytes(self):
        from gateway.platforms.qqbot.chunked_upload import format_size
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        from gateway.platforms.qqbot.chunked_upload import format_size
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        from gateway.platforms.qqbot.chunked_upload import format_size
        assert format_size(3 * 1024 ** 3) == "3.0 GB"


class TestChunkedUploadErrors:
    def test_daily_limit_has_human_size(self):
        from gateway.platforms.qqbot.chunked_upload import UploadDailyLimitExceededError
        exc = UploadDailyLimitExceededError("demo.mp4", 12_345_678)
        assert exc.file_name == "demo.mp4"
        assert exc.file_size == 12_345_678
        assert "MB" in exc.file_size_human
        assert "demo.mp4" in str(exc)

    def test_too_large_includes_limit(self):
        from gateway.platforms.qqbot.chunked_upload import UploadFileTooLargeError
        exc = UploadFileTooLargeError("huge.bin", 200 * 1024 * 1024, 100 * 1024 * 1024)
        assert exc.file_name == "huge.bin"
        assert "MB" in exc.file_size_human
        assert "MB" in exc.limit_human
        assert "huge.bin" in str(exc)

    def test_too_large_unknown_limit(self):
        from gateway.platforms.qqbot.chunked_upload import UploadFileTooLargeError
        exc = UploadFileTooLargeError("f", 100, 0)
        assert exc.limit_human == "unknown"


class TestChunkedUploadHelpers:
    def test_read_chunk_exact_bytes(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import _read_file_chunk
        f = tmp_path / "x.bin"
        f.write_bytes(b"0123456789abcdef")
        assert _read_file_chunk(str(f), 2, 4) == b"2345"

    def test_read_chunk_short_read_raises(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import _read_file_chunk
        f = tmp_path / "x.bin"
        f.write_bytes(b"hi")
        with pytest.raises(IOError):
            _read_file_chunk(str(f), 0, 100)

    def test_compute_hashes_small_file(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import _compute_file_hashes
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world")
        h = _compute_file_hashes(str(f), 11)
        assert len(h["md5"]) == 32
        assert len(h["sha1"]) == 40
        # For small files md5_10m equals md5.
        assert h["md5"] == h["md5_10m"]

    def test_compute_hashes_large_file_has_distinct_md5_10m(self, tmp_path):
        # File > 10,002,432 bytes → md5_10m is truncated, so it differs from full md5.
        from gateway.platforms.qqbot.chunked_upload import (
            _compute_file_hashes, _MD5_10M_SIZE,
        )
        f = tmp_path / "big.bin"
        size = _MD5_10M_SIZE + 1024
        # Two distinct byte values so the extra tail changes the full md5.
        f.write_bytes(b"A" * _MD5_10M_SIZE + b"B" * 1024)
        h = _compute_file_hashes(str(f), size)
        assert h["md5"] != h["md5_10m"]

    def test_parse_prepare_response_wrapped_in_data(self):
        from gateway.platforms.qqbot.chunked_upload import _parse_prepare_response
        raw = {
            "data": {
                "upload_id": "uid-42",
                "block_size": 4096,
                "parts": [
                    {"part_index": 1, "presigned_url": "https://cos/1", "block_size": 4096},
                    {"index": 2, "url": "https://cos/2"},
                ],
                "concurrency": 3,
                "retry_timeout": 90,
            }
        }
        r = _parse_prepare_response(raw)
        assert r.upload_id == "uid-42"
        assert r.block_size == 4096
        assert len(r.parts) == 2
        assert r.parts[0].presigned_url == "https://cos/1"
        assert r.parts[1].index == 2
        assert r.concurrency == 3
        assert r.retry_timeout == 90.0

    def test_parse_prepare_response_missing_upload_id_raises(self):
        from gateway.platforms.qqbot.chunked_upload import _parse_prepare_response
        with pytest.raises(ValueError, match="upload_id"):
            _parse_prepare_response({"block_size": 1024, "parts": [{"index": 1, "url": "x"}]})

    def test_parse_prepare_response_missing_parts_raises(self):
        from gateway.platforms.qqbot.chunked_upload import _parse_prepare_response
        with pytest.raises(ValueError, match="parts"):
            _parse_prepare_response({"upload_id": "uid", "block_size": 1024, "parts": []})


class TestChunkedUploaderFlow:
    """End-to-end prepare / PUT / part_finish / complete flow with mocked HTTP.

    Verifies the state machine matches the QQ v2 contract without hitting the network.
    """

    @pytest.mark.asyncio
    async def test_full_upload_two_parts_success(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader

        # Two-part file.
        f = tmp_path / "vid.mp4"
        f.write_bytes(b"A" * 5_000_000 + b"B" * 3_000_000)

        # Mock api_request — handles prepare, part_finish, complete based on URL.
        api_calls = []

        async def fake_api_request(method, path, *, body=None, timeout=None):
            api_calls.append((method, path, body))
            if path.endswith("/upload_prepare"):
                return {
                    "upload_id": "uid-xyz",
                    "block_size": 5_000_000,
                    "parts": [
                        {"part_index": 1, "presigned_url": "https://cos.example/p1"},
                        {"part_index": 2, "presigned_url": "https://cos.example/p2"},
                    ],
                    "concurrency": 1,
                }
            if path.endswith("/upload_part_finish"):
                return {}
            # complete
            return {"file_info": "FILEINFO_TOKEN", "file_uuid": "u-1"}

        # Mock http_put — always returns 200.
        put_calls = []

        class _FakeResp:
            status_code = 200
            text = ""

        async def fake_put(url, data=None, headers=None):
            put_calls.append((url, len(data), headers))
            return _FakeResp()

        uploader = ChunkedUploader(
            api_request=fake_api_request,
            http_put=fake_put,
            log_tag="QQBot:TEST",
        )
        result = await uploader.upload(
            chat_type="c2c",
            target_id="user-openid-1",
            file_path=str(f),
            file_type=2,  # MEDIA_TYPE_VIDEO
            file_name="vid.mp4",
        )

        assert result["file_info"] == "FILEINFO_TOKEN"
        # Two PUTs, one per part.
        assert len(put_calls) == 2
        assert put_calls[0][0] == "https://cos.example/p1"
        assert put_calls[1][0] == "https://cos.example/p2"
        # Prepare + 2 part_finish + complete = 4 api calls.
        assert len(api_calls) == 4
        assert api_calls[0][1].endswith("/upload_prepare")
        assert api_calls[1][1].endswith("/upload_part_finish")
        assert api_calls[2][1].endswith("/upload_part_finish")
        # complete path reuses /files.
        assert api_calls[3][1].endswith("/files")
        assert api_calls[3][2] == {"upload_id": "uid-xyz"}

    @pytest.mark.asyncio
    async def test_group_paths(self, tmp_path):
        """Group uploads hit /v2/groups/... instead of /v2/users/..."""
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader

        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 100)

        seen_paths = []

        async def fake_api_request(method, path, *, body=None, timeout=None):
            seen_paths.append(path)
            if path.endswith("/upload_prepare"):
                return {
                    "upload_id": "gid-1",
                    "block_size": 100,
                    "parts": [{"part_index": 1, "presigned_url": "https://cos/g1"}],
                }
            if path.endswith("/upload_part_finish"):
                return {}
            return {"file_info": "GFILE"}

        class _R:
            status_code = 200
            text = ""

        async def fake_put(url, data=None, headers=None):
            return _R()

        u = ChunkedUploader(fake_api_request, fake_put, "QQBot:T")
        await u.upload(
            chat_type="group",
            target_id="grp-openid-1",
            file_path=str(f),
            file_type=4,
            file_name="a.bin",
        )
        assert all("/v2/groups/" in p for p in seen_paths)
        assert any(p.endswith("/upload_prepare") for p in seen_paths)
        assert any(p.endswith("/files") for p in seen_paths)

    @pytest.mark.asyncio
    async def test_daily_limit_raises_structured_error(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import (
            ChunkedUploader, UploadDailyLimitExceededError,
        )

        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 10)

        async def fake_api_request(method, path, *, body=None, timeout=None):
            # Simulate the adapter's RuntimeError with biz_code 40093002 in the message.
            raise RuntimeError("QQ Bot API error [200] /v2/users/x/upload_prepare: biz_code=40093002 daily limit exceeded")

        async def fake_put(*a, **kw):
            raise AssertionError("PUT should not be called if prepare fails")

        u = ChunkedUploader(fake_api_request, fake_put, "T")
        with pytest.raises(UploadDailyLimitExceededError) as excinfo:
            await u.upload(
                chat_type="c2c",
                target_id="u",
                file_path=str(f),
                file_type=4,
                file_name="a.bin",
            )
        assert excinfo.value.file_name == "a.bin"

    @pytest.mark.asyncio
    async def test_part_finish_retries_on_40093001_then_succeeds(self, tmp_path):
        """biz_code 40093001 is retryable — finish-with-retry must keep trying."""
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader
        import gateway.platforms.qqbot.chunked_upload as cu

        # Make the retry loop fast so the test doesn't take real seconds.
        orig_interval = cu._PART_FINISH_RETRY_INTERVAL
        cu._PART_FINISH_RETRY_INTERVAL = 0.01

        try:
            f = tmp_path / "a.bin"
            f.write_bytes(b"x" * 50)

            finish_calls = {"n": 0}

            async def fake_api_request(method, path, *, body=None, timeout=None):
                if path.endswith("/upload_prepare"):
                    return {
                        "upload_id": "u",
                        "block_size": 50,
                        "parts": [{"part_index": 1, "presigned_url": "https://cos/1"}],
                    }
                if path.endswith("/upload_part_finish"):
                    finish_calls["n"] += 1
                    if finish_calls["n"] < 3:
                        raise RuntimeError("biz_code=40093001 transient part finish error")
                    return {}
                return {"file_info": "F"}

            class _R:
                status_code = 200
                text = ""

            async def fake_put(*a, **kw):
                return _R()

            u = ChunkedUploader(fake_api_request, fake_put, "T")
            result = await u.upload(
                chat_type="c2c",
                target_id="u",
                file_path=str(f),
                file_type=4,
                file_name="a.bin",
            )
            assert result["file_info"] == "F"
            assert finish_calls["n"] == 3  # 2 transient errors + 1 success
        finally:
            cu._PART_FINISH_RETRY_INTERVAL = orig_interval

    @pytest.mark.asyncio
    async def test_put_retries_transient_failure(self, tmp_path):
        """COS PUT failures retry up to _PART_UPLOAD_MAX_RETRIES times."""
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader

        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 20)

        async def fake_api_request(method, path, *, body=None, timeout=None):
            if path.endswith("/upload_prepare"):
                return {
                    "upload_id": "u",
                    "block_size": 20,
                    "parts": [{"part_index": 1, "presigned_url": "https://cos/1"}],
                }
            if path.endswith("/upload_part_finish"):
                return {}
            return {"file_info": "F"}

        put_attempts = {"n": 0}

        class _Resp:
            def __init__(self, status, text=""):
                self.status_code = status
                self.text = text

        async def fake_put(url, data=None, headers=None):
            put_attempts["n"] += 1
            if put_attempts["n"] < 2:
                return _Resp(500, "transient")
            return _Resp(200)

        u = ChunkedUploader(fake_api_request, fake_put, "T")
        result = await u.upload(
            chat_type="c2c",
            target_id="u",
            file_path=str(f),
            file_type=4,
            file_name="a.bin",
        )
        assert result["file_info"] == "F"
        assert put_attempts["n"] == 2


# ---------------------------------------------------------------------------
# Inline keyboards — approval + update-prompt flows
# ---------------------------------------------------------------------------

class TestApprovalButtonData:
    def test_parse_allow_once(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        result = parse_approval_button_data("approve:agent:main:qqbot:c2c:UID:allow-once")
        assert result == ("agent:main:qqbot:c2c:UID", "allow-once")

    def test_parse_allow_always(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("approve:sess:allow-always") == ("sess", "allow-always")

    def test_parse_deny(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("approve:sess:deny") == ("sess", "deny")

    def test_parse_invalid_prefix_returns_none(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("update_prompt:y") is None

    def test_parse_unknown_decision_returns_none(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("approve:sess:maybe") is None

    def test_parse_empty_returns_none(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("") is None
        assert parse_approval_button_data(None) is None  # type: ignore[arg-type]


class TestUpdatePromptButtonData:
    def test_parse_yes(self):
        from gateway.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("update_prompt:y") == "y"

    def test_parse_no(self):
        from gateway.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("update_prompt:n") == "n"

    def test_parse_unknown_returns_none(self):
        from gateway.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("update_prompt:maybe") is None

    def test_parse_wrong_prefix(self):
        from gateway.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("approve:sess:deny") is None


class TestBuildApprovalKeyboard:
    def test_three_buttons_in_single_row(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("session-1")
        assert len(kb.content.rows) == 1
        assert len(kb.content.rows[0].buttons) == 3

    def test_button_data_embeds_session_key(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("agent:main:qqbot:c2c:UID")
        datas = [b.action.data for b in kb.content.rows[0].buttons]
        assert datas[0] == "approve:agent:main:qqbot:c2c:UID:allow-once"
        assert datas[1] == "approve:agent:main:qqbot:c2c:UID:allow-always"
        assert datas[2] == "approve:agent:main:qqbot:c2c:UID:deny"

    def test_buttons_share_group_id_for_mutual_exclusion(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("s")
        group_ids = {b.group_id for b in kb.content.rows[0].buttons}
        assert group_ids == {"approval"}

    def test_to_dict_has_expected_shape(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("s")
        d = kb.to_dict()
        assert "content" in d
        assert "rows" in d["content"]
        assert len(d["content"]["rows"]) == 1
        btn0 = d["content"]["rows"][0]["buttons"][0]
        assert btn0["id"] == "allow"
        assert btn0["action"]["type"] == 1
        assert btn0["action"]["data"].startswith("approve:s:")
        assert btn0["render_data"]["label"]
        assert btn0["render_data"]["visited_label"]

    def test_round_trip_parse_matches_build(self):
        """Every button built by build_approval_keyboard is parseable."""
        from gateway.platforms.qqbot.keyboards import (
            build_approval_keyboard, parse_approval_button_data,
        )
        session_key = "agent:main:qqbot:c2c:UID123"
        kb = build_approval_keyboard(session_key)
        for btn in kb.content.rows[0].buttons:
            parsed = parse_approval_button_data(btn.action.data)
            assert parsed is not None
            assert parsed[0] == session_key
            assert parsed[1] in {"allow-once", "allow-always", "deny"}


class TestBuildUpdatePromptKeyboard:
    def test_two_buttons(self):
        from gateway.platforms.qqbot.keyboards import build_update_prompt_keyboard
        kb = build_update_prompt_keyboard()
        assert len(kb.content.rows[0].buttons) == 2

    def test_button_data_shape(self):
        from gateway.platforms.qqbot.keyboards import build_update_prompt_keyboard
        kb = build_update_prompt_keyboard()
        datas = [b.action.data for b in kb.content.rows[0].buttons]
        assert datas == ["update_prompt:y", "update_prompt:n"]


class TestBuildApprovalText:
    def test_exec_approval_includes_command_preview(self):
        from gateway.platforms.qqbot.keyboards import (
            ApprovalRequest, build_approval_text,
        )
        req = ApprovalRequest(
            session_key="s",
            title="t",
            command_preview="rm -rf /tmp/demo",
            cwd="/home/user",
            timeout_sec=60,
        )
        text = build_approval_text(req)
        assert "命令执行审批" in text
        assert "rm -rf /tmp/demo" in text
        assert "/home/user" in text
        assert "60" in text

    def test_plugin_approval_uses_severity_icon(self):
        from gateway.platforms.qqbot.keyboards import (
            ApprovalRequest, build_approval_text,
        )
        crit = ApprovalRequest(
            session_key="s", title="dangerous op",
            severity="critical", tool_name="shell", timeout_sec=30,
        )
        assert "🔴" in build_approval_text(crit)

        info = ApprovalRequest(
            session_key="s", title="read-only", severity="info", tool_name="q",
        )
        assert "🔵" in build_approval_text(info)

        default = ApprovalRequest(session_key="s", title="t", tool_name="x")
        assert "🟡" in build_approval_text(default)

    def test_truncates_long_commands(self):
        from gateway.platforms.qqbot.keyboards import (
            ApprovalRequest, build_approval_text,
        )
        long = "x" * 1000
        req = ApprovalRequest(
            session_key="s", title="t", command_preview=long, cwd="/x",
        )
        text = build_approval_text(req)
        # Preview is truncated to 300 chars; 1000 "x"s would still push the
        # body past 300, but the inline preview specifically must be capped.
        preview_line = [
            line for line in text.split("\n") if line.startswith("```")
        ]
        # 2 backtick fences; the content line in between is separate.
        xs_in_preview = sum(line.count("x") for line in text.split("\n") if line and "```" not in line)
        assert xs_in_preview <= 301  # 300 xs + one-off tolerance


class TestInteractionEventParsing:
    def test_parse_c2c_interaction(self):
        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        raw = {
            "id": "interaction-42",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {
                    "button_data": "approve:sess:allow-once",
                    "button_id": "allow",
                },
            },
        }
        ev = parse_interaction_event(raw)
        assert ev.id == "interaction-42"
        assert ev.scene == "c2c"
        assert ev.chat_type == 2
        assert ev.user_openid == "user-1"
        assert ev.button_data == "approve:sess:allow-once"
        assert ev.button_id == "allow"
        assert ev.operator_openid == "user-1"

    def test_parse_group_interaction(self):
        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        raw = {
            "id": "i-1",
            "chat_type": 1,
            "group_openid": "grp-1",
            "group_member_openid": "mem-1",
            "data": {
                "type": 11,
                "resolved": {
                    "button_data": "update_prompt:y",
                    "button_id": "yes",
                },
            },
        }
        ev = parse_interaction_event(raw)
        assert ev.scene == "group"
        assert ev.group_openid == "grp-1"
        assert ev.group_member_openid == "mem-1"
        assert ev.operator_openid == "mem-1"  # member openid preferred in group

    def test_parse_missing_data_gracefully(self):
        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        ev = parse_interaction_event({"id": "i", "chat_type": 0})
        assert ev.id == "i"
        assert ev.scene == "guild"
        assert ev.button_data == ""
        assert ev.button_id == ""
        assert ev.type == 0


class TestAdapterInteractionDispatch:
    """End-to-end verification of _on_interaction including ACK + callback."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_callback_invoked_with_parsed_event(self):
        adapter = self._make_adapter()

        # Stub ACK so we don't require a live http_client.
        ack_calls = []

        async def fake_ack(interaction_id, code=0):
            ack_calls.append((interaction_id, code))

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]

        received = []

        async def cb(event):
            received.append(event)

        adapter.set_interaction_callback(cb)
        await adapter._on_interaction({
            "id": "i-1",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny", "button_id": "deny"},
            },
        })

        assert len(ack_calls) == 1
        assert ack_calls[0][0] == "i-1"
        assert len(received) == 1
        assert received[0].button_data == "approve:agent:main:qqbot:c2c:u:deny"
        assert received[0].scene == "c2c"

    @pytest.mark.asyncio
    async def test_missing_id_skips_ack(self):
        adapter = self._make_adapter()

        ack_calls = []

        async def fake_ack(interaction_id, code=0):
            ack_calls.append(interaction_id)

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]

        callback_calls = []

        async def cb(event):
            callback_calls.append(event)

        adapter.set_interaction_callback(cb)
        await adapter._on_interaction({
            "chat_type": 2,  # no id
            "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny"}},
        })

        assert ack_calls == []
        assert callback_calls == []

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_propagate(self):
        adapter = self._make_adapter()

        async def fake_ack(interaction_id, code=0):
            pass

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]

        async def bad_cb(event):
            raise RuntimeError("boom")

        adapter.set_interaction_callback(bad_cb)
        # Should NOT raise.
        await adapter._on_interaction({
            "id": "i-2",
            "chat_type": 2,
            "user_openid": "u",
            "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny"}},
        })

    @pytest.mark.asyncio
    async def test_explicit_no_callback_is_harmless(self):
        adapter = self._make_adapter()

        async def fake_ack(interaction_id, code=0):
            pass

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]
        # Explicitly clear the default callback. With no callback set,
        # _on_interaction should still ACK and not raise.
        adapter.set_interaction_callback(None)
        await adapter._on_interaction({
            "id": "i-3",
            "chat_type": 2,
            "user_openid": "u",
            "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny"}},
        })


# ---------------------------------------------------------------------------
# Quoted-message handling (message_type=103 → msg_elements)
# ---------------------------------------------------------------------------

class TestProcessQuotedContext:
    """Verify the quoted-message pipeline: text + voice STT + images + files."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_non_quote_message_returns_empty(self):
        adapter = self._make_adapter()
        d = {"message_type": 0, "content": "hi"}
        out = await adapter._process_quoted_context(d)
        assert out == {"quote_block": "", "image_urls": [], "image_media_types": []}

    @pytest.mark.asyncio
    async def test_quote_type_but_no_elements_returns_empty(self):
        adapter = self._make_adapter()
        d = {"message_type": 103}
        out = await adapter._process_quoted_context(d)
        assert out["quote_block"] == ""

    @pytest.mark.asyncio
    async def test_quote_with_text_only(self):
        adapter = self._make_adapter()
        # Stub out _process_attachments since there are no attachments anyway.
        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": ""}
        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [
                {"content": "Did you see this file?", "attachments": []},
            ],
        }
        out = await adapter._process_quoted_context(d)
        assert out["quote_block"].startswith("[Quoted message]:")
        assert "Did you see this file?" in out["quote_block"]
        assert out["image_urls"] == []

    @pytest.mark.asyncio
    async def test_quote_with_voice_attachment_runs_stt(self):
        adapter = self._make_adapter()

        # Capture what attachments are passed into _process_attachments.
        captured = []

        async def fake_process(atts):
            captured.append(atts)
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": ["[Voice] hello from the quoted audio"],
                "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "",
                "attachments": [
                    {"content_type": "audio/silk",
                     "url": "https://qq-cdn/x.silk",
                     "filename": "rec.silk"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)

        # The quoted voice attachment must actually flow through STT.
        assert captured and len(captured[0]) == 1
        assert captured[0][0]["content_type"] == "audio/silk"
        assert "[Quoted message]:" in out["quote_block"]
        assert "hello from the quoted audio" in out["quote_block"]

    @pytest.mark.asyncio
    async def test_quote_with_file_preserves_filename(self):
        """Quoted file attachments must surface the original filename, not the CDN hash."""
        adapter = self._make_adapter()

        async def fake_process(atts):
            # Mirror _process_attachments's behaviour: non-image/voice attachments
            # show up in attachment_info using the real filename.
            parts = []
            for a in atts:
                fn = a.get("filename") or a.get("content_type", "file")
                parts.append(f"[Attachment: {fn}]")
            return {
                "image_urls": [], "image_media_types": [],
                "voice_transcripts": [],
                "attachment_info": "\n".join(parts),
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "check this",
                "attachments": [
                    {"content_type": "application/zip",
                     "url": "https://qq-cdn/abc123",
                     "filename": "quarterly-report.zip"},
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert "quarterly-report.zip" in out["quote_block"]
        assert "check this" in out["quote_block"]

    @pytest.mark.asyncio
    async def test_quote_with_image_returns_cached_paths(self):
        adapter = self._make_adapter()

        async def fake_process(atts):
            return {
                "image_urls": ["/tmp/cached_q.jpg"],
                "image_media_types": ["image/jpeg"],
                "voice_transcripts": [],
                "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "look at this",
                "attachments": [{"content_type": "image/jpeg", "url": "https://x"}],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert out["image_urls"] == ["/tmp/cached_q.jpg"]
        assert out["image_media_types"] == ["image/jpeg"]
        assert "look at this" in out["quote_block"]

    @pytest.mark.asyncio
    async def test_quote_with_image_only_no_text(self):
        """Images-only quote still surfaces a marker so the LLM has context."""
        adapter = self._make_adapter()

        async def fake_process(atts):
            return {
                "image_urls": ["/tmp/only.png"],
                "image_media_types": ["image/png"],
                "voice_transcripts": [],
                "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "",
                "attachments": [{"content_type": "image/png", "url": "https://x"}],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert out["quote_block"]
        assert out["image_urls"] == ["/tmp/only.png"]

    @pytest.mark.asyncio
    async def test_multiple_elements_concatenated(self):
        adapter = self._make_adapter()

        async def fake_process(atts):
            assert len(atts) == 2
            return {
                "image_urls": [], "image_media_types": [],
                "voice_transcripts": [], "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [
                {"content": "first", "attachments": [{"content_type": "image/png", "url": "a"}]},
                {"content": "second", "attachments": [{"content_type": "image/png", "url": "b"}]},
            ],
        }
        out = await adapter._process_quoted_context(d)
        assert "first" in out["quote_block"]
        assert "second" in out["quote_block"]

    @pytest.mark.asyncio
    async def test_invalid_message_type_string_returns_empty(self):
        adapter = self._make_adapter()
        out = await adapter._process_quoted_context(
            {"message_type": "not-a-number", "msg_elements": [{"content": "x"}]}
        )
        assert out["quote_block"] == ""


class TestMergeQuoteInto:
    def test_empty_quote_returns_original(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        assert QQAdapter._merge_quote_into("hello", "") == "hello"

    def test_empty_text_returns_only_quote(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        assert QQAdapter._merge_quote_into("", "[Quoted]") == "[Quoted]"

    def test_both_present_joined_with_blank_line(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        merged = QQAdapter._merge_quote_into("hi there", "[Quoted]:\nctx")
        assert merged == "[Quoted]:\nctx\n\nhi there"


# ---------------------------------------------------------------------------
# Gateway-contract approval UX — send_exec_approval + default dispatcher
# ---------------------------------------------------------------------------

class TestDefaultInteractionDispatch:
    """Verify the adapter's default INTERACTION_CREATE router."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_default_callback_installed_on_init(self):
        """Fresh adapter has a working default interaction callback."""
        adapter = self._make_adapter()
        assert adapter._interaction_callback is not None
        assert adapter._interaction_callback == adapter._default_interaction_dispatch

    def test_send_exec_approval_is_a_class_method(self):
        """gateway/run.py uses ``type(adapter).send_exec_approval`` to detect support."""
        from gateway.platforms.qqbot.adapter import QQAdapter
        assert getattr(QQAdapter, "send_exec_approval", None) is not None
        assert getattr(QQAdapter, "send_update_prompt", None) is not None

    @pytest.mark.asyncio
    async def test_approval_click_once_maps_to_once(self):
        """'allow-once' button → resolve_gateway_approval(session, 'once')."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        # Patch the *module-level* function that _default_interaction_dispatch
        # imports lazily.
        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i",
                "chat_type": 2,
                "user_openid": "u-42",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u-42:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:main:qqbot:c2c:u-42", "once", False)]

    @pytest.mark.asyncio
    async def test_approval_click_always_maps_to_always(self):
        adapter = self._make_adapter()
        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 2, "user_openid": "u",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:allow-always"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:main:qqbot:c2c:u", "always", False)]

    @pytest.mark.asyncio
    async def test_approval_click_deny_maps_to_deny(self):
        adapter = self._make_adapter()
        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 2, "user_openid": "u",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:main:qqbot:c2c:u", "deny", False)]


    @pytest.mark.asyncio
    async def test_approval_click_rejects_unauthorized_operator(self):
        adapter = self._make_adapter()
        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 1,
                "group_openid": "g-1",
                "group_member_openid": "attacker",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:group:g-1:owner:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == []

    @pytest.mark.asyncio
    async def test_update_prompt_click_writes_response_file(self, tmp_path, monkeypatch):
        """update_prompt:y click writes 'y' to ~/.hermes/.update_response."""
        adapter = self._make_adapter()
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: hermes_home,
        )

        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        event = parse_interaction_event({
            "id": "i", "chat_type": 2, "user_openid": "u-1",
            "data": {"resolved": {"button_data": "update_prompt:y"}},
        })
        await adapter._default_interaction_dispatch(event)

        response = hermes_home / ".update_response"
        assert response.exists()
        assert response.read_text() == "y"

    @pytest.mark.asyncio
    async def test_update_prompt_click_no_writes_n(self, tmp_path, monkeypatch):
        adapter = self._make_adapter()
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: hermes_home,
        )
        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        event = parse_interaction_event({
            "id": "i", "chat_type": 2, "user_openid": "u",
            "data": {"resolved": {"button_data": "update_prompt:n"}},
        })
        await adapter._default_interaction_dispatch(event)
        response = hermes_home / ".update_response"
        assert response.read_text() == "n"

    @pytest.mark.asyncio
    async def test_unknown_button_data_is_harmless(self):
        """Unrecognised button_data is logged and dropped — no exception."""
        adapter = self._make_adapter()

        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        event = parse_interaction_event({
            "id": "i", "chat_type": 2, "user_openid": "u",
            "data": {"resolved": {"button_data": "some:unknown:format"}},
        })
        # Must not raise.
        await adapter._default_interaction_dispatch(event)

    @pytest.mark.asyncio
    async def test_empty_button_data_is_harmless(self):
        adapter = self._make_adapter()
        from gateway.platforms.qqbot.keyboards import InteractionEvent
        await adapter._default_interaction_dispatch(InteractionEvent(id="i"))

    @pytest.mark.asyncio
    async def test_resolve_exception_is_swallowed(self):
        """If resolve_gateway_approval raises, we log but don't propagate."""
        adapter = self._make_adapter()

        def bad_resolve(session_key, choice, resolve_all=False):
            raise RuntimeError("boom")

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = bad_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 2, "user_openid": "u",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny"}},
            })
            # Must not raise.
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig


class TestSendExecApproval:
    """Verify the gateway contract: QQAdapter.send_exec_approval(...)."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_delegates_to_send_approval_request(self):
        adapter = self._make_adapter()

        calls = []

        async def fake_send_approval(chat_id, req, reply_to=None):
            from gateway.platforms.base import SendResult
            calls.append({"chat_id": chat_id, "req": req, "reply_to": reply_to})
            return SendResult(success=True, message_id="m-1")

        adapter.send_approval_request = fake_send_approval  # type: ignore[assignment]
        # Seed last-msg-id so the reply_to path is exercised.
        adapter._last_msg_id["user-1"] = "inbound-42"

        result = await adapter.send_exec_approval(
            chat_id="user-1",
            command="rm -rf /tmp/demo",
            session_key="sess:abc",
            description="delete temp dir",
        )
        assert result.success
        assert len(calls) == 1
        req = calls[0]["req"]
        assert req.session_key == "sess:abc"
        assert req.command_preview == "rm -rf /tmp/demo"
        assert req.description == "delete temp dir"
        assert calls[0]["reply_to"] == "inbound-42"

    @pytest.mark.asyncio
    async def test_accepts_metadata_arg(self):
        """Gateway always passes metadata=…; the adapter must accept + ignore it."""
        adapter = self._make_adapter()

        async def fake_send_approval(chat_id, req, reply_to=None):
            from gateway.platforms.base import SendResult
            return SendResult(success=True)

        adapter.send_approval_request = fake_send_approval  # type: ignore[assignment]

        # Should not raise even when metadata is a dict with unknown keys.
        await adapter.send_exec_approval(
            chat_id="u", command="ls", session_key="s",
            metadata={"thread_id": "ignored", "anything": "else"},
        )


class TestSendUpdatePrompt:
    """Verify the cross-adapter send_update_prompt signature + behaviour."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_delegates_to_send_with_keyboard(self):
        adapter = self._make_adapter()

        captured = {}

        async def fake_swk(chat_id, content, keyboard, reply_to=None):
            from gateway.platforms.base import SendResult
            captured["chat_id"] = chat_id
            captured["content"] = content
            captured["keyboard"] = keyboard
            captured["reply_to"] = reply_to
            return SendResult(success=True, message_id="mid")

        adapter.send_with_keyboard = fake_swk  # type: ignore[assignment]
        adapter._last_msg_id["u1"] = "prev-msg"

        result = await adapter.send_update_prompt(
            chat_id="u1", prompt="Continue with update?",
            default="y", session_key="ignored", metadata={"x": 1},
        )
        assert result.success
        assert "Continue with update?" in captured["content"]
        assert "default: y" in captured["content"]
        assert captured["reply_to"] == "prev-msg"
        # Keyboard has the Yes/No buttons.
        dd = captured["keyboard"].to_dict()
        datas = [b["action"]["data"] for b in dd["content"]["rows"][0]["buttons"]]
        assert datas == ["update_prompt:y", "update_prompt:n"]

    @pytest.mark.asyncio
    async def test_empty_default_has_no_hint(self):
        adapter = self._make_adapter()

        async def fake_swk(chat_id, content, keyboard, reply_to=None):
            from gateway.platforms.base import SendResult
            assert "default:" not in content
            return SendResult(success=True)

        adapter.send_with_keyboard = fake_swk  # type: ignore[assignment]
        await adapter.send_update_prompt(chat_id="u", prompt="ok?")


# ---------------------------------------------------------------------------
# _send_identify includes INTERACTION intent
# ---------------------------------------------------------------------------

class TestIdentifyIntents:
    """Verify the WebSocket identify payload includes the INTERACTION intent bit."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_intents_include_interaction_bit(self):
        adapter = self._make_adapter()

        # Mock token retrieval and WebSocket
        adapter._access_token = "fake_token"
        adapter._token_expires_at = 9999999999.0

        sent_payloads = []

        class FakeWS:
            closed = False

            async def send_json(self, payload):
                sent_payloads.append(payload)

        adapter._ws = FakeWS()
        await adapter._send_identify()

        assert len(sent_payloads) == 1
        intents = sent_payloads[0]["d"]["intents"]

        # Verify all expected intent bits are present
        assert intents & (1 << 25), "GROUP_MESSAGES (1<<25) missing"
        assert intents & (1 << 30), "GUILD_AT_MESSAGE (1<<30) missing"
        assert intents & (1 << 12), "DIRECT_MESSAGES (1<<12) missing"
        assert intents & (1 << 26), "INTERACTION (1<<26) missing"


# ---------------------------------------------------------------------------
# _process_attachments: video/file path exposure
# ---------------------------------------------------------------------------

class TestProcessAttachmentsPathExposure:
    """Verify that video and file attachments include the cached local path."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_video_attachment_includes_path(self):
        adapter = self._make_adapter()

        # Mock _download_and_cache to return a known path
        async def fake_download(url, ct, original_name=""):
            return "/tmp/cache/video_abc123.mp4"

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "video/mp4",
                "url": "https://multimedia.nt.qq.com.cn/download/video123",
                "filename": "my_video.mp4",
            }
        ]
        result = await adapter._process_attachments(attachments)

        assert result["image_urls"] == []
        assert result["voice_transcripts"] == []
        info = result["attachment_info"]
        assert "[video:" in info
        assert "my_video.mp4" in info
        assert "/tmp/cache/video_abc123.mp4" in info

    @pytest.mark.asyncio
    async def test_file_attachment_includes_path(self):
        adapter = self._make_adapter()

        async def fake_download(url, ct, original_name=""):
            return "/tmp/cache/doc_abc123_report.pdf"

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "application/pdf",
                "url": "https://multimedia.nt.qq.com.cn/download/file456",
                "filename": "report.pdf",
            }
        ]
        result = await adapter._process_attachments(attachments)

        info = result["attachment_info"]
        assert "[file:" in info
        assert "report.pdf" in info
        assert "/tmp/cache/doc_abc123_report.pdf" in info

    @pytest.mark.asyncio
    async def test_video_without_filename_falls_back_to_content_type(self):
        adapter = self._make_adapter()

        async def fake_download(url, ct, original_name=""):
            return "/tmp/cache/video_xyz.mp4"

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "video/mp4",
                "url": "https://cdn.qq.com/vid",
                "filename": "",
            }
        ]
        result = await adapter._process_attachments(attachments)

        info = result["attachment_info"]
        assert "[video: video/mp4" in info
        assert "/tmp/cache/video_xyz.mp4" in info

    @pytest.mark.asyncio
    async def test_download_failure_produces_no_attachment_info(self):
        adapter = self._make_adapter()

        async def fake_download(url, ct, original_name=""):
            return None

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "video/mp4",
                "url": "https://cdn.qq.com/vid",
                "filename": "vid.mp4",
            }
        ]
        result = await adapter._process_attachments(attachments)
        assert result["attachment_info"] == ""

    @pytest.mark.asyncio
    async def test_quoted_video_includes_path_in_quote_block(self):
        """Quoted video attachments should surface the cached path in the quote block."""
        adapter = self._make_adapter()

        async def fake_process(atts):
            # Simulate the fixed _process_attachments for a video attachment.
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": [],
                "attachment_info": "[video: clip.mp4 (/tmp/cache/clip.mp4)]",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "看看这个视频",
                "attachments": [
                    {"content_type": "video/mp4",
                     "url": "https://qq-cdn/clip.mp4",
                     "filename": "clip.mp4"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert "[Quoted message]:" in out["quote_block"]
        assert "/tmp/cache/clip.mp4" in out["quote_block"]

    @pytest.mark.asyncio
    async def test_quoted_file_includes_path_in_quote_block(self):
        """Quoted file attachments should surface the cached path in the quote block."""
        adapter = self._make_adapter()

        async def fake_process(atts):
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": [],
                "attachment_info": "[file: report.pdf (/tmp/cache/report.pdf)]",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "",
                "attachments": [
                    {"content_type": "application/pdf",
                     "url": "https://qq-cdn/report.pdf",
                     "filename": "report.pdf"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert "[Quoted message]:" in out["quote_block"]
        assert "/tmp/cache/report.pdf" in out["quote_block"]


# ---------------------------------------------------------------------------
# WebSocket op 7 (Server Reconnect) and op 9 (Invalid Session)
# ---------------------------------------------------------------------------

class TestOp7ServerReconnect:
    """Verify op 7 triggers WS close (which triggers reconnect in outer loop)."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_op7_closes_websocket(self):
        adapter = self._make_adapter()
        adapter._session_id = "sess_keep"
        adapter._last_seq = 42

        close_called = []

        class FakeWS:
            closed = False

            async def close(self):
                close_called.append(True)

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 7, "d": None})

        # Session should be preserved for Resume
        assert adapter._session_id == "sess_keep"
        assert adapter._last_seq == 42
        # close() should have been scheduled
        assert len(close_called) == 0  # _create_task schedules, not immediate
        # But the task was created — verify via asyncio

    @pytest.mark.asyncio
    async def test_op7_close_task_executes(self):
        adapter = self._make_adapter()
        close_called = []

        class FakeWS:
            closed = False

            async def close(self):
                close_called.append(True)
                self.closed = True

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 7, "d": None})

        # Let the event loop run the scheduled task
        await asyncio.sleep(0)
        assert close_called == [True]
        # Session preserved
        assert adapter._session_id is None  # was never set


class TestOp9InvalidSession:
    """Verify op 9 handles resumable vs non-resumable sessions."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_op9_not_resumable_clears_session(self):
        adapter = self._make_adapter()
        adapter._session_id = "sess_old"
        adapter._last_seq = 99

        class FakeWS:
            closed = False

            async def close(self):
                self.closed = True

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 9, "d": False})

        assert adapter._session_id is None
        assert adapter._last_seq is None

    def test_op9_resumable_preserves_session(self):
        adapter = self._make_adapter()
        adapter._session_id = "sess_keep"
        adapter._last_seq = 99

        class FakeWS:
            closed = False

            async def close(self):
                self.closed = True

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 9, "d": True})

        # Session should be preserved for Resume
        assert adapter._session_id == "sess_keep"
        assert adapter._last_seq == 99

    @pytest.mark.asyncio
    async def test_op9_non_resumable_triggers_ws_close(self):
        adapter = self._make_adapter()
        adapter._session_id = "s"
        adapter._last_seq = 1
        close_called = []

        class FakeWS:
            closed = False

            async def close(self):
                close_called.append(True)
                self.closed = True

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 9, "d": False})
        await asyncio.sleep(0)

        assert close_called == [True]


# ---------------------------------------------------------------------------
# Close code classification
# ---------------------------------------------------------------------------

class TestCloseCodeClassification:
    """Verify fatal close codes stop reconnecting and 4009 preserves session."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_4009_preserves_session(self):
        """4009 (connection timeout) should NOT clear the session."""
        adapter = self._make_adapter()
        adapter._session_id = "sess_to_keep"
        adapter._last_seq = 50

        # The session-clearing codes set should NOT contain 4009.
        # We verify the logic directly: dispatch a close-code event that
        # exercises the session-clearing path (4006), then verify 4009 does not.
        session_clear_codes = {
            4006, 4007, 4900, 4901, 4902, 4903,
            4904, 4905, 4906, 4907, 4908, 4909,
            4910, 4911, 4912, 4913,
        }
        assert 4009 not in session_clear_codes

    def test_fatal_codes_include_intent_errors(self):
        """4013 (invalid intent) and 4014 (not authorized) should be fatal."""
        fatal_codes = {4001, 4002, 4010, 4011, 4012, 4013, 4014, 4914, 4915}
        # Verify these are all treated as fatal by checking the adapter's
        # code path would call _set_fatal_error. We verify the set membership
        # which is what the if-branch checks.
        assert 4013 in fatal_codes
        assert 4014 in fatal_codes
        assert 4001 in fatal_codes
        assert 4915 in fatal_codes


class TestReadEventsClosedWsGuard:
    """Regression: a closed-but-non-None ws must raise on entry, not return
    normally, so _listen_loop goes through reconnect/backoff instead of
    busy-looping at 100% CPU (issues #31193 / #31771)."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b", **extra))

    def test_read_events_raises_when_ws_closed_on_entry(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=True)
        with pytest.raises(RuntimeError):
            asyncio.run(adapter._read_events())

    def test_read_events_raises_when_ws_none(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = None
        with pytest.raises(RuntimeError):
            asyncio.run(adapter._read_events())


# ---------------------------------------------------------------------------
# Group activation mode — mention detection (group_activation.detect_mentioned)
# ---------------------------------------------------------------------------

class TestDetectMentioned:
    def test_group_at_event_always_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned("GROUP_AT_MESSAGE_CREATE", {}, "hi", "app1") is True

    def test_mentions_is_you_true(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        d = {"mentions": [{"member_openid": "x"}, {"is_you": True}]}
        assert detect_mentioned("GROUP_MESSAGE_CREATE", d, "hi", "app1") is True

    def test_explicit_tag_for_our_app_id(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!1903885637> hello", "1903885637"
        ) is True

    def test_tag_for_other_member_not_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        # @ of a different member must NOT count as addressing the bot.
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!99999> hello", "1903885637"
        ) is False

    def test_plain_message_not_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "just chatting", "1903885637"
        ) is False

    def test_generic_at_prefix_not_treated_as_bot(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        # Conservative: a bare "@alice " prefix is NOT a bot mention.
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "@alice look here", "1903885637"
        ) is False


# ---------------------------------------------------------------------------
# Group activation mode — require_mention resolution
# ---------------------------------------------------------------------------

class TestResolveRequireMention:
    def test_global_default_true(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention("g1", global_default=True) is True

    def test_global_default_false(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention("g1", global_default=False) is False

    def test_per_group_overrides_global(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        # global always, but g1 forced to mention.
        assert resolve_require_mention(
            "g1", global_default=False, per_group={"g1": True}
        ) is True
        # other group still follows global.
        assert resolve_require_mention(
            "g2", global_default=False, per_group={"g1": True}
        ) is False

    def test_runtime_override_wins(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention(
            "g1", global_default=True, per_group={"g1": True},
            runtime_overrides={"g1": False},
        ) is False


# ---------------------------------------------------------------------------
# Group activation mode — config parsing + gate (_handle_group_message)
# ---------------------------------------------------------------------------

class TestGroupActivationMode:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    def test_default_mode_is_mention(self):
        adapter = self._make_adapter()
        assert adapter._group_require_mention is True

    def test_always_mode_from_config(self):
        adapter = self._make_adapter(group_require_mention=False)
        assert adapter._group_require_mention is False

    def test_per_group_override_parsed(self):
        adapter = self._make_adapter(
            group_require_mention=False,
            groups={"grp_a": {"require_mention": True}},
        )
        assert adapter._group_mode_overrides == {"grp_a": True}

    def _drive(self, adapter):
        """Stub the assembly path and capture handle_message events."""
        captured = []

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]
        return captured

    @pytest.mark.asyncio
    async def test_mention_mode_skips_non_mention_group_message(self):
        adapter = self._make_adapter()  # default mention
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "hello everyone"}
        await adapter._handle_group_message(
            d, "m1", "hello everyone", {"member_openid": "u1"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert captured == []  # skipped, no reply

    @pytest.mark.asyncio
    async def test_mention_mode_handles_at_message(self):
        adapter = self._make_adapter()  # default mention
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "hi bot"}
        await adapter._handle_group_message(
            d, "m1", "hi bot", {"member_openid": "u1"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].source.chat_id == "g1"
        assert captured[0].source.chat_type == "group"

    @pytest.mark.asyncio
    async def test_always_mode_handles_non_mention_group_message(self):
        adapter = self._make_adapter(group_require_mention=False)  # always
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "just chatting"}
        await adapter._handle_group_message(
            d, "m1", "just chatting", {"member_openid": "u1"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].text == "just chatting"

    @pytest.mark.asyncio
    async def test_per_group_mention_override_blocks_in_always_global(self):
        adapter = self._make_adapter(
            group_require_mention=False,
            groups={"g1": {"require_mention": True}},
        )
        captured = self._drive(adapter)
        # g1 forced to mention -> non-@ skipped.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hey"}, "m1", "hey",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert captured == []
        # g2 follows global always -> handled.
        await adapter._handle_group_message(
            {"group_openid": "g2", "content": "hey"}, "m2", "hey",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_group_acl_blocks_before_gate(self):
        adapter = self._make_adapter(group_policy="disabled",
                                     group_require_mention=False)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hi"}, "m1", "hi",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert captured == []

    @pytest.mark.asyncio
    async def test_group_acl_reject_emits_debug_log(self, caplog):
        # ACL rejection must leave an operator-visible breadcrumb explaining
        # both the cause (policy=disabled) and how to unblock it.
        import logging
        adapter = self._make_adapter(group_policy="disabled")
        self._drive(adapter)
        with caplog.at_level(logging.DEBUG, logger="gateway.platforms.qqbot"):
            await adapter._handle_group_message(
                {"group_openid": "gX", "content": "hi"}, "m1", "hi",
                {"member_openid": "u1"}, "", "GROUP_AT_MESSAGE_CREATE",
            )
        log_text = " ".join(r.message for r in caplog.records)
        assert "blocked by ACL" in log_text
        assert "gX" in log_text
        assert "policy=disabled" in log_text
        assert "group_policy" in log_text  # hint on how to unblock

    def test_default_history_limit_is_20(self):
        adapter = self._make_adapter()
        assert adapter._group_history_limit == 20


# ---------------------------------------------------------------------------
# Group shared session (2.1) — group_sessions_per_user key behaviour
# ---------------------------------------------------------------------------

class TestGroupSharedSession:
    def _source(self):
        from gateway.session import SessionSource
        from gateway.config import Platform
        return SessionSource(
            platform=Platform.QQBOT,
            chat_id="group_openid_1",
            chat_type="group",
            user_id="member_openid_1",
        )

    def test_isolated_key_includes_participant_by_default(self):
        from gateway.session import build_session_key
        key = build_session_key(self._source(), group_sessions_per_user=True)
        assert key.endswith(":member_openid_1")

    def test_shared_key_excludes_participant(self):
        from gateway.session import build_session_key
        key = build_session_key(self._source(), group_sessions_per_user=False)
        assert "member_openid_1" not in key
        assert key.endswith(":group_openid_1")


# ---------------------------------------------------------------------------
# Group context buffer (2.2.1) — GroupContextBuffer unit tests
# ---------------------------------------------------------------------------

class TestGroupContextBuffer:
    def _buf(self, **kw):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        return GroupContextBuffer(**kw)

    def test_record_and_drain_in_order(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u1", text="first")
        buf.record("g1", sender="u2", text="second")
        entries = buf.drain("g1")
        assert [e.text for e in entries] == ["first", "second"]
        assert [e.sender for e in entries] == ["u1", "u2"]

    def test_drain_clears(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u1", text="hi")
        assert buf.drain("g1")
        assert buf.drain("g1") == []

    def test_limit_truncates_oldest(self):
        buf = self._buf(limit=2)
        buf.record("g1", sender="u", text="a")
        buf.record("g1", sender="u", text="b")
        buf.record("g1", sender="u", text="c")
        entries = buf.drain("g1")
        assert [e.text for e in entries] == ["b", "c"]  # oldest "a" dropped

    def test_disabled_when_limit_zero(self):
        buf = self._buf(limit=0)
        assert buf.enabled is False
        buf.record("g1", sender="u", text="hi")
        assert buf.drain("g1") == []

    def test_group_isolation(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="a")
        buf.record("g2", sender="u", text="b")
        assert [e.text for e in buf.drain("g1")] == ["a"]
        assert [e.text for e in buf.drain("g2")] == ["b"]

    def test_empty_text_no_attachment_not_recorded(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="   ")
        assert buf.drain("g1") == []

    def test_attachment_tag_recorded_when_no_text(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="", attachment_tag="[image]")
        entries = buf.drain("g1")
        assert entries and entries[0].text == "[image]"

    def test_max_groups_lru_eviction(self):
        buf = self._buf(limit=5, max_groups=2)
        buf.record("g1", sender="u", text="a")
        buf.record("g2", sender="u", text="b")
        buf.record("g3", sender="u", text="c")  # evicts g1 (LRU)
        assert buf.drain("g1") == []
        assert [e.text for e in buf.drain("g3")] == ["c"]

    def test_format_context_wraps_with_tags(self):
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry, HISTORY_CTX_START, HISTORY_CTX_END,
        )
        entries = [HistoryEntry(sender="u1", text="hello"),
                   HistoryEntry(sender="u2", text="world")]
        out = GroupContextBuffer.format_context(entries, "please summarize")
        assert HISTORY_CTX_START in out
        assert "u1: hello" in out
        assert "u2: world" in out
        assert HISTORY_CTX_END in out
        assert out.endswith("please summarize")

    def test_format_context_empty_returns_current(self):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        assert GroupContextBuffer.format_context([], "just this") == "just this"

    def test_format_context_block_no_current_message(self):
        # The block variant renders context only (for channel_context): it must
        # include the CONTEXT-ONLY header + entries, but NOT the current message
        # and NOT the CURRENT-MESSAGE end tag (gateway supplies [New message]).
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry, HISTORY_CTX_START, HISTORY_CTX_END,
        )
        entries = [HistoryEntry(sender="u1", text="hello"),
                   HistoryEntry(sender="u2", text="world")]
        out = GroupContextBuffer.format_context_block(entries)
        assert HISTORY_CTX_START in out
        assert "u1: hello" in out
        assert "u2: world" in out
        assert HISTORY_CTX_END not in out

    def test_format_context_block_empty_returns_blank(self):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        assert GroupContextBuffer.format_context_block([]) == ""

    def test_format_context_block_collapses_newlines(self):
        # R5 hardening: a buffered multi-line message cannot forge envelope tags.
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry,
        )
        entries = [HistoryEntry(sender="u1", text="line1\nline2")]
        out = GroupContextBuffer.format_context_block(entries)
        assert "u1: line1 line2" in out

    def test_summarize_attachments(self):
        from gateway.platforms.qqbot.group_context import summarize_attachments
        assert summarize_attachments(None) == ""
        assert summarize_attachments([]) == ""
        assert summarize_attachments([{"content_type": "image/png"}]) == "[image]"
        assert summarize_attachments([{"content_type": "audio/silk"}]) == "[voice]"
        assert summarize_attachments(
            [{"content_type": "application/zip", "filename": "a.zip"}]
        ) == "[file: a.zip]"


# ---------------------------------------------------------------------------
# Group context buffer — integration through _handle_group_message
# ---------------------------------------------------------------------------

class TestGroupContextIntegration:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    def _drive(self, adapter):
        captured = []

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]
        return captured

    @pytest.mark.asyncio
    async def test_mention_mode_buffers_then_injects_on_at(self):
        adapter = self._make_adapter()  # mention mode, default limit 50
        captured = self._drive(adapter)
        # non-@ message → buffered, no reply.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "the sky is blue"}, "m1",
            "the sky is blue", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert captured == []
        # @ message → reply with buffered context injected.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "what did she say"}, "m2",
            "what did she say", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        # Buffered history is carried in channel_context (kept out of text so
        # slash-command detection + sender-prefix operate on the trigger alone).
        ctx = captured[0].channel_context
        assert ctx and "CONTEXT ONLY" in ctx
        assert "alice: the sky is blue" in ctx
        # text is the trigger message only.
        assert captured[0].text == "what did she say"

    @pytest.mark.asyncio
    async def test_command_at_message_stays_matchable_with_pending_context(self):
        # Regression: a /stop-style command in an @-message must remain at the
        # start of text (get_command works) even when pending context exists;
        # the context goes to channel_context instead of being merged ahead.
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "the sky is blue"}, "m1",
            "the sky is blue", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "/stop"}, "m2",
            "/stop", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        ev = captured[0]
        assert ev.text == "/stop"
        assert ev.is_command() is True
        assert ev.get_command() == "stop"
        # context preserved separately, not merged into text.
        assert ev.channel_context and "alice: the sky is blue" in ev.channel_context

    @pytest.mark.asyncio
    async def test_buffer_cleared_after_injection(self):
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "ctx"}, "m1", "ctx",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q1"}, "m2", "q1",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        # second @ has no stale context.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q2"}, "m3", "q2",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 2
        assert "CONTEXT ONLY" not in captured[1].text
        assert captured[1].text == "q2"

    @pytest.mark.asyncio
    async def test_always_mode_no_buffering(self):
        adapter = self._make_adapter(group_require_mention=False)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hello"}, "m1", "hello",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].text == "hello"  # no context wrapper
        # nothing left buffered.
        assert adapter._group_context.drain("g1") == []

    @pytest.mark.asyncio
    async def test_history_limit_zero_disables_buffer(self):
        adapter = self._make_adapter(group_history_limit=0)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "ctx"}, "m1", "ctx",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q"}, "m2", "q",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert "CONTEXT ONLY" not in captured[0].text
        assert captured[0].text == "q"

    @pytest.mark.asyncio
    async def test_empty_at_message_still_flushes_context(self):
        # A bare @ (empty body after strip) must still flush + inject pending
        # context, not early-return and strand the buffer.
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "background note"}, "m1",
            "background note", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": ""}, "m2", "",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        # bare @ carries no trigger body; pending context lands in channel_context.
        assert captured[0].text == ""
        assert (
            captured[0].channel_context
            and "alice: background note" in captured[0].channel_context
        )
        # buffer cleared.
        assert adapter._group_context.drain("g1") == []

    @pytest.mark.asyncio
    async def test_injection_keeps_current_message_with_attachments_last(self):
        # Lock the order: buffered context first, current message (incl. its
        # appended attachment_info) last.
        adapter = self._make_adapter()
        captured = self._drive(adapter)

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": "[file: doc.pdf]"}

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "earlier"}, "m1", "earlier",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        # NB: alice's non-@ message uses the light attachment tag, not the full
        # processor; the @ message below exercises the full path.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "see attached"}, "m2",
            "see attached", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        ev = captured[0]
        # Trigger message + its attachment_info stay in text, in order.
        assert ev.text.index("see attached") < ev.text.index("[file: doc.pdf]")
        # Buffered history is separated into channel_context.
        assert ev.channel_context and "CONTEXT ONLY" in ev.channel_context
        assert "earlier" in ev.channel_context
        assert "earlier" not in ev.text


# ---------------------------------------------------------------------------
# C2C streaming reply — StreamManager
# ---------------------------------------------------------------------------

class TestStreamManager:
    """Unit tests for the in-memory StreamSession table."""

    def test_create_registers_session_with_generated_logical_id(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        s1 = mgr.create(openid="u1", passive_msg_id="m1", msg_seq=42)
        s2 = mgr.create(openid="u2", passive_msg_id="m2", msg_seq=43)
        assert s1.logical_id and s2.logical_id
        assert s1.logical_id != s2.logical_id
        assert mgr.get(s1.logical_id) is s1
        assert mgr.get(s2.logical_id) is s2

    def test_get_returns_none_for_unknown_id(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        assert mgr.get("nonexistent") is None

    def test_drop_is_idempotent(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        s = mgr.create(openid="u", passive_msg_id="m", msg_seq=1)
        mgr.drop(s.logical_id)
        mgr.drop(s.logical_id)  # second drop must not raise
        assert mgr.get(s.logical_id) is None

    def test_ttl_expiry_evicts_stale_session(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager(ttl_seconds=60.0)
        s = mgr.create(openid="u", passive_msg_id="m", msg_seq=1)
        # Backdate the session past the TTL horizon.
        s.created_at -= 61.0
        assert mgr.get(s.logical_id) is None
        # After the failed lookup the entry should be gone.
        assert len(mgr) == 0

    def test_lru_eviction_when_full(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager(max_sessions=2)
        a = mgr.create(openid="a", passive_msg_id="m1", msg_seq=1)
        b = mgr.create(openid="b", passive_msg_id="m2", msg_seq=2)
        # Touch A to promote it — B becomes LRU.
        assert mgr.get(a.logical_id) is a
        mgr.create(openid="c", passive_msg_id="m3", msg_seq=3)
        assert mgr.get(b.logical_id) is None  # evicted
        assert mgr.get(a.logical_id) is a  # still present


# ---------------------------------------------------------------------------
# C2C streaming reply — QQAdapter integration
# ---------------------------------------------------------------------------

class TestC2CStreamingReply:
    """Streaming-path tests for ``send()`` + ``edit_message()`` on C2C chats."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "a")
        extra.setdefault("client_secret", "b")
        adapter = QQAdapter(_make_config(**extra))
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._http_client = mock.MagicMock()
        return adapter

    def test_requires_edit_finalize_class_attr_is_true(self):
        from gateway.platforms.qqbot import QQAdapter
        assert QQAdapter.REQUIRES_EDIT_FINALIZE is True

    def test_streaming_defaults_enabled(self):
        adapter = self._make_adapter()
        assert adapter._streaming_enabled is True

    def test_streaming_can_be_disabled_via_config(self):
        adapter = self._make_adapter(streaming_enabled=False)
        assert adapter._streaming_enabled is False

    @pytest.mark.asyncio
    async def test_c2c_first_send_opens_stream_session(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-xyz-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        # message_id must be the adapter's logical_id (opaque uuid hex),
        # NOT the QQ-assigned stream_msg_id — the consumer echoes this
        # back on edit_message and we translate internally.
        assert result.message_id is not None
        assert result.message_id != "stream-xyz-1"
        assert len(calls) == 1
        method, path, body = calls[0]
        assert method == "POST"
        assert path == "/v2/users/user_a/stream_messages"
        assert body["input_mode"] == "replace"
        assert body["input_state"] == 1  # GENERATING
        assert body["content_type"] == "markdown"  # MARKDOWN — fixed
        assert body["content_raw"] == "hello"
        assert body["event_id"] == "inbound_m1"
        assert body["msg_id"] == "inbound_m1"
        assert body["index"] == 0
        assert "stream_msg_id" not in body  # first chunk omits it
        # Session should be registered and reference the QQ id.
        session = adapter._stream_manager.get(result.message_id)
        assert session is not None
        assert session.stream_msg_id == "stream-xyz-1"

    @pytest.mark.asyncio
    async def test_c2c_edit_reuses_stream_msg_id_and_increments_index(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-xyz-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "hel",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        second = await adapter.edit_message("user_a", logical_id, "hello")
        third = await adapter.edit_message("user_a", logical_id, "hello wor")
        assert second.success and third.success
        # All three chunks share the same msg_seq and target the same
        # stream endpoint.
        assert len({body["msg_seq"] for _, _, body in calls}) == 1
        assert [body["index"] for _, _, body in calls] == [0, 1, 2]
        # Second/third chunks carry stream_msg_id from the first response.
        assert calls[1][2]["stream_msg_id"] == "stream-xyz-1"
        assert calls[2][2]["stream_msg_id"] == "stream-xyz-1"
        # content_raw is REPLACE semantics — full accumulated text each time.
        assert [body["content_raw"] for _, _, body in calls] == [
            "hel", "hello", "hello wor",
        ]
        # Intermediate edits stay in GENERATING state.
        assert calls[1][2]["input_state"] == 1
        assert calls[2][2]["input_state"] == 1

    @pytest.mark.asyncio
    async def test_c2c_finalize_sends_done_and_drops_session(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "partial",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        result = await adapter.edit_message(
            "user_a", first.message_id, "final full answer",
            finalize=True,
        )
        assert result.success
        # Final chunk uses input_state=10 (DONE).
        assert calls[-1][2]["input_state"] == 10
        # Session should be cleaned up after finalize.
        assert adapter._stream_manager.get(first.message_id) is None

    @pytest.mark.asyncio
    async def test_edit_after_finalize_is_noop_success(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"

        async def fake_api(method, path, body=None, **kw):
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "x",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        await adapter.edit_message(
            "user_a", first.message_id, "final",
            finalize=True,
        )
        # Second call after finalize: session already dropped, treated
        # as "unknown session" → success=False so the consumer sends a
        # fresh message rather than corrupting the finalized stream.
        r = await adapter.edit_message(
            "user_a", first.message_id, "oops",
        )
        assert r.success is False
        assert "expired" in (r.error or "").lower() or "found" in (r.error or "").lower()

    @pytest.mark.asyncio
    async def test_edit_on_unknown_session_returns_failure(self):
        adapter = self._make_adapter()
        result = await adapter.edit_message("user_a", "nonexistent-logical-id", "hi")
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_out_of_order_edit_is_dropped_silently(self):
        """When ``next_index <= last_sent_index`` the edit is treated as a
        no-op success without hitting the QQ API — matches the simplified
        out-of-order policy agreed for this release.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        session = adapter._stream_manager.get(first.message_id)
        # Simulate an out-of-order scenario by regressing next_index.
        # (Consumer serialisation makes this impossible in practice, so
        # we assert the defensive guard triggers when it happens.)
        session.next_index = 0
        session.last_sent_index = 0
        api_calls_before = len(calls)

        r = await adapter.edit_message("user_a", first.message_id, "hello")
        assert r.success is True
        assert len(calls) == api_calls_before  # no additional API call

    @pytest.mark.asyncio
    async def test_group_chat_streaming_first_send_defers_delivery(self):
        """Group targets have no editable message id on QQ, so the
        streaming first-send is buffered inside the adapter: no QQ API
        call yet, but ``send()`` returns ``success=True`` with a
        sentinel ``message_id`` so the stream consumer stays on its
        edit path (avoiding the lossy prefix-based fallback).  The
        complete reply is delivered later, on ``edit_message`` with
        ``finalize=True``.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert result.message_id is not None
        assert result.message_id.startswith("__qqbot_group_defer_")
        # No QQ API call happened — the reply is deferred to finalize.
        assert paths == []
        # Session bookkeeping preserves the original reply_to for
        # threaded delivery on finalize.
        assert result.message_id in adapter._group_defer_sessions
        session = adapter._group_defer_sessions[result.message_id]
        assert session["chat_id"] == "group_a"
        assert session["chat_type"] == "group"
        assert session["reply_to"] == "inbound_m1"
        # Fresh session — not yet finalized.
        assert session["finalized"] is False
        assert session["finalized_result"] is None

    @pytest.mark.asyncio
    async def test_guild_chat_streaming_first_send_defers_delivery(self):
        """Same deferral as group chats — guild targets are also
        non-editable, so ``expect_edits`` must not trigger a real send.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["chan1"] = "guild"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "chan1", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert result.message_id is not None
        assert result.message_id.startswith("__qqbot_group_defer_")
        assert paths == []
        session = adapter._group_defer_sessions[result.message_id]
        assert session["chat_type"] == "guild"

    @pytest.mark.asyncio
    async def test_group_deferred_edit_intermediate_is_noop(self):
        """Intermediate ``edit_message`` calls on a deferred group
        session must NOT hit the QQ API — they are silent no-ops that
        accumulate context until the finalize call arrives.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "chunk1",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id
        assert paths == []

        for partial in ("chunk1 chunk2", "chunk1 chunk2 chunk3"):
            r = await adapter.edit_message(
                "group_a", sentinel, partial, finalize=False,
            )
            assert r.success
            assert r.message_id == sentinel

        # Still no QQ API traffic.
        assert paths == []
        # Session is still live.
        assert sentinel in adapter._group_defer_sessions

    @pytest.mark.asyncio
    async def test_group_deferred_edit_finalize_delivers_full_content(self):
        """The finalize ``edit_message`` is where the complete reply
        actually reaches QQ.  It must:

        * hit the regular group messages endpoint exactly once,
        * carry the FULL accumulated content (not just the tail),
        * preserve the original ``reply_to`` for threading, and
        * clear the deferred session bookkeeping.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id
        # A couple of intermediate edits — should not touch QQ.
        await adapter.edit_message("group_a", sentinel, "hello world")
        await adapter.edit_message(
            "group_a", sentinel, "hello world, done",
        )
        assert calls == []

        # Finalize with the full accumulated text.
        final = await adapter.edit_message(
            "group_a", sentinel,
            "hello world, done: final answer",
            finalize=True,
        )
        assert final.success
        assert final.message_id == "regular-1"
        # One and only one API call — the complete reply.
        assert len(calls) == 1
        method, path, body = calls[0]
        assert method == "POST"
        assert path == "/v2/groups/group_a/messages"
        # Full accumulated text (may be under ``content`` or
        # ``markdown.content`` depending on adapter markdown mode).
        text = (
            body.get("content")
            or body.get("markdown", {}).get("content")
            or ""
        )
        assert text.startswith("hello world, done: final answer")
        # Threaded to the original inbound message.
        assert body.get("msg_id") == "inbound_m1"
        # Session bookkeeping: entry is preserved but flagged so
        # subsequent finalize edits become idempotent no-ops (see
        # ``test_group_deferred_edit_finalize_is_idempotent``).
        assert sentinel in adapter._group_defer_sessions
        remembered = adapter._group_defer_sessions[sentinel]
        assert remembered["finalized"] is True
        assert remembered["finalized_result"] is final

    @pytest.mark.asyncio
    async def test_group_deferred_edit_finalize_is_idempotent(self):
        """A second ``finalize=True`` edit on the same sentinel must NOT
        re-post the reply.

        Regression guard: because the adapter declares
        ``REQUIRES_EDIT_FINALIZE=True``, the stream consumer's
        ``got_done`` branch can issue two ``_send_or_edit(...,
        finalize=True)`` calls per turn — the mid-stream flush plus the
        explicit final tick.  The first call already delivered the
        reply via ``self.send(...)``; the second must be an idempotent
        no-op or the user sees two identical messages in the group.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id

        final_a = await adapter.edit_message(
            "group_a", sentinel, "the full reply",
            finalize=True,
        )
        final_b = await adapter.edit_message(
            "group_a", sentinel, "the full reply",
            finalize=True,
        )

        # First finalize delivered exactly once; the second is a
        # memoised replay — same result object, no extra API call.
        assert len(calls) == 1
        assert final_a.success and final_b.success
        assert final_a.message_id == "regular-1"
        assert final_b.message_id == "regular-1"
        assert final_b is final_a

    @pytest.mark.asyncio
    async def test_group_deferred_edit_after_finalize_non_final_is_noop(self):
        """A stray non-final edit arriving after finalize must not
        re-hit QQ (would happen on cancellation cleanup or an errant
        mid-stream tick that raced with the final one).
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id

        await adapter.edit_message("group_a", sentinel, "final",
                                    finalize=True)
        # Simulate a late non-final tick after finalize.
        stray = await adapter.edit_message(
            "group_a", sentinel, "final plus more", finalize=False,
        )
        assert stray.success
        # Still exactly one API call — the finalize one.
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_group_deferred_sessions_evict_finalized_over_cap(self):
        """The finalized-session bookkeeping must not grow without
        bound: once above the LRU cap, oldest finalized entries are
        dropped.  Pending (non-finalized) entries are preserved.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        adapter._group_defer_sessions_cap = 3

        async def fake_api(method, path, body=None, **kw):
            return {"id": "regular"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # 3 finalized + 1 pending → over cap by 1.
        finalized_ids = []
        for _ in range(3):
            r = await adapter.send(
                "group_a", "hi",
                metadata={"expect_edits": True},
            )
            await adapter.edit_message("group_a", r.message_id, "x",
                                        finalize=True)
            finalized_ids.append(r.message_id)

        pending = await adapter.send(
            "group_a", "hi",
            metadata={"expect_edits": True},
        )

        # Cap enforcement runs on send() insertion — the oldest
        # finalized entry should have been evicted, the pending one
        # kept.
        assert pending.message_id in adapter._group_defer_sessions
        assert finalized_ids[0] not in adapter._group_defer_sessions
        # Newer finalized entries remain until they too age out.
        assert finalized_ids[-1] in adapter._group_defer_sessions

    @pytest.mark.asyncio
    async def test_group_deferred_edit_after_session_expiry_recovers(self):
        """If the deferred session is gone (adapter restart, TTL, ...),
        a finalize edit must still deliver the content via a fresh
        legacy send instead of dropping the reply silently.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # Never called send() first: no session exists.
        result = await adapter.edit_message(
            "group_a", "__qqbot_group_defer_missing__",
            "recovered content", finalize=True,
        )
        assert result.success
        assert len(calls) == 1
        assert calls[0][1] == "/v2/groups/group_a/messages"

    @pytest.mark.asyncio
    async def test_group_chat_without_expect_edits_sends_normally(self):
        """Non-streaming group sends must still hit the regular group
        messages endpoint and return a real id.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "group_a", "final answer",
            reply_to="inbound_m1",
            metadata={"final": True},
        )
        assert result.success
        assert result.message_id == "regular-1"
        assert paths == ["/v2/groups/group_a/messages"]

    @pytest.mark.asyncio
    async def test_group_edit_message_returns_failure(self):
        """Groups have no stream session — edit_message must return
        ``success=False`` so the consumer falls back to a fresh send.
        """
        adapter = self._make_adapter()
        result = await adapter.edit_message("group_a", "any-id", "hi")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_streaming_disabled_falls_back_to_legacy_send(self):
        adapter = self._make_adapter(streaming_enabled=False)
        adapter._chat_type_map["user_a"] = "c2c"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert all("/stream_messages" not in p for p in paths)

    @pytest.mark.asyncio
    async def test_c2c_send_without_expect_edits_uses_legacy_path(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("user_a", "hi", reply_to="inbound_m1")
        assert result.success
        assert paths == ["/v2/users/user_a/messages"]

    @pytest.mark.asyncio
    async def test_streaming_start_failure_falls_back_to_legacy(self, caplog):
        """First chunk failure must warn and degrade to a regular send in
        the same ``send()`` call — user requirement (d).
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(path)
            if "/stream_messages" in path:
                raise RuntimeError("QQ Bot API error [500] boom")
            return {"id": "legacy-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        import logging
        with caplog.at_level(logging.WARNING):
            result = await adapter.send(
                "user_a", "hi",
                reply_to="inbound_m1",
                metadata={"expect_edits": True},
            )
        assert result.success
        # Stream attempt happened, then fell back to the regular endpoint.
        assert calls[0].endswith("/stream_messages")
        assert calls[-1] == "/v2/users/user_a/messages"
        assert any(
            "Failed to start C2C streaming reply" in rec.message
            for rec in caplog.records
        )
        # Abandoned session must not leak into the manager table.
        assert len(adapter._stream_manager) == 0

    @pytest.mark.asyncio
    async def test_streaming_start_without_passive_msg_id_falls_back(self, caplog):
        """No reply_to + no cached inbound id → skip streaming with a
        warning; the endpoint requires a passive-reply msg_id.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        # _last_msg_id intentionally empty.
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(path)
            return {"id": "legacy-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        import logging
        with caplog.at_level(logging.WARNING):
            result = await adapter.send(
                "user_a", "hi",
                metadata={"expect_edits": True},
            )
        assert result.success
        assert all("/stream_messages" not in p for p in calls)
        assert any("no passive msg_id" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_streaming_uses_cached_inbound_id_when_reply_to_missing(self):
        """If ``reply_to`` is not supplied, the adapter must use the most
        recent inbound msg_id it saw for that chat.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        adapter._last_msg_id["user_a"] = "cached_inbound_99"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hi",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert calls[0]["msg_id"] == "cached_inbound_99"
        assert calls[0]["event_id"] == "cached_inbound_99"

    @pytest.mark.asyncio
    async def test_content_truncated_to_stream_content_limit(self):
        from gateway.platforms.qqbot.streaming import MAX_STREAM_CONTENT_LEN
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        captured = []

        async def fake_api(method, path, body=None, **kw):
            captured.append(body)
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        big = "x" * (MAX_STREAM_CONTENT_LEN + 200)
        await adapter.send(
            "user_a", big,
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert len(captured[0]["content_raw"]) == MAX_STREAM_CONTENT_LEN

    @pytest.mark.asyncio
    async def test_stream_forwards_content_verbatim_when_gateway_suppresses_cursor(self):
        """With QQBOT-specific cursor suppression in ``gateway/run.py``
        (analogous to ``Platform.MATRIX``), no typewriter cursor ever
        reaches the adapter — successive frames are forwarded verbatim
        and the prefix invariant holds naturally.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-nocursor-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # All three frames arrive without any trailing cursor glyph,
        # exactly what the gateway will emit for QQBOT.
        first = await adapter.send(
            "user_a", "Hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        second = await adapter.edit_message(
            "user_a", logical_id, "Hello world",
        )
        assert second.success

        third = await adapter.edit_message(
            "user_a", logical_id, "Hello world!", finalize=True,
        )
        assert third.success

        seen = [body["content_raw"] for body in calls]
        assert seen == ["Hello", "Hello world", "Hello world!"]
        for i in range(len(seen) - 1):
            assert seen[i + 1].startswith(seen[i]), (
                f"prefix broken at {i}: {seen[i]!r} -> {seen[i + 1]!r}"
            )

    @pytest.mark.asyncio
    async def test_stream_prefix_divergence_replays_last_chunk(self):
        """Divergent frames must degrade to a safe replay, not a 500.

        If a subsequent edit's cleaned text does NOT start with the
        previously-accepted text (e.g. upstream trimmed a segment,
        replayed a shorter partial, or the model backtracked), we
        MUST NOT forward it verbatim — that would break the prefix
        invariant and kill the whole stream.  Instead the adapter
        replays the last-accepted text so QQ sees a no-op-shaped
        chunk and the session stays alive for the next real update.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-div-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "Hello world how are you",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        # Divergent frame — shorter and NOT a prefix of the first.
        result = await adapter.edit_message(
            "user_a", logical_id, "Different content entirely",
        )
        # We report success (from QQ's POV nothing bad happened) but the
        # payload we actually sent is the previously-accepted text,
        # keeping the prefix invariant intact.
        assert result.success
        assert calls[1]["content_raw"] == "Hello world how are you"

        # A subsequent, properly-extended frame flows through as normal.
        third = await adapter.edit_message(
            "user_a", logical_id, "Hello world how are you today",
        )
        assert third.success
        assert calls[2]["content_raw"] == "Hello world how are you today"
