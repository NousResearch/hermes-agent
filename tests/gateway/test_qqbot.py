"""Tests for the QQ Bot platform adapter."""

import asyncio
import json
import os
import sys
from unittest import mock

import pytest

from gateway.config import Platform, PlatformConfig


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
        assert adapter._dm_policy == "open"

    def test_dm_policy_explicit(self):
        adapter = self._make(app_id="a", client_secret="b", dm_policy="allowlist")
        assert adapter._dm_policy == "allowlist"

    def test_group_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._group_policy == "open"

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
        from gateway.platforms.qqbot.core.audio import is_voice_content_type
        return is_voice_content_type(content_type, filename)

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
    def test_stt_blocks_unsafe_download_url(self):
        from gateway.platforms.qqbot.core.attachment_processor import AttachmentDownloader

        http_client = mock.AsyncMock()
        downloader = AttachmentDownloader(
            http_client=http_client,
            cache_dir="/tmp",
            media_headers_fn=lambda: {},
            log_tag="test",
        )

        with mock.patch.object(AttachmentDownloader, "_is_safe_url", return_value=False):
            result = asyncio.run(
                downloader.download_audio(
                    "http://127.0.0.1/voice.silk",
                    "voice.silk",
                )
            )

        assert result is None
        http_client.get.assert_not_called()

    def test_connect_uses_redirect_guard_hook(self):
        from gateway.platforms.qqbot import QQAdapter, _ssrf_redirect_guard

        fake_client = mock.MagicMock()
        fake_client.aclose = mock.AsyncMock()
        with mock.patch("gateway.platforms.qqbot.adapter.httpx.AsyncClient", return_value=fake_client) as async_client_cls:
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            adapter._api.ensure_token = mock.AsyncMock(side_effect=RuntimeError("stop after client creation"))

            connected = asyncio.run(adapter.connect())

        assert connected is False
        assert async_client_cls.call_count == 1
        kwargs = async_client_cls.call_args.kwargs
        assert kwargs.get("follow_redirects") is True
        assert kwargs.get("event_hooks", {}).get("response") == [_ssrf_redirect_guard]

# ---------------------------------------------------------------------------
# _strip_at_mention
# ---------------------------------------------------------------------------

class TestStripAtMention:
    def _fn(self, content):
        from gateway.platforms.qqbot.core.event_parser import _strip_at_mention
        return _strip_at_mention(content)

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


# ---------------------------------------------------------------------------
# _is_dm_allowed
# ---------------------------------------------------------------------------

class TestDmAllowed:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_open_policy(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="open")
        assert adapter._is_dm_allowed("any_user") is True

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


# ---------------------------------------------------------------------------
# _resolve_stt_config
# ---------------------------------------------------------------------------

class TestResolveSTTConfig:
    def test_no_config(self):
        from gateway.platforms.qqbot.core.audio import resolve_stt_config
        with mock.patch.dict(os.environ, {}, clear=True):
            assert resolve_stt_config({}) is None

    def test_env_config(self):
        from gateway.platforms.qqbot.core.audio import resolve_stt_config
        with mock.patch.dict(os.environ, {
            "QQ_STT_API_KEY": "key123",
            "QQ_STT_BASE_URL": "https://example.com/v1",
            "QQ_STT_MODEL": "my-model",
        }, clear=True):
            cfg = resolve_stt_config({})
            assert cfg is not None
            assert cfg["api_key"] == "key123"
            assert cfg["base_url"] == "https://example.com/v1"
            assert cfg["model"] == "my-model"

    def test_extra_config(self):
        from gateway.platforms.qqbot.core.audio import resolve_stt_config
        extra = {"stt": {
            "baseUrl": "https://custom.api/v4",
            "apiKey": "sk_extra",
            "model": "glm-asr",
        }}
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = resolve_stt_config(extra)
            assert cfg is not None
            assert cfg["base_url"] == "https://custom.api/v4"
            assert cfg["api_key"] == "sk_extra"
            assert cfg["model"] == "glm-asr"


# ---------------------------------------------------------------------------
# _detect_message_type
# ---------------------------------------------------------------------------

class TestDetectMessageType:
    def _fn(self, media_urls, media_types):
        from gateway.platforms.qqbot.adapter import _detect_message_type
        return _detect_message_type(media_urls, media_types)

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
# WSCallbacks wiring — adapter state (session_id, last_seq, heartbeat_interval)
# updated via injected callbacks
# ---------------------------------------------------------------------------

class TestAdapterCallbacks:
    """Verify that WSCallbacks are wired so adapter state updates correctly."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_set_session_updates_state(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._set_session("sess_abc", 42)
        assert adapter._session_id == "sess_abc"
        assert adapter._last_seq == 42

    def test_set_session_preserves_seq_on_none(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._last_seq = 10
        adapter._set_session("sess_x", None)
        assert adapter._last_seq == 10

    def test_heartbeat_interval_updated_via_lambda(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # The lambda in _build_ws_callbacks calls setattr
        cbs = adapter._ws_manager._cb
        cbs.set_heartbeat_interval(30.5)
        assert adapter._heartbeat_interval == 30.5

    def test_mark_connected_disconnected(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._mark_connected()
        assert adapter.is_connected
        adapter._mark_disconnected()
        assert not adapter.is_connected


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def _fn(self, raw):
        from gateway.platforms.qqbot.core.websocket import QQWebSocket
        return QQWebSocket._parse_json(raw)

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
        msg = adapter._api.build_text_body("hello world", markdown=False)
        body = msg.to_dict()
        assert body["msg_type"] == 0  # MSG_TYPE_TEXT
        assert body["content"] == "hello world"

    def test_markdown_text(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=True)
        msg = adapter._api.build_text_body("**bold** text", markdown=True)
        body = msg.to_dict()
        assert body["msg_type"] == 2  # MSG_TYPE_MARKDOWN
        assert body["markdown"]["content"] == "**bold** text"

    def test_truncation(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        long_text = "x" * 10000
        msg = adapter._api.build_text_body(long_text, markdown=False, max_length=adapter.MAX_MESSAGE_LENGTH)
        body = msg.to_dict()
        assert len(body["content"]) == adapter.MAX_MESSAGE_LENGTH

    def test_empty_string(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        msg = adapter._api.build_text_body("", markdown=False)
        body = msg.to_dict()
        # empty content is omitted or empty string
        assert body.get("content", "") == ""

    def test_reply_to(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        msg = adapter._api.build_text_body("reply text", reply_to="msg_123", markdown=False)
        body = msg.to_dict()
        assert body.get("msg_id") == "msg_123"


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
        adapter._api.request = mock.AsyncMock(return_value={"id": "msg_123"})
        adapter._api.ensure_token = mock.AsyncMock()
        adapter._RECONNECT_POLL_INTERVAL = 0.1
        adapter._RECONNECT_WAIT_SECONDS = 5.0

        # Schedule reconnection after a short delay
        async def reconnect_after_delay():
            await asyncio.sleep(0.3)
            adapter._running = True

        asyncio.get_event_loop().create_task(reconnect_after_delay())

        result = await adapter.send("test_openid", "Hello, world!")
        assert result.success

    @pytest.mark.asyncio
    async def test_send_returns_error_without_http_client(self):
        """send() should return failure when HTTP client is not set up."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        result = await adapter.send("test_openid", "Hello, world!")
        assert not result.success

    @pytest.mark.asyncio
    async def test_send_succeeds_immediately_when_connected(self):
        """send() should not wait when already connected."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        adapter._http_client = mock.MagicMock()
        adapter._api.request = mock.AsyncMock(return_value={"id": "msg_immediate"})

        result = await adapter.send("test_openid", "Hello!")
        assert result.success

    @pytest.mark.asyncio
    async def test_send_media_returns_error_without_http_client(self):
        """_send_media should return failure when HTTP client is not set up."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        result = await adapter._send_media("test_openid", "http://example.com/img.jpg", 1, None, None)
        assert not result.success
