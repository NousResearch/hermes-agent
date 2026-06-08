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
            assert parsed[1] in ("allow-once", "allow-always", "deny")


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
                "resolved": {"button_data": "approve:s:deny", "button_id": "deny"},
            },
        })

        assert len(ack_calls) == 1
        assert ack_calls[0][0] == "i-1"
        assert len(received) == 1
        assert received[0].button_data == "approve:s:deny"
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
            "data": {"resolved": {"button_data": "approve:s:deny"}},
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
            "data": {"resolved": {"button_data": "approve:s:deny"}},
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
            "data": {"resolved": {"button_data": "approve:s:deny"}},
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
                "data": {"resolved": {"button_data": "approve:sess-abc:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("sess-abc", "once", False)]

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
                "data": {"resolved": {"button_data": "approve:s:allow-always"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("s", "always", False)]

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
                "data": {"resolved": {"button_data": "approve:s:deny"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("s", "deny", False)]

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
                "data": {"resolved": {"button_data": "approve:s:deny"}},
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
