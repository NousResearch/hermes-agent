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
