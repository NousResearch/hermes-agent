"""Tests for the WeCom platform adapter."""

import base64
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


class TestWeComRequirements:
    def test_returns_false_without_aiohttp(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", True)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_false_without_httpx(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", False)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_true_when_available(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wecom.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.wecom.HTTPX_AVAILABLE", True)
        from gateway.platforms.wecom import check_wecom_requirements

        assert check_wecom_requirements() is True


class TestWeComAdapterInit:
    def test_reads_config_from_extra(self):
        from gateway.platforms.wecom import WeComAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "bot_id": "cfg-bot",
                "secret": "cfg-secret",
                "websocket_url": "wss://custom.wecom.example/ws",
                "group_policy": "allowlist",
                "group_allow_from": ["group-1"],
            },
        )
        adapter = WeComAdapter(config)

        assert adapter._bot_id == "cfg-bot"
        assert adapter._secret == "cfg-secret"
        assert adapter._ws_url == "wss://custom.wecom.example/ws"
        assert adapter._group_policy == "allowlist"
        assert adapter._group_allow_from == ["group-1"]

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "env-bot")
        monkeypatch.setenv("WECOM_SECRET", "env-secret")
        monkeypatch.setenv("WECOM_WEBSOCKET_URL", "wss://env.example/ws")
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter._bot_id == "env-bot"
        assert adapter._secret == "env-secret"
        assert adapter._ws_url == "wss://env.example/ws"


class TestWeComConnect:
    @pytest.mark.asyncio
    async def test_connect_records_missing_credentials(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import WeComAdapter

        monkeypatch.setattr(wecom_module, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(wecom_module, "HTTPX_AVAILABLE", True)

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        success = await adapter.connect()

        assert success is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "wecom_missing_credentials"
        assert "WECOM_BOT_ID" in (adapter.fatal_error_message or "")

    @pytest.mark.asyncio
    async def test_connect_records_handshake_failure_details(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import WeComAdapter

        class DummyClient:
            async def aclose(self):
                return None

        monkeypatch.setattr(wecom_module, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(wecom_module, "HTTPX_AVAILABLE", True)
        monkeypatch.setattr(
            wecom_module,
            "httpx",
            SimpleNamespace(AsyncClient=lambda **kwargs: DummyClient()),
        )

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"bot_id": "bot-1", "secret": "secret-1"})
        )
        adapter._open_connection = AsyncMock(side_effect=RuntimeError("invalid secret (errcode=40013)"))

        success = await adapter.connect()

        assert success is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "wecom_connect_error"
        assert "invalid secret" in (adapter.fatal_error_message or "")


class TestWeComReplyMode:
    @pytest.mark.asyncio
    async def test_send_uses_passive_reply_stream_when_reply_context_exists(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-1"] = "req-1"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-1"}, "errcode": 0}
        )

        result = await adapter.send("chat-123", "hello from reply", reply_to="msg-1")

        assert result.success is True
        adapter._send_reply_request.assert_awaited_once()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "req-1"
        assert args[1]["msgtype"] == "stream"
        assert args[1]["stream"]["finish"] is True
        assert args[1]["stream"]["content"] == "hello from reply"

    @pytest.mark.asyncio
    async def test_send_image_file_uses_passive_reply_media_when_reply_context_exists(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-1"] = "req-1"
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"image-bytes",
                "content_type": "image/png",
                "file_name": "demo.png",
                "detected_type": "image",
                "final_type": "image",
                "rejected": False,
                "reject_reason": None,
                "downgraded": False,
                "downgrade_note": None,
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "image"})
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-1"}, "errcode": 0}
        )

        result = await adapter.send_image_file("chat-123", "/tmp/demo.png", reply_to="msg-1")

        assert result.success is True
        adapter._send_reply_request.assert_awaited_once()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "req-1"
        assert args[1] == {"msgtype": "image", "image": {"media_id": "media-1"}}


class TestExtractText:
    def test_extracts_plain_text(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "text",
            "text": {"content": "  hello world  "},
        }
        text, reply_text = WeComAdapter._extract_text(body)
        assert text == "hello world"
        assert reply_text is None

    def test_extracts_mixed_text(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "mixed",
            "mixed": {
                "msg_item": [
                    {"msgtype": "text", "text": {"content": "part1"}},
                    {"msgtype": "image", "image": {"url": "https://example.com/x.png"}},
                    {"msgtype": "text", "text": {"content": "part2"}},
                ]
            },
        }
        text, _reply_text = WeComAdapter._extract_text(body)
        assert text == "part1\npart2"

    def test_extracts_voice_and_quote(self):
        from gateway.platforms.wecom import WeComAdapter

        body = {
            "msgtype": "voice",
            "voice": {"content": "spoken text"},
            "quote": {"msgtype": "text", "text": {"content": "quoted"}},
        }
        text, reply_text = WeComAdapter._extract_text(body)
        assert text == "spoken text"
        assert reply_text == "quoted"


class TestCallbackDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cmd", ["aibot_msg_callback", "aibot_callback"])
    async def test_dispatch_accepts_new_and_legacy_callback_cmds(self, cmd):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()

        await adapter._dispatch_payload({"cmd": cmd, "headers": {"req_id": "req-1"}, "body": {}})

        adapter._on_message.assert_awaited_once()


class TestPolicyHelpers:
    def test_dm_allowlist(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"dm_policy": "allowlist", "allow_from": ["user-1"]})
        )
        assert adapter._is_dm_allowed("user-1") is True
        assert adapter._is_dm_allowed("user-2") is False

    def test_group_allowlist_and_per_group_sender_allowlist(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "group_policy": "allowlist",
                    "group_allow_from": ["group-1"],
                    "groups": {"group-1": {"allow_from": ["user-1"]}},
                },
            )
        )

        assert adapter._is_group_allowed("group-1", "user-1") is True
        assert adapter._is_group_allowed("group-1", "user-2") is False
        assert adapter._is_group_allowed("group-2", "user-1") is False


class TestMediaHelpers:
    def test_detect_wecom_media_type(self):
        from gateway.platforms.wecom import WeComAdapter

        assert WeComAdapter._detect_wecom_media_type("image/png") == "image"
        assert WeComAdapter._detect_wecom_media_type("video/mp4") == "video"
        assert WeComAdapter._detect_wecom_media_type("audio/amr") == "voice"
        assert WeComAdapter._detect_wecom_media_type("application/pdf") == "file"

    def test_voice_non_amr_downgrades_to_file(self):
        from gateway.platforms.wecom import WeComAdapter

        result = WeComAdapter._apply_file_size_limits(128, "voice", "audio/mpeg")

        assert result["final_type"] == "file"
        assert result["downgraded"] is True
        assert "AMR" in (result["downgrade_note"] or "")

    def test_oversized_file_is_rejected(self):
        from gateway.platforms.wecom import ABSOLUTE_MAX_BYTES, WeComAdapter

        result = WeComAdapter._apply_file_size_limits(ABSOLUTE_MAX_BYTES + 1, "file", "application/pdf")

        assert result["rejected"] is True
        assert "20MB" in (result["reject_reason"] or "")

    def test_decrypt_file_bytes_round_trip(self):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from gateway.platforms.wecom import WeComAdapter

        plaintext = b"wecom-secret"
        key = os.urandom(32)
        pad_len = 32 - (len(plaintext) % 32)
        padded = plaintext + bytes([pad_len]) * pad_len
        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()

        decrypted = WeComAdapter._decrypt_file_bytes(encrypted, base64.b64encode(key).decode("ascii"))

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_load_outbound_media_rejects_placeholder_path(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        with pytest.raises(ValueError, match="placeholder was not replaced"):
            await adapter._load_outbound_media("<path>")


class TestMediaUpload:
    @pytest.mark.asyncio
    async def test_upload_media_bytes_uses_sdk_sequence(self, monkeypatch):
        import gateway.platforms.wecom as wecom_module
        from gateway.platforms.wecom import (
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_FINISH,
            APP_CMD_UPLOAD_MEDIA_INIT,
            WeComAdapter,
        )

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        calls = []

        async def fake_send_request(cmd, body, timeout=0):
            calls.append((cmd, body))
            if cmd == APP_CMD_UPLOAD_MEDIA_INIT:
                return {"errcode": 0, "body": {"upload_id": "upload-1"}}
            if cmd == APP_CMD_UPLOAD_MEDIA_CHUNK:
                return {"errcode": 0}
            if cmd == APP_CMD_UPLOAD_MEDIA_FINISH:
                return {
                    "errcode": 0,
                    "body": {
                        "media_id": "media-1",
                        "type": "file",
                        "created_at": "2026-03-18T00:00:00Z",
                    },
                }
            raise AssertionError(f"unexpected cmd {cmd}")

        monkeypatch.setattr(wecom_module, "UPLOAD_CHUNK_SIZE", 4)
        adapter._send_request = fake_send_request

        result = await adapter._upload_media_bytes(b"abcdefghij", "file", "demo.bin")

        assert result["media_id"] == "media-1"
        assert [cmd for cmd, _body in calls] == [
            APP_CMD_UPLOAD_MEDIA_INIT,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_CHUNK,
            APP_CMD_UPLOAD_MEDIA_FINISH,
        ]
        assert calls[1][1]["chunk_index"] == 0
        assert calls[2][1]["chunk_index"] == 1
        assert calls[3][1]["chunk_index"] == 2

    @pytest.mark.asyncio
    @patch("tools.url_safety.is_safe_url", return_value=True)
    async def test_download_remote_bytes_rejects_large_content_length(self, _mock_safe):
        from gateway.platforms.wecom import WeComAdapter

        class FakeResponse:
            headers = {"content-length": "10"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"abc"

        class FakeClient:
            def stream(self, method, url, headers=None):
                return FakeResponse()

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._http_client = FakeClient()

        with pytest.raises(ValueError, match="exceeds WeCom limit"):
            await adapter._download_remote_bytes("https://example.com/file.bin", max_bytes=4)

    @pytest.mark.asyncio
    async def test_cache_media_decrypts_url_payload_before_writing(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        plaintext = b"secret document bytes"
        key = os.urandom(32)
        pad_len = 32 - (len(plaintext) % 32)
        padded = plaintext + bytes([pad_len]) * pad_len

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        adapter._download_remote_bytes = AsyncMock(
            return_value=(
                encrypted,
                {
                    "content-type": "application/octet-stream",
                    "content-disposition": 'attachment; filename="secret.bin"',
                },
            )
        )

        cached = await adapter._cache_media(
            "file",
            {
                "url": "https://example.com/secret.bin",
                "aeskey": base64.b64encode(key).decode("ascii"),
            },
        )

        assert cached is not None
        cached_path, content_type = cached
        assert Path(cached_path).read_bytes() == plaintext
        assert content_type == "application/octet-stream"


class TestSend:
    @pytest.mark.asyncio
    async def test_send_uses_proactive_payload(self):
        from gateway.platforms.wecom import APP_CMD_SEND, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(return_value={"headers": {"req_id": "req-1"}, "errcode": 0})

        result = await adapter.send("chat-123", "Hello WeCom")

        assert result.success is True
        adapter._send_request.assert_awaited_once_with(
            APP_CMD_SEND,
            {
                "chatid": "chat-123",
                "msgtype": "markdown",
                "markdown": {"content": "Hello WeCom"},
            },
        )

    @pytest.mark.asyncio
    async def test_send_reports_wecom_errors(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(return_value={"errcode": 40001, "errmsg": "bad request"})

        result = await adapter.send("chat-123", "Hello WeCom")

        assert result.success is False
        assert "40001" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_image_falls_back_to_text_for_remote_url(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_media_source = AsyncMock(return_value=SendResult(success=False, error="upload failed"))
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_image("chat-123", "https://example.com/demo.png", caption="demo")

        assert result.success is True
        adapter.send.assert_awaited_once_with(chat_id="chat-123", content="demo\nhttps://example.com/demo.png", reply_to=None)

    @pytest.mark.asyncio
    async def test_send_voice_sends_caption_and_downgrade_note(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"voice-bytes",
                "content_type": "audio/mpeg",
                "file_name": "voice.mp3",
                "detected_type": "voice",
                "final_type": "file",
                "rejected": False,
                "reject_reason": None,
                "downgraded": True,
                "downgrade_note": "语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "file"})
        adapter._send_media_message = AsyncMock(return_value={"headers": {"req_id": "req-media"}, "errcode": 0})
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_voice("chat-123", "/tmp/voice.mp3", caption="listen")

        assert result.success is True
        adapter._send_media_message.assert_awaited_once_with("chat-123", "file", "media-1")
        assert adapter.send.await_count == 2
        adapter.send.assert_any_await(chat_id="chat-123", content="listen", reply_to=None)
        adapter.send.assert_any_await(
            chat_id="chat-123",
            content="ℹ️ 语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            reply_to=None,
        )


class TestInboundMessages:
    @pytest.mark.asyncio
    async def test_on_message_builds_event(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0  # disable batching for tests
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=(["/tmp/test.png"], ["image/png"]))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hello"},
            },
        }

        await adapter._on_message(payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello"
        assert event.source.chat_id == "group-1"
        assert event.source.user_id == "user-1"
        assert event.media_urls == ["/tmp/test.png"]
        assert event.media_types == ["image/png"]

    @pytest.mark.asyncio
    async def test_on_message_preserves_quote_context(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0  # disable batching for tests
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "follow up"},
                "quote": {"msgtype": "text", "text": {"content": "quoted message"}},
            },
        }

        await adapter._on_message(payload)

        event = adapter.handle_message.await_args.args[0]
        assert event.reply_to_text == "quoted message"
        assert event.reply_to_message_id == "quote:msg-1"

    @pytest.mark.asyncio
    async def test_on_message_respects_group_policy(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-allowed"]},
            )
        )
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-blocked",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hello"},
            },
        }

        await adapter._on_message(payload)
        adapter.handle_message.assert_not_awaited()


class TestPlatformEnum:
    def test_wecom_in_platform_enum(self):
        assert Platform.WECOM.value == "wecom"
