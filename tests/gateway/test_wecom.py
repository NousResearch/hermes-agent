"""Tests for the WeCom platform adapter."""

import asyncio
import base64
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


class TestWeComRequirements:
    def test_returns_false_without_aiohttp(self, monkeypatch):
        monkeypatch.setattr("plugins.platforms.wecom.adapter.AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr("plugins.platforms.wecom.adapter.HTTPX_AVAILABLE", True)
        from plugins.platforms.wecom.adapter import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_false_without_httpx(self, monkeypatch):
        monkeypatch.setattr("plugins.platforms.wecom.adapter.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("plugins.platforms.wecom.adapter.HTTPX_AVAILABLE", False)
        from plugins.platforms.wecom.adapter import check_wecom_requirements

        assert check_wecom_requirements() is False

    def test_returns_true_when_available(self, monkeypatch):
        monkeypatch.setattr("plugins.platforms.wecom.adapter.AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr("plugins.platforms.wecom.adapter.HTTPX_AVAILABLE", True)
        from plugins.platforms.wecom.adapter import check_wecom_requirements

        assert check_wecom_requirements() is True


class TestWeComAdapterInit:
    def test_declares_non_editable_message_capability(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        assert WeComAdapter.SUPPORTS_MESSAGE_EDITING is False

    def test_reads_config_from_extra(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter._bot_id == "env-bot"
        assert adapter._secret == "env-secret"
        assert adapter._ws_url == "wss://env.example/ws"


class TestWeComConnect:
    @pytest.mark.asyncio
    async def test_connect_records_missing_credentials(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_module
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        import plugins.platforms.wecom.adapter as wecom_module
        from plugins.platforms.wecom.adapter import WeComAdapter

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


class TestWeComQrScan:
    @patch("plugins.platforms.wecom.adapter.time")
    @patch("plugins.platforms.wecom.adapter.json.loads")
    @patch("plugins.platforms.wecom.adapter.logger")
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_qr_scan_timeout_uses_monotonic_clock(
        self,
        mock_request,
        mock_urlopen,
        _mock_logger,
        mock_json_loads,
        mock_time,
    ):
        from plugins.platforms.wecom.adapter import qr_scan_for_bot_info

        generate_resp = MagicMock()
        generate_resp.read.return_value = b'{"data":{"scode":"abc","auth_url":"https://example.com/qr"}}'
        generate_resp.__enter__.return_value = generate_resp
        generate_resp.__exit__.return_value = False

        poll_resp = MagicMock()
        poll_resp.read.return_value = b'{"data":{"status":"pending"}}'
        poll_resp.__enter__.return_value = poll_resp
        poll_resp.__exit__.return_value = False

        mock_urlopen.side_effect = [generate_resp, poll_resp]
        mock_json_loads.side_effect = [
            {"data": {"scode": "abc", "auth_url": "https://example.com/qr"}},
            {"data": {"status": "pending"}},
        ]
        mock_time.monotonic.side_effect = [1000, 1000.2, 1001.1]
        mock_time.time.side_effect = [1000, 900, 901, 902]
        mock_time.sleep = MagicMock()

        with patch("builtins.print"), patch.dict("sys.modules", {"qrcode": None}):
            result = qr_scan_for_bot_info(timeout_seconds=1)

        assert result is None
        assert mock_urlopen.call_count == 2


class TestWeComReplyMode:
    @pytest.mark.asyncio
    async def test_send_uses_passive_reply_markdown_when_reply_context_exists(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        # msgtype: stream triggers WeCom errcode 600039 on many mobile clients
        # (unsupported type). Markdown renders everywhere.
        assert args[1]["msgtype"] == "markdown"
        assert args[1]["markdown"]["content"] == "hello from reply"

    @pytest.mark.asyncio
    async def test_send_image_file_uses_passive_reply_media_when_reply_context_exists(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        body = {
            "msgtype": "text",
            "text": {"content": "  hello world  "},
        }
        text, reply_text = WeComAdapter._extract_text(body)
        assert text == "hello world"
        assert reply_text is None

    def test_extracts_mixed_text(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()

        await adapter._dispatch_payload({"cmd": cmd, "headers": {"req_id": "req-1"}, "body": {}})

        adapter._on_message.assert_awaited_once()


class TestPolicyHelpers:
    def test_dm_allowlist(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"dm_policy": "allowlist", "allow_from": ["user-1"]})
        )
        assert adapter._is_dm_allowed("user-1") is True
        assert adapter._is_dm_allowed("user-2") is False

    def test_dm_allowlist_honors_env_only_allowed_users(self, monkeypatch):
        """Env-only setup (WECOM_DM_POLICY + WECOM_ALLOWED_USERS, no config
        ``extra``) must populate the DM allowlist. Otherwise ``dm_policy:
        allowlist`` runs with an empty allowlist and drops every listed user
        at intake — the documented env vars become no-ops."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_DM_POLICY", "allowlist")
        monkeypatch.setenv("WECOM_ALLOWED_USERS", "user-1, user-2")

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        assert adapter._dm_policy == "allowlist"
        assert adapter._allow_from == ["user-1", "user-2"]
        assert adapter._is_dm_allowed("user-1") is True
        assert adapter._is_dm_allowed("user-2") is True
        assert adapter._is_dm_allowed("stranger") is False

    def test_dm_allowlist_extra_takes_precedence_over_env(self, monkeypatch):
        """Config ``extra`` wins over the env fallback, so an explicit
        allowlist is never silently widened by a stray WECOM_ALLOWED_USERS."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_ALLOWED_USERS", "env-user")

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"dm_policy": "allowlist", "allow_from": ["cfg-user"]})
        )

        assert adapter._allow_from == ["cfg-user"]
        assert adapter._is_dm_allowed("cfg-user") is True
        assert adapter._is_dm_allowed("env-user") is False

    def test_group_allowlist_and_per_group_sender_allowlist(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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

    def test_pairing_group_policy_blocks_without_explicit_group_allow_from(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"group_policy": "pairing"})
        )

        assert adapter._is_group_allowed("group-1", "user-1") is False

    def test_pairing_dm_policy_strict_auth_denies_unknown(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"dm_policy": "pairing"}))
        assert adapter._is_dm_allowed("user-1") is False
        assert adapter._is_dm_intake_allowed("user-1") is True

class TestMediaHelpers:
    def test_detect_wecom_media_type(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        assert WeComAdapter._detect_wecom_media_type("image/png") == "image"
        assert WeComAdapter._detect_wecom_media_type("video/mp4") == "video"
        assert WeComAdapter._detect_wecom_media_type("audio/amr") == "voice"
        assert WeComAdapter._detect_wecom_media_type("application/pdf") == "file"

    def test_voice_non_amr_downgrades_to_file(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        result = WeComAdapter._apply_file_size_limits(128, "voice", "audio/mpeg")

        assert result["final_type"] == "file"
        assert result["downgraded"] is True
        assert "AMR" in (result["downgrade_note"] or "")

    def test_oversized_file_is_rejected(self):
        from plugins.platforms.wecom.adapter import ABSOLUTE_MAX_BYTES, WeComAdapter

        result = WeComAdapter._apply_file_size_limits(ABSOLUTE_MAX_BYTES + 1, "file", "application/pdf")

        assert result["rejected"] is True
        assert "20MB" in (result["reject_reason"] or "")

    def test_decrypt_file_bytes_round_trip(self):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        with pytest.raises(ValueError, match="placeholder was not replaced"):
            await adapter._load_outbound_media("<path>")


class TestMediaUpload:
    @pytest.mark.asyncio
    async def test_upload_media_bytes_uses_sdk_sequence(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_module
        from plugins.platforms.wecom.adapter import (
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
        from plugins.platforms.wecom.adapter import WeComAdapter

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
    async def test_download_remote_bytes_blocks_connect_time_rebind(self, monkeypatch):
        import httpcore
        from httpcore._backends.auto import AutoBackend
        from plugins.platforms.wecom.adapter import WeComAdapter
        from tools.url_safety import SSRFConnectionBlocked

        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.delenv(proxy_var, raising=False)

        answers = iter(("93.184.216.34", "169.254.169.254"))

        def fake_getaddrinfo(_host, port, *_args, **_kwargs):
            ip = next(answers)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
            ]

        connect_attempts = []

        async def fake_connect_tcp(
            _self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            connect_attempts.append((host, port))
            raise httpcore.ConnectError("stop before network")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(AutoBackend, "connect_tcp", fake_connect_tcp)

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        with pytest.raises(SSRFConnectionBlocked):
            await adapter._download_remote_bytes(
                "http://rebind.example/file.bin", max_bytes=1024
            )

        assert connect_attempts == []

    @pytest.mark.asyncio
    async def test_cache_media_decrypts_url_payload_before_writing(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import APP_CMD_SEND, WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(return_value={"errcode": 40001, "errmsg": "bad request"})

        result = await adapter.send("chat-123", "Hello WeCom")

        assert result.success is False
        assert "40001" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_image_falls_back_to_text_for_remote_url(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_media_source = AsyncMock(return_value=SendResult(success=False, error="upload failed"))
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_image("chat-123", "https://example.com/demo.png", caption="demo")

        assert result.success is True
        adapter.send.assert_awaited_once_with(chat_id="chat-123", content="demo\nhttps://example.com/demo.png", reply_to=None)

    @pytest.mark.asyncio
    async def test_send_voice_sends_caption_and_downgrade_note(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-1"]},
            )
        )
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
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-1"]},
            )
        )
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
        from plugins.platforms.wecom.adapter import WeComAdapter

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


class TestWeComZombieSessionFix:
    """Tests for PR #11572 — device_id, markdown reply, group req_id fallback."""

    def test_adapter_generates_stable_device_id_per_instance(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert isinstance(adapter._device_id, str)
        assert len(adapter._device_id) > 0
        # Second snapshot on the same adapter must be identical — only a fresh
        # adapter instance should get a new device_id (one-per-reconnect is the
        # zombie-session footgun we're fixing).
        assert adapter._device_id == adapter._device_id

    def test_different_adapter_instances_get_distinct_device_ids(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        a = WeComAdapter(PlatformConfig(enabled=True))
        b = WeComAdapter(PlatformConfig(enabled=True))
        assert a._device_id != b._device_id

    @pytest.mark.asyncio
    async def test_open_connection_includes_device_id_in_subscribe(self):
        from plugins.platforms.wecom.adapter import APP_CMD_SUBSCRIBE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._bot_id = "test-bot"
        adapter._secret = "test-secret"

        sent_payloads = []

        class _FakeWS:
            closed = False

            async def send_json(self, payload):
                sent_payloads.append(payload)

            async def close(self):
                return None

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            async def ws_connect(self, *args, **kwargs):
                return _FakeWS()

            async def close(self):
                return None

        async def _fake_cleanup():
            return None

        async def _fake_handshake(req_id):
            return {"errcode": 0, "headers": {"req_id": req_id}}

        adapter._cleanup_ws = _fake_cleanup
        adapter._wait_for_handshake = _fake_handshake

        with patch("plugins.platforms.wecom.adapter.aiohttp.ClientSession", _FakeSession):
            await adapter._open_connection()

        assert len(sent_payloads) == 1
        subscribe = sent_payloads[0]
        assert subscribe["cmd"] == APP_CMD_SUBSCRIBE
        assert subscribe["body"]["bot_id"] == "test-bot"
        assert subscribe["body"]["secret"] == "test-secret"
        assert subscribe["body"]["device_id"] == adapter._device_id

    @pytest.mark.asyncio
    async def test_on_message_caches_last_req_id_per_chat(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-1"]},
            )
        )
        adapter._text_batch_delay_seconds = 0
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-abc"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hi"},
            },
        }

        await adapter._on_message(payload)
        assert adapter._last_chat_req_ids["group-1"] == "req-abc"

    @pytest.mark.asyncio
    async def test_on_message_does_not_cache_blocked_sender_req_id(self):
        """Blocked chats shouldn't populate the proactive-send fallback cache."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-ok"]},
            )
        )
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-abc"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-blocked",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hi"},
            },
        }

        await adapter._on_message(payload)
        adapter.handle_message.assert_not_awaited()
        assert "group-blocked" not in adapter._last_chat_req_ids

    def test_remember_chat_req_id_is_bounded(self):
        from plugins.platforms.wecom.adapter import DEDUP_MAX_SIZE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        for i in range(DEDUP_MAX_SIZE + 50):
            adapter._remember_chat_req_id(f"chat-{i}", f"req-{i}")
        assert len(adapter._last_chat_req_ids) <= DEDUP_MAX_SIZE
        # The most recently remembered chat must still be present.
        latest = f"chat-{DEDUP_MAX_SIZE + 49}"
        assert adapter._last_chat_req_ids[latest] == f"req-{DEDUP_MAX_SIZE + 49}"

    def test_remember_chat_req_id_ignores_empty_values(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._remember_chat_req_id("", "req-1")
        adapter._remember_chat_req_id("chat-1", "")
        adapter._remember_chat_req_id("   ", "   ")
        assert adapter._last_chat_req_ids == {}

    @pytest.mark.asyncio
    async def test_proactive_group_send_falls_back_to_cached_req_id(self):
        """Sending into a group without reply_to should use the last cached
        req_id via APP_CMD_RESPONSE — WeCom AI Bots cannot initiate APP_CMD_SEND
        in group chats (errcode 600039)."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["group-1"] = "inbound-req-42"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "inbound-req-42"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(
            return_value={"headers": {"req_id": "new"}, "errcode": 0}
        )

        result = await adapter.send("group-1", "ping", reply_to=None)

        assert result.success is True
        # Must route through reply (APP_CMD_RESPONSE), not proactive send.
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "inbound-req-42"
        assert args[1]["msgtype"] == "markdown"
        assert args[1]["markdown"]["content"] == "ping"

    @pytest.mark.asyncio
    async def test_proactive_send_without_cached_req_id_uses_app_cmd_send(self):
        """When we have no prior req_id (fresh DM target), APP_CMD_SEND is used."""
        from plugins.platforms.wecom.adapter import APP_CMD_SEND, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_request = AsyncMock(
            return_value={"headers": {"req_id": "new"}, "errcode": 0}
        )

        result = await adapter.send("fresh-dm-chat", "ping", reply_to=None)

        assert result.success is True
        adapter._send_request.assert_awaited_once()
        cmd = adapter._send_request.await_args.args[0]
        assert cmd == APP_CMD_SEND



class TestTextBatchFlushRace:
    """Regression tests for the cancel-delivery race in _flush_text_batch.

    When asyncio.sleep() fires and Task.cancel() is called before the task
    runs, CPython sets _must_cancel but cannot cancel the already-done sleep
    future.  CancelledError is then delivered at the *next* await
    (handle_message), after the task has already popped the event — the
    superseding task sees an empty batch and silently drops the message.
    The fix adds a synchronous task-registry check between the sleep and
    the pop so a superseded task returns before touching the event.
    """

    @pytest.mark.asyncio
    async def test_superseded_task_does_not_pop_or_process_event(self):
        """A flush task that has been superseded must leave the event in the
        batch dict for the new task to handle."""
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="hello", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        # Create T1 and register it.
        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # Simulate T2 superseding T1 before T1 wakes from sleep.
        t2 = asyncio.create_task(asyncio.sleep(9999))
        adapter._pending_text_batch_tasks[key] = t2

        # Yield long enough for T1's sleep(0) to complete and T1 to run.
        await asyncio.sleep(0.05)

        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass

        # T1 must have returned without processing or removing the event.
        assert handle_calls == [], "superseded task must not call handle_message"
        assert adapter._pending_text_batches.get(key) is event, (
            "superseded task must not pop the event"
        )

    @pytest.mark.asyncio
    async def test_active_task_processes_event_normally(self):
        """When the task is not superseded it must still process the event."""
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="world", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # No superseding task — T1 should process normally.
        await asyncio.sleep(0.05)

        assert handle_calls == [event], "active task must call handle_message"
        assert adapter._pending_text_batches.get(key) is None, (
            "active task must pop the event after processing"
        )


# === NATIVE STREAMING (msgtype: stream) ===


class TestWeComNativeStreamingCapability:
    """SUPPORTS_NATIVE_STREAMING + supports_native_streaming() probe."""

    def test_class_attribute_declares_native_streaming(self):
        from gateway.platforms.wecom import WeComAdapter

        assert WeComAdapter.SUPPORTS_NATIVE_STREAMING is True

    def test_supports_native_streaming_returns_true_for_dm(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter.supports_native_streaming(chat_type="dm") is True

    def test_supports_native_streaming_returns_true_for_group(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter.supports_native_streaming(chat_type="group") is True

    def test_max_stream_content_length_is_20480(self):
        from gateway.platforms.wecom import (
            MAX_STREAM_CONTENT_LENGTH, WeComAdapter,
        )

        assert MAX_STREAM_CONTENT_LENGTH == 20480
        assert WeComAdapter.MAX_STREAM_CONTENT_LENGTH == 20480

    def test_stream_expired_errcode_constant(self):
        from gateway.platforms.wecom import STREAM_EXPIRED_ERRCODE

        assert STREAM_EXPIRED_ERRCODE == 846608

# === STREAM TESTS PLACEHOLDER ===


class TestResolveStreamReqId:
    """`_resolve_stream_req_id` precedence: reply_to → cached chat → None."""

    def test_prefers_explicit_reply_to(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-123"] = "explicit-req"
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", "msg-123") == "explicit-req"

    def test_falls_back_to_cached_chat_req_id(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", reply_to=None) == "cached-req"

    def test_returns_none_when_no_anchor(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter._resolve_stream_req_id("unknown-chat", None) is None

    def test_quoted_reply_to_falls_through_to_chat_cache(self):
        """``quote:msg-id`` (quote-context marker) is not a real reply anchor."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", "quote:m-1") == "cached-req"


# === LIFECYCLE TESTS PLACEHOLDER ===


class TestSendStreamFrame:
    """`send_stream_frame` lifecycle: init → cumulative updates → finalize."""

    @pytest.mark.asyncio
    async def test_first_call_seeds_empty_frame_then_returns_true(self):
        """First frame for a chat must be an empty seed (triggers typing UI)."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        # Intermediate frames (finish=False) are fire-and-forget via _send_json;
        # we mock _send_json to capture them and _send_reply_request for finish frames.
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        ok = await adapter.send_stream_frame("hello", chat_id="chat-1")

        assert ok is True
        # Two fire-and-forget calls via _send_json: empty seed + content.
        assert adapter._send_json.await_count == 2
        seed_payload = adapter._send_json.await_args_list[0].args[0]
        assert seed_payload["body"]["msgtype"] == "stream"
        assert seed_payload["body"]["stream"]["content"] == ""
        assert seed_payload["body"]["stream"]["finish"] is False

        content_payload = adapter._send_json.await_args_list[1].args[0]
        assert content_payload["body"]["stream"]["content"] == "hello"
        assert content_payload["body"]["stream"]["finish"] is False

    @pytest.mark.asyncio
    async def test_first_and_second_call_share_stream_id(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        await adapter.send_stream_frame("alpha", chat_id="chat-1")
        await adapter.send_stream_frame("alpha beta", chat_id="chat-1")

        # All fire-and-forget: seed + alpha + alpha beta = 3 calls
        assert adapter._send_json.await_count == 3
        ids = [
            call.args[0]["body"]["stream"]["id"]
            for call in adapter._send_json.await_args_list
        ]
        assert ids[0] == ids[1] == ids[2]
        assert ids[0].startswith("stream_")

    @pytest.mark.asyncio
    async def test_finalize_sends_finish_true_and_resets_state(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        await adapter.send_stream_frame("partial", chat_id="chat-1")
        assert adapter._active_stream_id is not None

        ok = await adapter.send_stream_frame(
            "partial final", chat_id="chat-1", finalize=True,
        )

        assert ok is True
        assert adapter._active_stream_id is None
        assert adapter._active_stream_req_id is None
        assert adapter._active_stream_chat_id is None
        # Finalize goes through _send_reply_request (with ack).
        last_call = adapter._send_reply_request.await_args_list[-1]
        assert last_call.args[1]["stream"]["finish"] is True
        assert last_call.args[1]["stream"]["content"] == "partial final"

# === FAILURE TESTS PLACEHOLDER ===


class TestSendStreamFrameFailures:
    """Behavior when no req_id, 846608 expiry, or generic transport errors."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_req_id_available(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        # No reply_to, nothing in _last_chat_req_ids.
        adapter._send_reply_request = AsyncMock()

        ok = await adapter.send_stream_frame("hi", chat_id="unknown-chat")

        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_false_when_chat_id_missing_on_first_call(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_reply_request = AsyncMock()

        ok = await adapter.send_stream_frame("hi", chat_id=None)

        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_846608_marks_chat_expired_and_returns_false(self):
        """846608 is only detected on finish=True frames (the only ones that
        await an ack). Intermediate frames are fire-and-forget."""
        from gateway.platforms.wecom import (
            STREAM_EXPIRED_ERRCODE, WeComAdapter,
        )

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        # The finalize frame (finish=True) returns 846608.
        adapter._send_reply_request = AsyncMock(
            return_value={"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "stream expired"},
        )

        # First call (with seed + content) succeeds (fire-and-forget).
        await adapter.send_stream_frame("hello", chat_id="chat-1")
        # Now try to finalize — this hits _send_reply_request and gets 846608.
        ok = await adapter.send_stream_frame("hello final", chat_id="chat-1", finalize=True)

        assert ok is False
        assert "chat-1" in adapter._stream_expired_chats
        # State must reset so the next inbound req_id can rebuild a new stream.
        assert adapter._active_stream_id is None

    @pytest.mark.asyncio
    async def test_subsequent_call_to_expired_chat_short_circuits(self):
        """Once a chat is in ``_stream_expired_chats``, send_stream_frame
        bails immediately without touching the WS — caller must use send()."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._stream_expired_chats.add("chat-1")
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_reply_request = AsyncMock()

        ok = await adapter.send_stream_frame("hi", chat_id="chat-1")

        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inbound_message_clears_expired_marker(self):
        """A fresh inbound req_id must resurrect the stream channel."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._stream_expired_chats.add("chat-1")

        adapter._remember_chat_req_id("chat-1", "fresh-req-id")

        assert "chat-1" not in adapter._stream_expired_chats

    @pytest.mark.asyncio
    async def test_generic_transport_error_resets_state(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_reply_request = AsyncMock(
            side_effect=RuntimeError("ws disconnected"),
        )

        ok = await adapter.send_stream_frame("hi", chat_id="chat-1")

        assert ok is False
        # Generic error resets state but does NOT mark chat as expired —
        # next call may still try (e.g. ws reconnects).
        assert adapter._active_stream_id is None
        assert "chat-1" not in adapter._stream_expired_chats

# === SEND_TYPING TESTS PLACEHOLDER ===


class TestSendTypingTriggersThinking:
    """``send_typing`` is a no-op — typing is handled by stream consumer."""

    @pytest.mark.asyncio
    async def test_send_typing_is_noop(self):
        """send_typing must not open any stream — the consumer seed frame does."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        await adapter.send_typing("chat-1")

        adapter._send_json.assert_not_awaited()
        adapter._send_reply_request.assert_not_awaited()
        assert adapter._active_stream_id is None

    @pytest.mark.asyncio
    async def test_send_typing_does_not_raise(self):
        """send_typing must never raise regardless of state."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        await adapter.send_typing("chat-1")
        await adapter.send_typing("")
        await adapter.send_typing(None)  # type: ignore


class TestStreamContentTruncation:
    """Bytes (not codepoints) are truncated to MAX_STREAM_CONTENT_LENGTH."""

    def test_ascii_below_limit_passes_through(self):
        from gateway.platforms.wecom import WeComAdapter

        out = WeComAdapter._truncate_stream_content("hello", 1000)
        assert out == "hello"

    def test_ascii_above_limit_is_byte_capped(self):
        from gateway.platforms.wecom import WeComAdapter

        big = "x" * 30000
        out = WeComAdapter._truncate_stream_content(big, 20480)
        assert len(out.encode("utf-8")) <= 20480

    def test_multibyte_truncation_does_not_split_codepoints(self):
        """A 3-byte CJK char must not be sliced mid-byte and emit garbage."""
        from gateway.platforms.wecom import WeComAdapter

        # Each "你" is 3 UTF-8 bytes.  Limit at 5 bytes — must keep one
        # full char and drop the half-cut second char rather than emit ï¿½.
        out = WeComAdapter._truncate_stream_content("你你", 5)
        assert out == "你"
        # Crucially, must be valid UTF-8 (no replacement chars from
        # mid-byte slices).
        assert "�" not in out


class TestSendClosesActiveStream:
    """``send()`` must finalize any active stream so typing doesn't linger."""

    @pytest.mark.asyncio
    async def test_send_finalizes_active_stream_opened_by_consumer(self):
        """When the stream consumer opened a stream and then send() delivers
        the response (e.g. fallback path), send() must close the stream."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        # Manually set active stream state (as the consumer would).
        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-1", "Hello world!")

        assert result.success is True
        assert adapter._active_stream_id is None
        finalize_calls = [
            call for call in adapter._send_reply_request.await_args_list
            if call.args[1].get("stream", {}).get("finish") is True
        ]
        assert len(finalize_calls) == 1
        assert finalize_calls[0].args[1]["stream"]["content"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_send_ignores_stream_for_different_chat(self):
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-2"] = "req-2"
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-2", "Hi")

        assert result.success is True
        assert adapter._active_stream_id is not None  # untouched

    @pytest.mark.asyncio
    async def test_send_falls_through_when_stream_expired(self):
        from gateway.platforms.wecom import STREAM_EXPIRED_ERRCODE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"

        async def fake(req_id, body, **kwargs):
            if body.get("stream", {}).get("finish"):
                return {"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "expired"}
            return {"errcode": 0, "headers": {"req_id": req_id}}

        adapter._send_reply_request = AsyncMock(side_effect=fake)
        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-1", "Final answer")

        assert result.success is True
        assert adapter._active_stream_id is None
        assert "chat-1" in adapter._stream_expired_chats

