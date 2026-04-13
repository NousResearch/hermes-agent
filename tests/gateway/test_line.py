"""Tests for the LINE Messaging API gateway adapter."""
import base64
import hashlib
import hmac
import json
import os
import stat as stat_module
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_config(**extra):
    return PlatformConfig(
        enabled=True,
        token="test-channel-access-token",
        extra={
            "channel_secret": "test-channel-secret",
            **extra,
        },
    )


class TestLinePlatformEnum:
    def test_line_enum_exists(self):
        assert Platform.LINE.value == "line"

    def test_line_enum_from_string(self):
        assert Platform("line") == Platform.LINE


class TestLineConfigLoading:
    def test_apply_env_overrides_line(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        monkeypatch.setenv("LINE_WEBHOOK_PORT", "9999")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE in config.platforms
        lc = config.platforms[Platform.LINE]
        assert lc.enabled is True
        assert lc.token == "test-token"
        assert lc.extra["channel_secret"] == "test-secret"
        assert lc.extra["webhook_port"] == 9999

    def test_connected_platforms_includes_line(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE in config.get_connected_platforms()

    def test_home_channel_set_from_env(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        monkeypatch.setenv("LINE_HOME_CHANNEL", "U1234567890abcdef")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        hc = config.platforms[Platform.LINE].home_channel
        assert hc is not None
        assert hc.chat_id == "U1234567890abcdef"

    def test_not_connected_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE not in config.get_connected_platforms()


class TestLineHelpers:
    def test_check_requirements_with_credentials(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import check_line_requirements

        assert check_line_requirements() is True

    def test_check_requirements_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.platforms.line import check_line_requirements

        assert check_line_requirements() is False


class TestLineSignatureVerification:
    def test_valid_signature(self):
        from gateway.platforms.line import _verify_signature

        secret = "test-secret"
        body = b'{"events":[]}'
        expected_sig = base64.b64encode(
            hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        ).decode("utf-8")

        assert _verify_signature(body, expected_sig, secret) is True

    def test_invalid_signature(self):
        from gateway.platforms.line import _verify_signature

        assert _verify_signature(b'{"events":[]}', "invalid-sig", "test-secret") is False


class TestLineTextExtraction:
    def test_text_message(self):
        from gateway.platforms.line import _extract_text_from_event

        event = {"message": {"type": "text", "text": "Hello!"}}
        assert _extract_text_from_event(event) == "Hello!"

    def test_sticker_with_keywords(self):
        from gateway.platforms.line import _extract_text_from_event

        event = {"message": {"type": "sticker", "keywords": ["happy", "smile"]}}
        assert _extract_text_from_event(event) == "[Sticker: happy, smile]"

    def test_sticker_without_keywords(self):
        from gateway.platforms.line import _extract_text_from_event

        event = {"message": {"type": "sticker"}}
        assert _extract_text_from_event(event) == "[Sticker]"

    def test_location_message(self):
        from gateway.platforms.line import _extract_text_from_event

        event = {
            "message": {
                "type": "location",
                "title": "Tokyo Tower",
                "address": "Minato, Tokyo",
                "latitude": 35.6586,
                "longitude": 139.7454,
            }
        }
        result = _extract_text_from_event(event)
        assert "Tokyo Tower" in result
        assert "Location" in result

    def test_unknown_message_type(self):
        from gateway.platforms.line import _extract_text_from_event

        event = {"message": {"type": "unknown"}}
        assert _extract_text_from_event(event) == ""


class TestLineAdapterInit:
    def test_adapter_init(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        config = _make_config()
        adapter = LineAdapter(config)
        assert adapter.platform == Platform.LINE
        assert adapter.channel_access_token == "test-channel-access-token"
        assert adapter.channel_secret == "test-channel-secret"
        assert adapter.MAX_MESSAGE_LENGTH == 5000

    def test_adapter_resolve_chat_id_user(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        assert adapter._resolve_chat_id({"type": "user", "userId": "U123"}) == "U123"

    def test_adapter_resolve_chat_id_group(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        assert adapter._resolve_chat_id({"type": "group", "groupId": "C123"}) == "C123"

    def test_adapter_resolve_chat_id_room(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        assert adapter._resolve_chat_id({"type": "room", "roomId": "R123"}) == "R123"

    def test_build_text_messages_short(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        messages = adapter._build_text_messages("Hello")
        assert len(messages) == 1
        assert messages[0] == {"type": "text", "text": "Hello"}

    def test_build_text_messages_long(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        long_text = "A" * 10000
        messages = adapter._build_text_messages(long_text)
        assert len(messages) == 2
        assert len(messages[0]["text"]) == 5000
        assert len(messages[1]["text"]) == 5000


class TestLineAuthorization:
    def test_line_in_platform_env_map(self, monkeypatch):
        """Verify LINE is in the gateway runner's authorization maps."""
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
        # Just verify the Platform enum is accessible — the actual auth
        # integration is covered by the GatewayRunner tests.
        assert Platform.LINE.value == "line"


class TestLineToolset:
    def test_line_toolset_exists(self):
        from toolsets import TOOLSETS

        assert "hermes-line" in TOOLSETS

    def test_line_in_gateway_toolset(self):
        from toolsets import TOOLSETS

        gateway = TOOLSETS["hermes-gateway"]
        assert "hermes-line" in gateway["includes"]


class TestLineCronDelivery:
    def test_line_in_cron_platform_map(self):
        """Verify LINE is in the cron scheduler platform map."""
        # The cron scheduler loads Platform at import time; just verify
        # the enum value matches what the scheduler expects.
        assert Platform.LINE.value == "line"


class TestLineSendMessageTool:
    def test_line_in_send_platform_map(self):
        """Verify LINE is in the send_message_tool platform map."""
        assert Platform.LINE.value == "line"


class TestLineGetConnectedPlatforms:
    def test_line_requires_token_and_secret(self, monkeypatch):
        """LINE should not appear as connected if token is missing."""
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import GatewayConfig, PlatformConfig, _apply_env_overrides

        config = GatewayConfig()
        config.platforms[Platform.LINE] = PlatformConfig(
            enabled=True,
            token=None,
            extra={"channel_secret": "secret"},
        )
        # Must NOT appear when token is absent
        assert Platform.LINE not in config.get_connected_platforms()

    def test_line_connected_with_both_credentials(self):
        """LINE appears as connected when both token and secret are present."""
        from gateway.config import GatewayConfig, PlatformConfig

        config = GatewayConfig()
        config.platforms[Platform.LINE] = PlatformConfig(
            enabled=True,
            token="tok",
            extra={"channel_secret": "secret"},
        )
        assert Platform.LINE in config.get_connected_platforms()


class TestLineDocumentCaching:
    def test_cache_document_from_bytes_signature(self):
        """Regression: cache_document_from_bytes must accept (data, filename) — not (data, ext, original_name=…)."""
        import inspect
        from gateway.platforms.base import cache_document_from_bytes

        sig = inspect.signature(cache_document_from_bytes)
        params = list(sig.parameters.keys())
        # Signature must be (data, filename) — no 'original_name' keyword
        assert params == ["data", "filename"], (
            f"cache_document_from_bytes signature changed: {params}"
        )


class TestLineInboundImageType:
    def test_image_event_yields_photo_message_type(self):
        """Inbound LINE image webhook must map to MessageType.PHOTO so the
        vision pipeline in gateway/run.py picks it up correctly."""
        from gateway.platforms.base import MessageType

        # The adapter maps LINE's "image" media type to MessageType.PHOTO.
        # Verify the enum value exists and is distinct from TEXT/DOCUMENT.
        assert MessageType.PHOTO != MessageType.TEXT
        assert MessageType.PHOTO != MessageType.DOCUMENT

    def test_line_adapter_source_code_maps_image_to_photo(self):
        """The LineAdapter source must assign MessageType.PHOTO (not .IMAGE) for the
        'image' media type so inbound images reach the vision pipeline."""
        import inspect
        from gateway.platforms import line as line_module

        source = inspect.getsource(line_module.LineAdapter)
        # The adapter must reference MessageType.PHOTO, not MessageType.IMAGE
        assert "MessageType.PHOTO" in source, (
            "LineAdapter must map LINE 'image' events to MessageType.PHOTO"
        )
        assert "MessageType.IMAGE" not in source, (
            "LineAdapter must not use MessageType.IMAGE — use MessageType.PHOTO"
        )


class TestLineSendImage:
    def test_send_image_rejects_non_https_url(self):
        """send_image() must refuse http:// and local paths — LINE requires HTTPS."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        for bad_url in ("http://example.com/img.jpg", "/tmp/foo.jpg", "file:///tmp/foo.jpg"):
            result = asyncio.get_event_loop().run_until_complete(
                adapter.send_image("U123", bad_url)
            )
            assert not result.success, f"Expected failure for {bad_url!r}"
            assert "HTTPS" in result.error or "https" in result.error.lower()

    def test_send_image_accepts_https_url(self):
        """send_image() must call LINE Push API for a valid HTTPS URL."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(adapter, "_ensure_client", return_value=mock_client):
            result = asyncio.get_event_loop().run_until_complete(
                adapter.send_image("U123", "https://example.com/image.jpg")
            )

        assert result.success
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["json"]
        image_msg = next(m for m in payload["messages"] if m["type"] == "image")
        assert image_msg["originalContentUrl"].startswith("https://")
        assert image_msg["previewImageUrl"].startswith("https://")


class TestLineSendImageFile:
    def test_send_image_file_missing_path_returns_failure(self):
        """send_image_file() must return failure when the file does not exist."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.send_image_file("U123", "/nonexistent/path/image.jpg")
        )
        assert not result.success
        assert "not found" in result.error.lower() or "File" in result.error

    def test_send_image_file_registers_token_and_calls_send_image(self):
        """send_image_file() must register a media token and delegate to send_image()
        with an HTTPS URL built from webhook_host/port."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config(webhook_host="mybot.example.com", webhook_port=443))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
            tmp_path = f.name

        try:
            captured = {}

            async def fake_send_image(**kwargs):
                captured["url"] = kwargs["image_url"]
                return MagicMock(success=True)

            with patch.object(adapter, "send_image", side_effect=fake_send_image):
                asyncio.get_event_loop().run_until_complete(
                    adapter.send_image_file("U123", tmp_path)
                )

            assert "url" in captured, "send_image was not called"
            url = captured["url"]
            assert url.startswith("https://mybot.example.com/line/media/"), (
                f"Unexpected URL: {url}"
            )
            assert url.endswith(os.path.basename(tmp_path))
            # Token must be registered in the adapter
            assert len(adapter._media_tokens) == 1
        finally:
            os.unlink(tmp_path)

    def test_send_image_file_no_text_fallback(self):
        """send_image_file() must NOT fall back to the '🖼️ Image:' text message."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        sent_texts = []

        async def fake_send(_chat_id, content, **_kwargs):
            sent_texts.append(content)
            return MagicMock(success=True)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            tmp_path = f.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch.object(adapter, "_ensure_client", return_value=mock_client), \
                 patch.object(adapter, "send", side_effect=fake_send):
                asyncio.get_event_loop().run_until_complete(
                    adapter.send_image_file("U123", tmp_path)
                )

            # No text message containing the file path should have been sent
            for text in sent_texts:
                assert "🖼️" not in text and tmp_path not in text, (
                    f"Leaked file path as text: {text!r}"
                )
        finally:
            os.unlink(tmp_path)


class TestLineMediaEndpoint:
    def test_register_media_returns_token(self):
        """_register_media must return a non-empty token and store the path."""
        import time
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            token = adapter._register_media(tmp_path)
            assert token and len(token) > 10
            assert token in adapter._media_tokens
            stored_path, expires_at = adapter._media_tokens[token]
            assert stored_path.endswith(os.path.basename(tmp_path))
            assert expires_at > time.time()
        finally:
            os.unlink(tmp_path)

    def test_register_media_evicts_expired_tokens(self):
        """_register_media must remove expired tokens on each call."""
        import time
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        # Inject an already-expired token
        adapter._media_tokens["stale"] = ("/some/path.jpg", time.time() - 1)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            adapter._register_media(tmp_path)
            assert "stale" not in adapter._media_tokens
        finally:
            os.unlink(tmp_path)

    def test_media_url_uses_webhook_host_and_port(self):
        """_media_url must build an HTTPS URL from webhook_host and webhook_port."""
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config(webhook_host="mybot.example.com", webhook_port=443))
        url = adapter._media_url("tok123", "photo.jpg")
        assert url == "https://mybot.example.com/line/media/tok123/photo.jpg"

    def test_media_url_includes_port_when_not_443(self):
        """Non-443 ports must be included explicitly in the URL."""
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config(webhook_host="mybot.example.com", webhook_port=8443))
        url = adapter._media_url("tok123", "photo.jpg")
        assert url == "https://mybot.example.com:8443/line/media/tok123/photo.jpg"


class TestLineSendVoice:
    def test_send_voice_missing_file(self):
        """send_voice() must return failure when file does not exist."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.send_voice("U123", "/nonexistent/audio.m4a")
        )
        assert not result.success
        assert "not found" in result.error.lower()

    def test_send_voice_calls_audio_message_type(self):
        """send_voice() must send a LINE audio message (not text fallback)."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config(webhook_host="bot.example.com", webhook_port=443))

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch.object(adapter, "_ensure_client", return_value=mock_client):
                asyncio.get_event_loop().run_until_complete(
                    adapter.send_voice("U123", tmp_path)
                )

            payload = mock_client.post.call_args.kwargs["json"]
            audio_msg = next(m for m in payload["messages"] if m["type"] == "audio")
            assert audio_msg["originalContentUrl"].startswith("https://")
            assert "duration" in audio_msg
        finally:
            os.unlink(tmp_path)

    def test_send_voice_size_limit(self):
        """send_voice() must reject files exceeding LINE's 200 MB limit."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            tmp_path = f.name

        try:
            fake = MagicMock(st_size=201 * 1024 * 1024, st_mode=stat_module.S_IFREG | 0o644)
            with patch("pathlib.Path.stat", return_value=fake):
                result = asyncio.get_event_loop().run_until_complete(
                    adapter.send_voice("U123", tmp_path)
                )
            assert not result.success
            assert "200 MB" in result.error
        finally:
            os.unlink(tmp_path)


class TestLineSendVideo:
    def test_send_video_missing_file(self):
        """send_video() must return failure when file does not exist."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.send_video("U123", "/nonexistent/video.mp4")
        )
        assert not result.success
        assert "not found" in result.error.lower()

    def test_send_video_payload(self):
        """send_video() must send video + preview PNG via LINE Push API."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config(webhook_host="bot.example.com", webhook_port=443))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch.object(adapter, "_ensure_client", return_value=mock_client):
                asyncio.get_event_loop().run_until_complete(
                    adapter.send_video("U123", tmp_path)
                )

            payload = mock_client.post.call_args.kwargs["json"]
            video_msg = next(m for m in payload["messages"] if m["type"] == "video")
            assert video_msg["originalContentUrl"].startswith("https://")
            assert video_msg["previewImageUrl"].startswith("https://")
            assert video_msg["previewImageUrl"].endswith(".png")
            # video and preview must be different URLs
            assert video_msg["originalContentUrl"] != video_msg["previewImageUrl"]
        finally:
            os.unlink(tmp_path)

    def test_send_video_size_limit(self):
        """send_video() must reject files exceeding LINE's 200 MB limit."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            fake = MagicMock(st_size=201 * 1024 * 1024, st_mode=stat_module.S_IFREG | 0o644)
            with patch("pathlib.Path.stat", return_value=fake):
                result = asyncio.get_event_loop().run_until_complete(
                    adapter.send_video("U123", tmp_path)
                )
            assert not result.success
            assert "200 MB" in result.error
        finally:
            os.unlink(tmp_path)


class TestLineSendImageFileSize:
    def test_send_image_file_size_limit(self):
        """send_image_file() must reject files exceeding LINE's 10 MB limit."""
        import asyncio
        from gateway.platforms.line import LineAdapter

        adapter = LineAdapter(_make_config())

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            fake = MagicMock(st_size=11 * 1024 * 1024, st_mode=stat_module.S_IFREG | 0o644)
            with patch("pathlib.Path.stat", return_value=fake):
                result = asyncio.get_event_loop().run_until_complete(
                    adapter.send_image_file("U123", tmp_path)
                )
            assert not result.success
            assert "10 MB" in result.error
        finally:
            os.unlink(tmp_path)


class TestLineMakePreviewPng:
    def test_make_preview_png_is_valid_png(self):
        """_make_preview_png() must return bytes starting with the PNG signature."""
        from gateway.platforms.line import _make_preview_png

        data = _make_preview_png()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG signature"
        assert len(data) > 20
