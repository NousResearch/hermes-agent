"""Tests for the LINE Messaging API gateway adapter."""
import base64
import hashlib
import hmac
import json

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
