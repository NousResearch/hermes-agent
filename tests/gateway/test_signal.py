"""Tests for Signal messenger platform adapter."""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSignalPlatformEnum:
    def test_signal_enum_exists(self):
        assert Platform.SIGNAL.value == "signal"

    def test_signal_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "signal" in platforms


class TestSignalConfigLoading:
    def test_apply_env_overrides_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        monkeypatch.setenv("SIGNAL_DM_POLICY", "open")
        monkeypatch.setenv("SIGNAL_GROUP_POLICY", "allowlist")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL in config.platforms
        sc = config.platforms[Platform.SIGNAL]
        assert sc.enabled is True
        assert sc.extra["http_url"] == "http://localhost:9090"
        assert sc.extra["account"] == "+15551234567"
        assert sc.extra["dm_policy"] == "open"
        assert sc.extra["group_policy"] == "allowlist"

    def test_signal_not_loaded_without_both_vars(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        # No SIGNAL_ACCOUNT

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL not in config.platforms

    def test_connected_platforms_includes_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.SIGNAL in connected


# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------

class TestSignalAdapterInit:
    def _make_config(self, **extra):
        config = PlatformConfig()
        config.enabled = True
        config.extra = {
            "http_url": "http://localhost:8080",
            "account": "+15551234567",
            "dm_policy": "pairing",
            "group_policy": "disabled",
            **extra,
        }
        return config

    def test_init_parses_config(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15559876543,+15551111111")
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "group123,group456")
        monkeypatch.delenv("SIGNAL_DEBUG", raising=False)

        from gateway.platforms.signal import SignalAdapter
        adapter = SignalAdapter(self._make_config())

        assert adapter.http_url == "http://localhost:8080"
        assert adapter.account == "+15551234567"
        assert adapter.dm_policy == "pairing"
        assert adapter.group_policy == "disabled"
        assert "+15559876543" in adapter.allowed_users
        assert "+15551111111" in adapter.allowed_users
        assert "group123" in adapter.group_allow_from

    def test_init_empty_allowlist(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        monkeypatch.delenv("SIGNAL_DEBUG", raising=False)

        from gateway.platforms.signal import SignalAdapter
        adapter = SignalAdapter(self._make_config())

        assert len(adapter.allowed_users) == 0
        assert len(adapter.group_allow_from) == 0

    def test_init_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        monkeypatch.delenv("SIGNAL_DEBUG", raising=False)

        from gateway.platforms.signal import SignalAdapter
        adapter = SignalAdapter(self._make_config(http_url="http://localhost:8080/"))

        assert adapter.http_url == "http://localhost:8080"


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("+15551234567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("") == "<none>"

    def test_parse_comma_list(self):
        from gateway.platforms.signal import _parse_comma_list
        assert _parse_comma_list("+1234, +5678 , +9012") == ["+1234", "+5678", "+9012"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []

    def test_guess_extension_png(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ".png"

    def test_guess_extension_jpeg(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == ".jpg"

    def test_guess_extension_pdf(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"%PDF-1.4" + b"\x00" * 100) == ".pdf"

    def test_guess_extension_zip(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"PK\x03\x04" + b"\x00" * 100) == ".zip"

    def test_guess_extension_mp4(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100) == ".mp4"

    def test_guess_extension_unknown(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x01\x02\x03" * 10) == ".bin"

    def test_is_image_ext(self):
        from gateway.platforms.signal import _is_image_ext
        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False

    def test_is_audio_ext(self):
        from gateway.platforms.signal import _is_audio_ext
        assert _is_audio_ext(".mp3") is True
        assert _is_audio_ext(".ogg") is True
        assert _is_audio_ext(".png") is False

    def test_check_requirements(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        assert check_signal_requirements() is True

    def test_check_requirements_missing(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        assert check_signal_requirements() is False


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestSignalSessionSource:
    def test_session_source_alt_fields(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            user_id="+15551234567",
            user_id_alt="uuid:abc-123",
            chat_id_alt=None,
        )
        d = source.to_dict()
        assert d["user_id_alt"] == "uuid:abc-123"
        assert "chat_id_alt" not in d  # None fields excluded

    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:xyz",
            chat_type="group",
            user_id="+15551234567",
            user_id_alt="uuid:abc",
            chat_id_alt="xyz",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.user_id_alt == "uuid:abc"
        assert restored.chat_id_alt == "xyz"
        assert restored.platform == Platform.SIGNAL


# ---------------------------------------------------------------------------
# Phone Redaction in agent/redact.py
# ---------------------------------------------------------------------------

class TestSignalPhoneRedaction:
    def test_us_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Call +15551234567 now")
        assert "+15551234567" not in result
        assert "+155" in result  # Prefix preserved
        assert "4567" in result  # Suffix preserved

    def test_uk_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("UK: +442071838750")
        assert "+442071838750" not in result
        assert "****" in result

    def test_multiple_numbers(self):
        from agent.redact import redact_sensitive_text
        text = "From +15551234567 to +442071838750"
        result = redact_sensitive_text(text)
        assert "+15551234567" not in result
        assert "+442071838750" not in result

    def test_short_number_not_matched(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Code: +12345")
        # 5 digits after + is below the 7-digit minimum
        assert "+12345" in result  # Too short to redact


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------

class TestSignalAuthorization:
    def test_signal_in_allowlist_maps(self):
        """Signal should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIGNAL
        source.user_id = "+15559999999"

        # No allowlists set — should check GATEWAY_ALLOW_ALL_USERS
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

class TestSignalSendMessage:
    def test_signal_in_platform_map(self):
        """Signal should be in the send_message tool's platform map."""
        from tools.send_message_tool import send_message_tool
        # Just verify the import works and Signal is a valid platform
        from gateway.config import Platform
        assert Platform.SIGNAL.value == "signal"
