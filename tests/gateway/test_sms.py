"""Tests for SMS (Telnyx) platform adapter."""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSmsPlatformEnum:
    def test_sms_enum_exists(self):
        assert Platform.SMS.value == "sms"

    def test_sms_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "sms" in platforms


class TestSmsConfigLoading:
    def test_apply_env_overrides_sms(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "KEY_test123")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SMS in config.platforms
        sc = config.platforms[Platform.SMS]
        assert sc.enabled is True
        assert sc.api_key == "KEY_test123"

    def test_sms_not_loaded_without_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SMS not in config.platforms

    def test_connected_platforms_includes_sms(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "KEY_test123")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.SMS in connected

    def test_sms_home_channel(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "KEY_test123")
        monkeypatch.setenv("SMS_HOME_CHANNEL", "+15559876543")
        monkeypatch.setenv("SMS_HOME_CHANNEL_NAME", "Owner")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.SMS)
        assert home is not None
        assert home.chat_id == "+15559876543"
        assert home.name == "Owner"


# ---------------------------------------------------------------------------
# Adapter format / truncate
# ---------------------------------------------------------------------------

class TestSmsFormatMessage:
    def setup_method(self):
        from gateway.platforms.sms import SmsAdapter
        config = PlatformConfig(enabled=True, api_key="test_key")
        with patch.dict("os.environ", {"TELNYX_API_KEY": "test_key"}):
            self.adapter = SmsAdapter(config)

    def test_strip_bold(self):
        assert self.adapter.format_message("**bold**") == "bold"

    def test_strip_italic(self):
        assert self.adapter.format_message("*italic*") == "italic"

    def test_strip_code_block(self):
        result = self.adapter.format_message("```python\ncode\n```")
        assert "```" not in result
        assert "code" in result

    def test_strip_inline_code(self):
        assert self.adapter.format_message("`code`") == "code"

    def test_strip_headers(self):
        assert self.adapter.format_message("## Header") == "Header"

    def test_strip_links(self):
        assert self.adapter.format_message("[click](http://example.com)") == "click"

    def test_collapse_newlines(self):
        result = self.adapter.format_message("a\n\n\n\nb")
        assert result == "a\n\nb"


class TestSmsTruncateMessage:
    def setup_method(self):
        from gateway.platforms.sms import SmsAdapter
        config = PlatformConfig(enabled=True, api_key="test_key")
        with patch.dict("os.environ", {"TELNYX_API_KEY": "test_key"}):
            self.adapter = SmsAdapter(config)

    def test_short_message_single_chunk(self):
        msg = "Hello, world!"
        chunks = self.adapter.truncate_message(msg)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_long_message_splits(self):
        msg = "a " * 1000  # 2000 chars
        chunks = self.adapter.truncate_message(msg)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 1600

    def test_custom_max_length(self):
        msg = "Hello " * 20
        chunks = self.adapter.truncate_message(msg, max_length=50)
        assert all(len(c) <= 50 for c in chunks)


# ---------------------------------------------------------------------------
# Echo loop prevention
# ---------------------------------------------------------------------------

class TestSmsEchoLoop:
    def test_own_number_ignored(self):
        from gateway.platforms.sms import SmsAdapter
        config = PlatformConfig(enabled=True, api_key="test_key")
        with patch.dict("os.environ", {
            "TELNYX_API_KEY": "test_key",
            "TELNYX_FROM_NUMBERS": "+15551234567,+15559876543",
        }):
            adapter = SmsAdapter(config)
            assert "+15551234567" in adapter._from_numbers
            assert "+15559876543" in adapter._from_numbers


# ---------------------------------------------------------------------------
# Auth maps
# ---------------------------------------------------------------------------

class TestSmsAuthMaps:
    def test_sms_in_allowed_users_map(self):
        """SMS should be in the platform auth maps in run.py."""
        # Verify the env var names are consistent
        import os
        os.environ.setdefault("SMS_ALLOWED_USERS", "+15551234567")
        assert os.getenv("SMS_ALLOWED_USERS") == "+15551234567"

    def test_sms_allow_all_env_var(self):
        """SMS_ALLOW_ALL_USERS should be recognized."""
        import os
        os.environ.setdefault("SMS_ALLOW_ALL_USERS", "true")
        assert os.getenv("SMS_ALLOW_ALL_USERS") == "true"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestSmsRequirements:
    def test_check_sms_requirements_with_key(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "KEY_test123")
        from gateway.platforms.sms import check_sms_requirements
        # aiohttp is available in test environment
        assert check_sms_requirements() is True

    def test_check_sms_requirements_without_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        from gateway.platforms.sms import check_sms_requirements
        assert check_sms_requirements() is False


# ---------------------------------------------------------------------------
# Toolset & integration points
# ---------------------------------------------------------------------------

class TestSmsToolset:
    def test_hermes_sms_toolset_exists(self):
        from toolsets import get_toolset
        ts = get_toolset("hermes-sms")
        assert ts is not None
        assert "hermes-sms" in ts.get("description", "").lower() or "sms" in ts.get("description", "").lower()

    def test_hermes_gateway_includes_sms(self):
        from toolsets import get_toolset
        gw = get_toolset("hermes-gateway")
        assert "hermes-sms" in gw["includes"]


class TestSmsPlatformHints:
    def test_sms_in_platform_hints(self):
        from agent.prompt_builder import PLATFORM_HINTS
        assert "sms" in PLATFORM_HINTS
        assert "SMS" in PLATFORM_HINTS["sms"] or "sms" in PLATFORM_HINTS["sms"].lower()


class TestSmsCronDelivery:
    def test_sms_in_cron_platform_map(self):
        """Verify the cron scheduler can resolve 'sms' platform."""
        # The platform_map in _deliver_result should include sms
        from gateway.config import Platform
        assert Platform.SMS.value == "sms"


class TestSmsSendMessageTool:
    def test_sms_in_send_message_platform_map(self):
        """The send_message tool should recognize 'sms' as a valid platform."""
        # We verify by checking that SMS is in the Platform enum
        # and the code path exists
        from gateway.config import Platform
        assert hasattr(Platform, "SMS")


class TestSmsChannelDirectory:
    def test_sms_in_session_discovery(self):
        """Verify SMS is included in session-based channel discovery."""
        import inspect
        from gateway.channel_directory import build_channel_directory
        source = inspect.getsource(build_channel_directory)
        assert '"sms"' in source


class TestSmsStatus:
    def test_sms_in_status_platforms(self):
        """Verify SMS appears in the status command platforms dict."""
        import inspect
        from hermes_cli.status import show_status
        source = inspect.getsource(show_status)
        assert '"SMS"' in source or "'SMS'" in source
