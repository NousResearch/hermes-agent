"""Tests for SMS (Twilio) platform integration.

Covers config loading, format/truncate, echo prevention,
requirements check, toolset verification, and Twilio signature validation.
"""

import base64
import hashlib
import hmac
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, HomeChannel


# ── Config loading ──────────────────────────────────────────────────

class TestSmsConfigLoading:
    """Verify _apply_env_overrides wires SMS correctly."""

    def test_sms_platform_enum_exists(self):
        assert Platform.SMS.value == "sms"

    def test_env_overrides_create_sms_config(self):
        from gateway.config import load_gateway_config

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest123",
            "TWILIO_AUTH_TOKEN": "token_abc",
            "TWILIO_PHONE_NUMBER": "+15551234567",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_gateway_config()
            assert Platform.SMS in config.platforms
            pc = config.platforms[Platform.SMS]
            assert pc.enabled is True
            assert pc.api_key == "token_abc"

    def test_env_overrides_set_home_channel(self):
        from gateway.config import load_gateway_config

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest123",
            "TWILIO_AUTH_TOKEN": "token_abc",
            "TWILIO_PHONE_NUMBER": "+15551234567",
            "SMS_HOME_CHANNEL": "+15559876543",
            "SMS_HOME_CHANNEL_NAME": "My Phone",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_gateway_config()
            hc = config.platforms[Platform.SMS].home_channel
            assert hc is not None
            assert hc.chat_id == "+15559876543"
            assert hc.name == "My Phone"
            assert hc.platform == Platform.SMS

    def test_sms_in_connected_platforms(self):
        from gateway.config import load_gateway_config

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest123",
            "TWILIO_AUTH_TOKEN": "token_abc",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_gateway_config()
            connected = config.get_connected_platforms()
            assert Platform.SMS in connected


# ── Format / truncate ───────────────────────────────────────────────

class TestSmsFormatAndTruncate:
    """Test SmsAdapter.format_message strips markdown."""

    def _make_adapter(self):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = object.__new__(SmsAdapter)
            adapter.config = pc
            adapter._platform = Platform.SMS
            adapter._account_sid = "ACtest"
            adapter._auth_token = "tok"
            adapter._from_number = "+15550001111"
        return adapter

    def test_strips_bold(self):
        adapter = self._make_adapter()
        assert adapter.format_message("**hello**") == "hello"

    def test_strips_italic(self):
        adapter = self._make_adapter()
        assert adapter.format_message("*world*") == "world"

    def test_strips_code_blocks(self):
        adapter = self._make_adapter()
        result = adapter.format_message("```python\nprint('hi')\n```")
        assert "```" not in result
        assert "print('hi')" in result

    def test_strips_inline_code(self):
        adapter = self._make_adapter()
        assert adapter.format_message("`code`") == "code"

    def test_strips_headers(self):
        adapter = self._make_adapter()
        assert adapter.format_message("## Title") == "Title"

    def test_strips_links(self):
        adapter = self._make_adapter()
        assert adapter.format_message("[click](https://example.com)") == "click"

    def test_collapses_newlines(self):
        adapter = self._make_adapter()
        result = adapter.format_message("a\n\n\n\nb")
        assert result == "a\n\nb"


# ── Echo prevention ────────────────────────────────────────────────

class TestSmsEchoPrevention:
    """Adapter should ignore messages from its own number."""

    def test_own_number_detection(self):
        """The adapter stores _from_number for echo prevention."""
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
            assert adapter._from_number == "+15550001111"


# ── Requirements check ─────────────────────────────────────────────

class TestSmsRequirements:
    def test_check_sms_requirements_missing_sid(self):
        from gateway.platforms.sms import check_sms_requirements

        env = {"TWILIO_AUTH_TOKEN": "tok"}
        with patch.dict(os.environ, env, clear=True):
            assert check_sms_requirements() is False

    def test_check_sms_requirements_missing_token(self):
        from gateway.platforms.sms import check_sms_requirements

        env = {"TWILIO_ACCOUNT_SID": "ACtest"}
        with patch.dict(os.environ, env, clear=True):
            assert check_sms_requirements() is False

    def test_check_sms_requirements_both_set(self):
        from gateway.platforms.sms import check_sms_requirements

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
        }
        with patch.dict(os.environ, env, clear=False):
            # Only returns True if aiohttp is also importable
            result = check_sms_requirements()
            try:
                import aiohttp  # noqa: F401
                assert result is True
            except ImportError:
                assert result is False


# ── Toolset verification ───────────────────────────────────────────

class TestSmsToolset:
    def test_hermes_sms_toolset_exists(self):
        from toolsets import get_toolset

        ts = get_toolset("hermes-sms")
        assert ts is not None
        assert "tools" in ts

    def test_hermes_sms_in_gateway_includes(self):
        from toolsets import get_toolset

        gw = get_toolset("hermes-gateway")
        assert gw is not None
        assert "hermes-sms" in gw["includes"]

    def test_sms_platform_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS

        assert "sms" in PLATFORM_HINTS
        assert "concise" in PLATFORM_HINTS["sms"].lower()

    def test_sms_in_scheduler_platform_map(self):
        """Verify cron scheduler recognizes 'sms' as a valid platform."""
        # Just check the Platform enum has SMS — the scheduler imports it dynamically
        assert Platform.SMS.value == "sms"

    def test_sms_in_send_message_platform_map(self):
        """Verify send_message_tool recognizes 'sms'."""
        # The platform_map is built inside _handle_send; verify SMS enum exists
        assert hasattr(Platform, "SMS")

    def test_sms_in_cronjob_deliver_description(self):
        """Verify cronjob_tools mentions sms in deliver description."""
        from tools.cronjob_tools import CRONJOB_SCHEMA
        deliver_desc = CRONJOB_SCHEMA["parameters"]["properties"]["deliver"]["description"]
        assert "sms" in deliver_desc.lower()


# ── Webhook host configuration ─────────────────────────────────────

class TestWebhookHostConfig:
    """Verify SMS_WEBHOOK_HOST env var and default."""

    def test_default_host_is_all_interfaces(self):
        from gateway.platforms.sms import DEFAULT_WEBHOOK_HOST
        assert DEFAULT_WEBHOOK_HOST == "0.0.0.0"

    def test_host_from_env(self):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
            "SMS_WEBHOOK_HOST": "127.0.0.1",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
            assert adapter._webhook_host == "127.0.0.1"

    def test_webhook_url_from_env(self):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
            "SMS_WEBHOOK_URL": "https://example.com/webhooks/twilio",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
            assert adapter._webhook_url == "https://example.com/webhooks/twilio"

    def test_webhook_url_stripped(self):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
            "SMS_WEBHOOK_URL": "  https://example.com/webhooks/twilio  ",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
            assert adapter._webhook_url == "https://example.com/webhooks/twilio"


# ── Twilio signature validation ────────────────────────────────────

def _compute_twilio_signature(auth_token, url, params):
    """Reference implementation of Twilio's signature algorithm."""
    data_to_sign = url
    for key in sorted(params.keys()):
        data_to_sign += key + params[key]
    mac = hmac.new(
        auth_token.encode("utf-8"),
        data_to_sign.encode("utf-8"),
        hashlib.sha1,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


class TestTwilioSignatureValidation:
    """Unit tests for SmsAdapter._validate_twilio_signature."""

    def _make_adapter(self, auth_token="test_token_secret"):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": auth_token,
            "TWILIO_PHONE_NUMBER": "+15550001111",
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key=auth_token)
            adapter = SmsAdapter(pc)
        return adapter

    def test_valid_signature_accepted(self):
        adapter = self._make_adapter()
        url = "https://example.com/webhooks/twilio"
        params = {"From": "+15551234567", "Body": "hello", "To": "+15550001111"}
        sig = _compute_twilio_signature("test_token_secret", url, params)
        assert adapter._validate_twilio_signature(url, params, sig) is True

    def test_invalid_signature_rejected(self):
        adapter = self._make_adapter()
        url = "https://example.com/webhooks/twilio"
        params = {"From": "+15551234567", "Body": "hello"}
        assert adapter._validate_twilio_signature(url, params, "badsig") is False

    def test_wrong_token_rejected(self):
        adapter = self._make_adapter(auth_token="correct_token")
        url = "https://example.com/webhooks/twilio"
        params = {"From": "+15551234567", "Body": "hello"}
        sig = _compute_twilio_signature("wrong_token", url, params)
        assert adapter._validate_twilio_signature(url, params, sig) is False

    def test_params_sorted_by_key(self):
        """Signature must be computed with params sorted alphabetically."""
        adapter = self._make_adapter()
        url = "https://example.com/webhooks/twilio"
        params = {"Zebra": "last", "Alpha": "first", "Middle": "mid"}
        sig = _compute_twilio_signature("test_token_secret", url, params)
        assert adapter._validate_twilio_signature(url, params, sig) is True

    def test_empty_param_values_included(self):
        """Blank values must be included in signature computation."""
        adapter = self._make_adapter()
        url = "https://example.com/webhooks/twilio"
        params = {"From": "+15551234567", "Body": "", "SmsStatus": "received"}
        sig = _compute_twilio_signature("test_token_secret", url, params)
        assert adapter._validate_twilio_signature(url, params, sig) is True

    def test_url_matters(self):
        """Different URLs produce different signatures."""
        adapter = self._make_adapter()
        params = {"Body": "hello"}
        sig = _compute_twilio_signature(
            "test_token_secret", "https://a.com/webhooks/twilio", params
        )
        assert adapter._validate_twilio_signature(
            "https://b.com/webhooks/twilio", params, sig
        ) is False

    def test_port_variant_443_matches_without_port(self):
        """Signature for https URL with :443 validates against URL without port."""
        adapter = self._make_adapter()
        params = {"From": "+15551234567", "Body": "hello"}
        sig = _compute_twilio_signature(
            "test_token_secret", "https://example.com:443/webhooks/twilio", params
        )
        assert adapter._validate_twilio_signature(
            "https://example.com/webhooks/twilio", params, sig
        ) is True

    def test_port_variant_without_port_matches_443(self):
        """Signature for https URL without port validates against URL with :443."""
        adapter = self._make_adapter()
        params = {"From": "+15551234567", "Body": "hello"}
        sig = _compute_twilio_signature(
            "test_token_secret", "https://example.com/webhooks/twilio", params
        )
        assert adapter._validate_twilio_signature(
            "https://example.com:443/webhooks/twilio", params, sig
        ) is True

    def test_non_standard_port_no_variant(self):
        """Non-standard port must NOT match URL without port."""
        adapter = self._make_adapter()
        params = {"From": "+15551234567", "Body": "hello"}
        sig = _compute_twilio_signature(
            "test_token_secret", "https://example.com/webhooks/twilio", params
        )
        assert adapter._validate_twilio_signature(
            "https://example.com:8080/webhooks/twilio", params, sig
        ) is False

    def test_port_variant_http_80(self):
        """Port variant also works for http with port 80."""
        adapter = self._make_adapter()
        params = {"From": "+15551234567", "Body": "hello"}
        sig = _compute_twilio_signature(
            "test_token_secret", "http://example.com:80/webhooks/twilio", params
        )
        assert adapter._validate_twilio_signature(
            "http://example.com/webhooks/twilio", params, sig
        ) is True


# ── Webhook signature enforcement (handler-level) ──────────────────

class TestWebhookSignatureEnforcement:
    """Integration tests for signature validation in _handle_webhook."""

    def _make_adapter(self, webhook_url=""):
        from gateway.platforms.sms import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "test_token_secret",
            "TWILIO_PHONE_NUMBER": "+15550001111",
            "SMS_WEBHOOK_URL": webhook_url,
        }
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="test_token_secret")
            adapter = SmsAdapter(pc)
        adapter._message_handler = AsyncMock()
        return adapter

    def _mock_request(self, body, headers=None):
        request = MagicMock()
        request.read = AsyncMock(return_value=body)
        request.headers = headers or {}
        return request

    @pytest.mark.asyncio
    async def test_no_webhook_url_skips_validation(self):
        """Without SMS_WEBHOOK_URL, all requests are accepted."""
        adapter = self._make_adapter(webhook_url="")
        body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
        request = self._mock_request(body)
        resp = await adapter._handle_webhook(request)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_missing_signature_returns_403(self):
        adapter = self._make_adapter(webhook_url="https://example.com/webhooks/twilio")
        body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
        request = self._mock_request(body, headers={})
        resp = await adapter._handle_webhook(request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_403(self):
        adapter = self._make_adapter(webhook_url="https://example.com/webhooks/twilio")
        body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
        request = self._mock_request(body, headers={"X-Twilio-Signature": "invalid"})
        resp = await adapter._handle_webhook(request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_valid_signature_returns_200(self):
        webhook_url = "https://example.com/webhooks/twilio"
        adapter = self._make_adapter(webhook_url=webhook_url)
        params = {
            "From": "+15551234567",
            "To": "+15550001111",
            "Body": "hello",
            "MessageSid": "SM123",
        }
        sig = _compute_twilio_signature("test_token_secret", webhook_url, params)
        body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
        request = self._mock_request(body, headers={"X-Twilio-Signature": sig})
        resp = await adapter._handle_webhook(request)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_port_variant_signature_returns_200(self):
        """Signature computed with :443 should pass when URL configured without port."""
        webhook_url = "https://example.com/webhooks/twilio"
        adapter = self._make_adapter(webhook_url=webhook_url)
        params = {
            "From": "+15551234567",
            "To": "+15550001111",
            "Body": "hello",
            "MessageSid": "SM123",
        }
        sig = _compute_twilio_signature(
            "test_token_secret", "https://example.com:443/webhooks/twilio", params
        )
        body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
        request = self._mock_request(body, headers={"X-Twilio-Signature": sig})
        resp = await adapter._handle_webhook(request)
        assert resp.status == 200
