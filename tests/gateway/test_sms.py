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

from gateway.config import Platform, PlatformConfig


# ── Config loading ──────────────────────────────────────────────────

class TestSmsConfigLoading:
    """Verify _apply_env_overrides wires SMS correctly."""


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

# ── Format / truncate ───────────────────────────────────────────────

class TestSmsFormatAndTruncate:
    """Test SmsAdapter.format_message strips markdown."""

    def _make_adapter(self):
        from plugins.platforms.sms.adapter import SmsAdapter

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


    def test_strips_code_blocks(self):
        adapter = self._make_adapter()
        result = adapter.format_message("```python\nprint('hi')\n```")
        assert "```" not in result
        assert "print('hi')" in result


    def test_collapses_newlines(self):
        adapter = self._make_adapter()
        result = adapter.format_message("a\n\n\n\nb")
        assert result == "a\n\nb"


# ── Echo prevention ────────────────────────────────────────────────

class TestSmsEchoPrevention:
    """Adapter should ignore messages from its own number."""

    def test_own_number_detection(self):
        """The adapter stores _from_number for echo prevention."""
        from plugins.platforms.sms.adapter import SmsAdapter

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


    def test_check_sms_requirements_both_set(self):
        from plugins.platforms.sms.adapter import check_sms_requirements

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

# ── Webhook host configuration ─────────────────────────────────────

class TestWebhookHostConfig:
    """Verify SMS_WEBHOOK_HOST env var and default."""


    def test_webhook_url_from_env(self):
        from plugins.platforms.sms.adapter import SmsAdapter

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


# ── Startup guard (fail-closed) ────────────────────────────────────

class TestStartupGuard:
    """Adapter must refuse to start without SMS_WEBHOOK_URL."""

    def _make_adapter(self, extra_env=None):
        from plugins.platforms.sms.adapter import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "+15550001111",
        }
        if extra_env:
            env.update(extra_env)
        with patch.dict(os.environ, env, clear=False):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
        return adapter


    @pytest.mark.asyncio
    async def test_missing_phone_number_is_non_retryable(self):
        from plugins.platforms.sms.adapter import SmsAdapter

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_PHONE_NUMBER": "",
            "SMS_WEBHOOK_URL": "",
        }
        with patch.dict(os.environ, env, clear=True):
            pc = PlatformConfig(enabled=True, api_key="tok")
            adapter = SmsAdapter(pc)
        await adapter.connect()
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "sms_missing_phone_number"

    @pytest.mark.asyncio
    async def test_insecure_flag_does_not_set_fatal_error(self):
        mock_session = AsyncMock()
        with patch.dict(os.environ, {"SMS_INSECURE_NO_SIGNATURE": "true"}), \
             patch("aiohttp.web.AppRunner") as mock_runner_cls, \
             patch("aiohttp.web.TCPSite") as mock_site_cls, \
             patch("aiohttp.ClientSession", return_value=mock_session):
            mock_runner_cls.return_value.setup = AsyncMock()
            mock_runner_cls.return_value.cleanup = AsyncMock()
            mock_site_cls.return_value.start = AsyncMock()
            adapter = self._make_adapter()
            result = await adapter.connect()
            assert result is True
            assert adapter.has_fatal_error is False
            await adapter.disconnect()


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
        from plugins.platforms.sms.adapter import SmsAdapter

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


# ── Webhook signature enforcement (handler-level) ──────────────────

class TestWebhookSignatureEnforcement:
    """Integration tests for signature validation in _handle_webhook."""

    def _make_adapter(self, webhook_url=""):
        from plugins.platforms.sms.adapter import SmsAdapter

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

    def _mock_request(self, body, headers=None, content_length=None):
        request = MagicMock()
        request.read = AsyncMock(return_value=body)
        request.headers = headers or {}
        request.content_length = content_length
        return request

    @pytest.mark.asyncio
    async def test_insecure_flag_skips_validation(self):
        """With SMS_INSECURE_NO_SIGNATURE=true and no URL, requests are accepted."""
        env = {"SMS_INSECURE_NO_SIGNATURE": "true"}
        with patch.dict(os.environ, env):
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
    async def test_webhook_rejects_oversized_body_via_read_length(self):
        """POST whose actual read size exceeds 64 KiB returns 413.

        Covers the case where Content-Length is absent (chunked transfer) but
        the body still exceeds the cap.
        """
        adapter = self._make_adapter(webhook_url="")
        oversized = b"x" * 65_537
        request = self._mock_request(oversized, content_length=None)
        resp = await adapter._handle_webhook(request)
        assert resp.status == 413


# ── Standalone send: markdown stripping ────────────────────────────

class TestStandaloneSendMarkdownStripping:
    """_strip_markdown_for_sms and the standalone send path must work.

    Regression: the helper was added in the plugin migration (560010547)
    using ``re.sub`` without importing ``re``, so every out-of-process SMS
    delivery (cron jobs, the standalone sender contract) crashed with
    ``NameError: name 're' is not defined`` before reaching Twilio.
    """

    def test_strips_bold_italic_headings(self):
        from plugins.platforms.sms.adapter import _strip_markdown_for_sms

        result = _strip_markdown_for_sms(
            "# Title\n**bold** and *italic*\n__under__ and _em_"
        )
        assert result == "Title\nbold and italic\nunder and em"

    def test_strips_code_and_links_keeps_text(self):
        from plugins.platforms.sms.adapter import _strip_markdown_for_sms

        result = _strip_markdown_for_sms("run `cmd` then [docs](https://x.com/y)")
        assert result == "run cmd then docs"

    def test_collapses_excess_newlines(self):
        from plugins.platforms.sms.adapter import _strip_markdown_for_sms

        assert _strip_markdown_for_sms("a\n\n\n\nb") == "a\n\nb"

    @pytest.mark.asyncio
    async def test_standalone_send_delivers_stripped_body(self):
        """The out-of-process send path strips markdown from the SMS body.

        Before the missing-import fix this failed with NameError before the
        Twilio request was ever built.
        """
        from plugins.platforms.sms import adapter as sms_adapter_mod

        captured = {}

        class _FakeForm:
            def add_field(self, name, value):
                captured[name] = value

        class _FakeResp:
            status = 201

            async def json(self):
                return {"sid": "SM123"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _FakeSession:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def post(self, *a, **kw):
                return _FakeResp()

        pc = PlatformConfig(enabled=True, api_key="tok")
        fake_secrets = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tok",
        }
        with patch.dict(
            os.environ, {"TWILIO_PHONE_NUMBER": "+15550001111"}, clear=False
        ), patch.object(
            sms_adapter_mod, "_get_scoped_secret", lambda name, default=None: fake_secrets.get(name, default)
        ), patch(
            "aiohttp.ClientSession", _FakeSession
        ), patch(
            "aiohttp.FormData", _FakeForm
        ):
            result = await sms_adapter_mod._standalone_send(
                pc, "+15559876543", "**urgent** cron `status` report"
            )

        assert result == {
            "success": True,
            "platform": "sms",
            "chat_id": "+15559876543",
            "message_id": "SM123",
        }
        assert captured["From"] == "+15550001111"
        assert captured["To"] == "+15559876543"
        assert captured["Body"] == "urgent cron status report"
