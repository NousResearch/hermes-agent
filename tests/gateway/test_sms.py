"""Tests for SMS (Twilio) platform integration.

Covers config loading, format/truncate, echo prevention,
requirements check, toolset verification, and Twilio signature validation.
"""

import asyncio
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



class TestMultiplexProfileScope:
    """TWILIO_PHONE_NUMBER must resolve through the same profile scope as the Twilio secrets: under
    multiplex, os.environ holds the DEFAULT profile's number."""

    @pytest.fixture(autouse=True)
    def _default_profile_env(self, monkeypatch):
        from agent.secret_scope import set_multiplex_active
        for key, value in (("TWILIO_ACCOUNT_SID", "AC-default"), ("TWILIO_AUTH_TOKEN", "token-default"),
                           ("TWILIO_PHONE_NUMBER", "+15550000000")):
            monkeypatch.setenv(key, value)
        set_multiplex_active(True)
        yield
        set_multiplex_active(False)

    def test_init_pairs_secondary_secrets_with_secondary_from_number(self):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from plugins.platforms.sms.adapter import SmsAdapter

        token = set_secret_scope({"TWILIO_ACCOUNT_SID": "AC-profile", "TWILIO_AUTH_TOKEN": "token-profile",
                                  "TWILIO_PHONE_NUMBER": "+15551112222"})
        try:
            adapter = SmsAdapter(PlatformConfig(enabled=True))
        finally:
            reset_secret_scope(token)
        assert (adapter._account_sid, adapter._from_number) == ("AC-profile", "+15551112222")

    @pytest.mark.asyncio
    async def test_standalone_send_without_own_number_fails_closed(self):
        """A secondary lacking its own from-number must NOT send from the default's +15550000000."""
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from plugins.platforms.sms.adapter import _standalone_send

        token = set_secret_scope({"TWILIO_ACCOUNT_SID": "AC-profile", "TWILIO_AUTH_TOKEN": "token-profile"})
        try:
            result = await _standalone_send(PlatformConfig(enabled=True), "+15559998888", "hi")
        finally:
            reset_secret_scope(token)
        assert "TWILIO_PHONE_NUMBER required" in result["error"]


# ── GSM-7 normalization ─────────────────────────────────────────────
class TestGsmNormalize:
    def test_typography_becomes_ascii(self):
        from plugins.platforms.sms.adapter import _gsm_normalize
        out = _gsm_normalize("a — b ‘c’ “d” … • e  f")
        assert out.isascii()
        assert out == 'a - b \'c\' "d" ... - e  f'

    def test_chunks_respect_max_sms_length(self):
        from plugins.platforms.sms.adapter import MAX_SMS_LENGTH, SmsAdapter, _gsm_normalize
        long = "word — " * 400
        chunks = SmsAdapter.truncate_message(_gsm_normalize(long), MAX_SMS_LENGTH)
        assert len(chunks) > 1
        assert all(len(c) <= MAX_SMS_LENGTH for c in chunks)

    def test_standalone_send_normalizes(self):
        import inspect
        from plugins.platforms.sms import adapter
        assert "_gsm_normalize(_strip_markdown_for_sms" in inspect.getsource(adapter._standalone_send)


# ── Delivery-status callbacks ───────────────────────────────────────
class TestStatusCallback:
    def _make_adapter(self, webhook_url="https://example.com/webhooks/twilio", status_url=None):
        from plugins.platforms.sms.adapter import SmsAdapter
        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "test_token_secret",
            "TWILIO_PHONE_NUMBER": "+15550001111",
            "SMS_WEBHOOK_URL": webhook_url,
        }
        if status_url is not None:
            env["SMS_STATUS_WEBHOOK_URL"] = status_url
        with patch.dict(os.environ, env):
            pc = PlatformConfig(enabled=True, api_key="test_token_secret")
            adapter = SmsAdapter(pc)
            adapter._env = env
        return adapter

    def _mock_request(self, body, headers=None, content_length=None):
        request = MagicMock()
        request.read = AsyncMock(return_value=body)
        request.headers = headers or {}
        request.content_length = content_length
        return request

    @staticmethod
    def _sign(url, params, token="test_token_secret"):
        s = url + "".join(k + params[k] for k in sorted(params))
        return base64.b64encode(hmac.new(token.encode(), s.encode(), hashlib.sha1).digest()).decode()

    _PARAMS = {"MessageSid": "SM123", "MessageStatus": "undelivered", "ErrorCode": "30019",
               "To": "+15551234567", "From": "+15550001111"}

    def _body(self):
        from urllib.parse import urlencode
        return urlencode(self._PARAMS).encode()

    @pytest.mark.asyncio
    async def test_missing_signature_returns_403(self):
        adapter = self._make_adapter()
        with patch.dict(os.environ, adapter._env):
            resp = await adapter._handle_status(self._mock_request(self._body()))
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_signed_callback_accepted_and_alert_scheduled(self):
        adapter = self._make_adapter(status_url="https://example.com/webhooks/twilio/status")
        adapter._notify_operator = AsyncMock()
        sig = self._sign("https://example.com/webhooks/twilio/status", self._PARAMS)
        env = dict(adapter._env, TELEGRAM_BOT_TOKEN="t", TELEGRAM_HOME_CHANNEL="1")
        with patch.dict(os.environ, env):
            resp = await adapter._handle_status(self._mock_request(self._body(), headers={"X-Twilio-Signature": sig}))
            for t in list(adapter._background_tasks):
                await t
        assert resp.status == 200
        adapter._notify_operator.assert_awaited_once()
        assert "undelivered" in adapter._notify_operator.await_args.args[2]

    @pytest.mark.asyncio
    async def test_status_url_derived_from_webhook_url(self):
        """With SMS_STATUS_WEBHOOK_URL unset the inbound URL + '/status' is what the signature is checked against."""
        adapter = self._make_adapter()
        sig = self._sign("https://example.com/webhooks/twilio/status", self._PARAMS)
        with patch.dict(os.environ, adapter._env):
            resp = await adapter._handle_status(self._mock_request(self._body(), headers={"X-Twilio-Signature": sig}))
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_alerts_are_rate_limited(self):
        adapter = self._make_adapter(status_url="https://example.com/webhooks/twilio/status")
        adapter._notify_operator = AsyncMock()
        sig = self._sign("https://example.com/webhooks/twilio/status", self._PARAMS)
        env = dict(adapter._env, TELEGRAM_BOT_TOKEN="t", TELEGRAM_HOME_CHANNEL="1")
        with patch.dict(os.environ, env):
            for _ in range(3):
                await adapter._handle_status(self._mock_request(self._body(), headers={"X-Twilio-Signature": sig}))
            for t in list(adapter._background_tasks):
                await t
        assert adapter._notify_operator.await_count == 1
        assert adapter._alert_suppressed == 2

    @pytest.mark.asyncio
    async def test_oversized_body_returns_413(self):
        adapter = self._make_adapter()
        resp = await adapter._handle_status(self._mock_request(b"x" * 65_537, content_length=65_537))
        assert resp.status == 413


# ── Inbound MMS ─────────────────────────────────────────────────────
class TestInboundMedia:
    def _make_adapter(self):
        from plugins.platforms.sms.adapter import SmsAdapter
        env = {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "tok", "TWILIO_PHONE_NUMBER": "+15550001111",
               "SMS_WEBHOOK_URL": "", "SMS_INSECURE_NO_SIGNATURE": "true"}
        with patch.dict(os.environ, env):
            adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="tok"))
        adapter._message_handler = AsyncMock()
        adapter._env = env
        return adapter

    @pytest.mark.asyncio
    async def test_media_message_is_ingested_in_background(self):
        adapter = self._make_adapter()
        adapter._ingest_with_media = AsyncMock()
        body = b"From=%2B15551234567&To=%2B15550001111&Body=&MessageSid=MM1&NumMedia=1&MediaUrl0=https%3A%2F%2Fapi.twilio.com%2Fm%2F1&MediaContentType0=audio%2Famr"
        request = MagicMock(); request.read = AsyncMock(return_value=body); request.headers = {}; request.content_length = None
        with patch.dict(os.environ, adapter._env):
            resp = await adapter._handle_webhook(request)
            for t in list(adapter._background_tasks):
                await t
        assert resp.status == 200
        adapter._ingest_with_media.assert_awaited_once()
        assert adapter._ingest_with_media.await_args.args[-1] == 1

    @pytest.mark.asyncio
    async def test_unretrievable_media_yields_marker_text(self):
        from gateway.platforms.event import MessageType
        adapter = self._make_adapter()
        adapter._download_inbound_media = AsyncMock(return_value=([], []))
        adapter.handle_message = AsyncMock()
        await adapter._ingest_with_media({}, "+15551234567", "", "MM1", 1)
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert "could not be retrieved" in event.text

    @pytest.mark.asyncio
    async def test_audio_media_becomes_voice_event(self):
        from gateway.platforms.event import MessageType
        adapter = self._make_adapter()
        adapter._download_inbound_media = AsyncMock(return_value=(["/tmp/a.m4a"], ["audio/mp4"]))
        adapter.handle_message = AsyncMock()
        await adapter._ingest_with_media({}, "+15551234567", "", "MM1", 1)
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VOICE and event.media_urls == ["/tmp/a.m4a"]

    @pytest.mark.asyncio
    async def test_slash_commands_dropped_unless_allowed(self):
        adapter = self._make_adapter()
        body = b"From=%2B15551234567&To=%2B15550001111&Body=%2Fnew&MessageSid=SM1"
        request = MagicMock(); request.read = AsyncMock(return_value=body); request.headers = {}; request.content_length = None
        with patch.dict(os.environ, adapter._env):
            resp = await adapter._handle_webhook(request)
            await asyncio.sleep(0)
        assert resp.status == 200
        adapter._message_handler.assert_not_called()
