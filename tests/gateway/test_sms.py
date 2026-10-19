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



# ── Requirements check ─────────────────────────────────────────────



# ── Toolset verification ───────────────────────────────────────────

# ── Webhook host configuration ─────────────────────────────────────



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
        # A whitespace-only token reads as unset: signatures forged with the blank or empty key are refused.
        blank = self._make_adapter(auth_token="   ")
        assert not any(blank._validate_twilio_signature(url, params, _compute_twilio_signature(k, url, params))
                       for k in ("   ", ""))
        from plugins.platforms.sms.adapter import check_sms_requirements
        with patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "   "}):
            assert check_sms_requirements() is False


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

    @pytest.mark.asyncio
    async def test_standalone_send_with_a_non_string_or_blank_key_fails_closed(self):
        """A YAML int api_key must not raise, and a blank one must not reach Twilio."""
        from plugins.platforms.sms.adapter import _standalone_send

        env = {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "", "TWILIO_PHONE_NUMBER": "+15550001111"}
        with patch.dict(os.environ, env):
            result = await _standalone_send(PlatformConfig(enabled=True, api_key="   "), "+15550002222", "hi")
            assert "not configured" in result["error"]
            env["TWILIO_ACCOUNT_SID"] = ""  # keep the int case off the network
            with patch.dict(os.environ, env):
                result = await _standalone_send(PlatformConfig(enabled=True, api_key=123), "+15550002222", "hi")
        assert "not configured" in result["error"]


@pytest.mark.asyncio
async def test_oversized_cron_output_reaches_twilio_in_1600_char_chunks():
    """The router hands SMS the full cron payload and send() splits it under Twilio's 1600-char
    cap (a larger Body is rejected with 21617, which used to fail the whole delivery)."""
    from gateway.config import GatewayConfig
    from gateway.delivery import DeliveryRouter
    from plugins.platforms.sms.adapter import SmsAdapter

    with patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "tok",
                                 "TWILIO_PHONE_NUMBER": "+15550001111"}):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="tok"))
    bodies = []

    class _Resp:
        status = 201
        async def json(self):
            return {"sid": "SM1"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, data, headers):
            bodies.append(next(value for opts, _, value in data._fields if opts["name"] == "Body"))
            return _Resp()

    adapter._http_session = _Session()
    content = "\n\n".join(f"line {i} " + "x" * 200 for i in range(40))
    payload = DeliveryRouter(GatewayConfig())._cap_oversized_output(adapter, content, "job")
    result = await adapter.send("+15550002222", payload)

    assert result.success
    assert len(bodies) > 1 and max(map(len, bodies)) <= 1600
    assert "line 39 " in bodies[-1]


@pytest.mark.asyncio
async def test_failure_after_delivered_chunks_is_never_resent_whole():
    """Twilio rejects chunk 3 of a long reply. The chunks already on the phone must not arrive again
    via the retry/plain-text fallback, which would also drop the tail (it resends content[:3500])."""
    from plugins.platforms.sms.adapter import SmsAdapter

    with patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "tok",
                                 "TWILIO_PHONE_NUMBER": "+15550001111"}):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="tok"))
    bodies = []

    class _Resp:
        def __init__(self, status):
            self.status = status
        async def json(self):
            return {"sid": f"SM{len(bodies)}"} if self.status < 400 else {"message": "rejected"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, data, headers):
            bodies.append(next(value for opts, _, value in data._fields if opts["name"] == "Body"))
            return _Resp(400 if len(bodies) == 3 else 201)

    adapter._http_session = _Session()
    content = "\n\n".join(f"para {i:02d} " + "y" * 300 for i in range(20))
    result = await adapter._send_with_retry("+15550002222", content)

    assert not result.success
    assert result.raw_response["partial_overflow"] and result.raw_response["delivered_chunks"] == 2
    assert all("".join(bodies).count(f"para {i:02d} ") <= 1 for i in range(20)), "a delivered chunk was re-sent"
