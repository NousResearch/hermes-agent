"""Test Twilio signature validation for SMS webhook security fix.

This test validates that the SMS webhook properly authenticates
incoming requests using the X-Twilio-Signature header.

Issue: #7089 - Unauthenticated Remote Code Execution via SMS Webhook
"""

import hashlib
import hmac
import base64
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock


def compute_twilio_signature(auth_token: str, url: str, form: dict) -> str:
    """Compute expected Twilio signature for test requests."""
    signed_data = url
    for key in sorted(form.keys()):
        values = form.get(key, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            signed_data += f"{key}{value}"
    
    return base64.b64encode(
        hmac.new(
            auth_token.encode("utf-8"),
            signed_data.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("ascii")


class TestTwilioSignatureValidation:
    """Test cases for _validate_twilio_signature method."""

    def test_valid_signature_passes(self):
        """Valid Twilio signature should be accepted."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        form = {
            "From": ["+15551234567"],
            "To": ["+15550001111"],
            "Body": ["test message"],
            "MessageSid": ["SM123456"],
        }
        
        # Compute valid signature
        valid_signature = compute_twilio_signature(auth_token, url, form)
        
        # Mock request
        request = Mock()
        request.url = url
        request.headers = {"X-Twilio-Signature": valid_signature}
        
        # Mock adapter
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        
        # Call actual method
        result = SmsAdapter._validate_twilio_signature(adapter, request, form)
        
        assert result is True, "Valid signature should pass validation"

    def test_missing_signature_rejected(self):
        """Request without X-Twilio-Signature header should be rejected."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        form = {
            "From": ["+15551234567"],
            "Body": ["malicious command"],
        }
        
        # Mock request without signature header
        request = Mock()
        request.url = url
        request.headers = {}  # No X-Twilio-Signature
        
        # Mock adapter
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        
        # Call actual method
        result = SmsAdapter._validate_twilio_signature(adapter, request, form)
        
        assert result is False, "Missing signature should be rejected"

    def test_invalid_signature_rejected(self):
        """Request with wrong signature should be rejected."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        form = {
            "From": ["+15551234567"],
            "Body": ["attack payload"],
        }
        
        # Mock request with invalid signature
        request = Mock()
        request.url = url
        request.headers = {"X-Twilio-Signature": "invalid_signature_xyz"}
        
        # Mock adapter
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        
        # Call actual method
        result = SmsAdapter._validate_twilio_signature(adapter, request, form)
        
        assert result is False, "Invalid signature should be rejected"

    def test_tampered_body_rejected(self):
        """Request with tampered body (wrong signature) should be rejected."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        
        # Original form used for signature
        original_form = {
            "From": ["+15551234567"],
            "Body": ["hello"],
        }
        
        # Tampered form sent in request
        tampered_form = {
            "From": ["+15551234567"],
            "Body": ["malicious command"],  # Changed!
        }
        
        # Compute signature for ORIGINAL (not tampered)
        valid_signature = compute_twilio_signature(auth_token, url, original_form)
        
        # Mock request with tampered body but original signature
        request = Mock()
        request.url = url
        request.headers = {"X-Twilio-Signature": valid_signature}
        
        # Mock adapter
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        
        # Call actual method with TAMPERED form
        result = SmsAdapter._validate_twilio_signature(adapter, request, tampered_form)
        
        assert result is False, "Tampered body should be rejected"

    def test_signature_injection_attack_rejected(self):
        """Attacker cannot inject additional parameters."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        
        # Legitimate form
        legitimate_form = {
            "From": ["+15551234567"],
            "Body": ["hello"],
        }
        
        # Attacker adds extra parameter
        injected_form = {
            "From": ["+15551234567"],
            "Body": ["hello"],
            "SmsSid": ["SM_ATTACK"],  # Injected!
        }
        
        # Compute signature for legitimate form
        valid_signature = compute_twilio_signature(auth_token, url, legitimate_form)
        
        # Mock request
        request = Mock()
        request.url = url
        request.headers = {"X-Twilio-Signature": valid_signature}
        
        # Mock adapter
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        
        # Call with injected form
        result = SmsAdapter._validate_twilio_signature(adapter, request, injected_form)
        
        assert result is False, "Injected parameters should be rejected"

    def test_configured_webhook_url_used(self):
        """When SMS_WEBHOOK_URL is set, it should be used for validation."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        
        # Public URL (behind reverse proxy)
        public_url = "https://example.com/webhooks/twilio"
        
        # Internal URL (actual request)
        internal_url = "http://localhost:8080/webhooks/twilio"
        
        form = {
            "From": ["+15551234567"],
            "Body": ["test message"],
        }
        
        # Compute signature using PUBLIC URL
        valid_signature = compute_twilio_signature(auth_token, public_url, form)
        
        # Mock request with INTERNAL URL
        request = Mock()
        request.url = internal_url  # Different from public URL!
        request.headers = {"X-Twilio-Signature": valid_signature}
        
        # Mock adapter WITH configured webhook URL
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        adapter._webhook_url = public_url  # Configured!
        
        # Call actual method
        result = SmsAdapter._validate_twilio_signature(adapter, request, form)
        
        assert result is True, "Configured webhook URL should be used for validation"

    def test_no_configured_url_uses_request_url(self):
        """When SMS_WEBHOOK_URL is NOT set, request URL should be used."""
        from gateway.platforms.sms import SmsAdapter
        
        auth_token = "test_auth_token_12345"
        url = "http://localhost:8080/webhooks/twilio"
        form = {
            "From": ["+15551234567"],
            "Body": ["test message"],
        }
        
        # Compute signature using same URL
        valid_signature = compute_twilio_signature(auth_token, url, form)
        
        # Mock request
        request = Mock()
        request.url = url
        request.headers = {"X-Twilio-Signature": valid_signature}
        
        # Mock adapter WITHOUT configured webhook URL
        adapter = Mock(spec=SmsAdapter)
        adapter._auth_token = auth_token
        adapter._webhook_url = None  # Not configured
        
        # Call actual method
        result = SmsAdapter._validate_twilio_signature(adapter, request, form)
        
        assert result is True, "Request URL should be used when webhook URL not configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])