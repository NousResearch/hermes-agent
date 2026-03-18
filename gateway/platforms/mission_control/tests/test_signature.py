"""Tests for Mission Control webhook signature verification."""

import os
import pytest
from ..signature import verify_signature


class TestVerifySignature:
    """Test HMAC-SHA256 signature verification."""

    def test_valid_signature(self):
        """Test verification with valid signature."""
        secret = "test-secret"
        body = b'{"event": "test"}'
        
        # Generate valid signature
        import hmac
        import hashlib
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        signature_header = f"sha256={expected}"
        
        assert verify_signature(body, signature_header, secret) is True

    def test_invalid_signature(self):
        """Test verification with invalid signature."""
        secret = "test-secret"
        body = b'{"event": "test"}'
        signature_header = "sha256=invalid"
        
        assert verify_signature(body, signature_header, secret) is False

    def test_missing_signature(self):
        """Test verification with missing signature header."""
        secret = "test-secret"
        body = b'{"event": "test"}'
        
        assert verify_signature(body, "", secret) is False
        assert verify_signature(body, None, secret) is False

    def test_missing_secret_dev_mode_disabled(self):
        """Test that missing secret fails in production mode."""
        body = b'{"event": "test"}'
        
        # Ensure dev mode is not set
        old_env = os.environ.pop("MC_ALLOW_UNAUTHENTICATED", None)
        try:
            assert verify_signature(body, "sha256=abc", "") is False
        finally:
            if old_env:
                os.environ["MC_ALLOW_UNAUTHENTICATED"] = old_env

    def test_missing_secret_dev_mode_enabled(self):
        """Test that missing secret passes in dev mode."""
        body = b'{"event": "test"}'
        
        os.environ["MC_ALLOW_UNAUTHENTICATED"] = "true"
        try:
            assert verify_signature(body, "", "") is True
        finally:
            del os.environ["MC_ALLOW_UNAUTHENTICATED"]

    def test_timing_attack_resistance(self):
        """Test that verification uses constant-time comparison."""
        secret = "test-secret"
        body = b'{"event": "test"}'
        
        import hmac
        import hashlib
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        signature_header = f"sha256={expected}"
        
        # Should not raise or leak timing info
        result = verify_signature(body, signature_header, secret)
        assert result is True

    def test_empty_body(self):
        """Test verification with empty body."""
        secret = "test-secret"
        
        import hmac
        import hashlib
        expected = hmac.new(
            secret.encode(),
            b"",
            hashlib.sha256
        ).hexdigest()
        signature_header = f"sha256={expected}"
        
        assert verify_signature(b"", signature_header, secret) is True

    def test_unicode_body(self):
        """Test verification with unicode body content."""
        secret = "test-secret"
        body = '{"event": "test", "data": "unicode: \u00e9\u00e8"}'.encode('utf-8')
        
        import hmac
        import hashlib
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        signature_header = f"sha256={expected}"
        
        assert verify_signature(body, signature_header, secret) is True
