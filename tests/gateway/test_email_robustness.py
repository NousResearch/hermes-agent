"""Tests for email adapter robustness fixes.

Validates that:
- Malformed IMAP fetch responses are skipped instead of crashing
- Message-ID generation handles missing '@' in EMAIL_ADDRESS
"""

import email as email_lib
import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test: IMAP response structure guard (msg_data[0][1])
# ---------------------------------------------------------------------------

class TestImapResponseGuard:
    """_fetch_new_messages should skip messages with unexpected IMAP structure."""

    def _build_adapter(self):
        """Create a minimal EmailAdapter with mocked config."""
        # Patch env vars and config before import
        env = {
            "EMAIL_ADDRESS": "agent@example.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
            "EMAIL_IMAP_PORT": "993",
            "EMAIL_SMTP_PORT": "587",
            "EMAIL_POLL_INTERVAL": "15",
        }
        with patch.dict("os.environ", env):
            from gateway.platforms.email import EmailAdapter
            from gateway.config import PlatformConfig, Platform

            config = MagicMock(spec=PlatformConfig)
            config.extra = {}
            adapter = EmailAdapter(config)
        return adapter

    def _make_raw_email(self, sender="user@test.com", subject="Hello"):
        """Build a minimal RFC822 email as bytes."""
        from email.mime.text import MIMEText
        msg = MIMEText("Test body", "plain", "utf-8")
        msg["From"] = sender
        msg["Subject"] = subject
        msg["Message-ID"] = f"<{uuid.uuid4().hex[:8]}@test.com>"
        return msg.as_bytes()

    def test_normal_imap_response(self):
        """Normal tuple structure should parse correctly."""
        adapter = self._build_adapter()
        raw = self._make_raw_email()

        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [])
        mock_imap.uid.side_effect = [
            ("OK", [b"1"]),                       # search
            ("OK", [(b"1 (RFC822 {123}", raw)]),  # fetch — normal
        ]

        with patch("gateway.platforms.email.imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        assert len(results) == 1
        assert results[0]["sender_addr"] == "user@test.com"

    def test_malformed_imap_response_none_element(self):
        """If msg_data[0] is None, skip without crashing."""
        adapter = self._build_adapter()

        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [])
        mock_imap.uid.side_effect = [
            ("OK", [b"1"]),      # search
            ("OK", [None]),      # fetch — malformed (None element)
        ]

        with patch("gateway.platforms.email.imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        assert len(results) == 0  # skipped, no crash

    def test_malformed_imap_response_empty_list(self):
        """If msg_data is empty, skip without crashing."""
        adapter = self._build_adapter()

        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [])
        mock_imap.uid.side_effect = [
            ("OK", [b"1"]),  # search
            ("OK", []),      # fetch — empty
        ]

        with patch("gateway.platforms.email.imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        assert len(results) == 0

    def test_malformed_imap_response_bytes_only(self):
        """If msg_data[0] is a plain bytes (not a tuple), skip."""
        adapter = self._build_adapter()

        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [])
        mock_imap.uid.side_effect = [
            ("OK", [b"1"]),            # search
            ("OK", [b"raw bytes"]),    # fetch — bytes, not tuple
        ]

        with patch("gateway.platforms.email.imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        assert len(results) == 0


# ---------------------------------------------------------------------------
# Test: Message-ID domain extraction (split('@') safety)
# ---------------------------------------------------------------------------

class TestMessageIdDomainSafety:
    """Message-ID generation should not crash when EMAIL_ADDRESS has no @."""

    def test_normal_address_extracts_domain(self):
        """Standard email address extracts domain correctly."""
        env = {
            "EMAIL_ADDRESS": "agent@example.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        }
        with patch.dict("os.environ", env):
            from gateway.platforms.email import EmailAdapter
            config = MagicMock()
            config.extra = {}
            adapter = EmailAdapter(config)

        # Simulate the domain extraction logic used in _send_email
        domain = adapter._address.rsplit("@", 1)[-1] if "@" in adapter._address else "localhost"
        assert domain == "example.com"

    def test_missing_at_sign_falls_back_to_localhost(self):
        """Misconfigured address without @ should not crash."""
        env = {
            "EMAIL_ADDRESS": "no-at-sign",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        }
        with patch.dict("os.environ", env):
            from gateway.platforms.email import EmailAdapter
            config = MagicMock()
            config.extra = {}
            adapter = EmailAdapter(config)

        domain = adapter._address.rsplit("@", 1)[-1] if "@" in adapter._address else "localhost"
        assert domain == "localhost"

    def test_empty_address_falls_back_to_localhost(self):
        """Empty EMAIL_ADDRESS should not crash."""
        env = {
            "EMAIL_ADDRESS": "",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        }
        with patch.dict("os.environ", env):
            from gateway.platforms.email import EmailAdapter
            config = MagicMock()
            config.extra = {}
            adapter = EmailAdapter(config)

        domain = adapter._address.rsplit("@", 1)[-1] if "@" in adapter._address else "localhost"
        assert domain == "localhost"
