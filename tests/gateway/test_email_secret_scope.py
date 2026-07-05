"""Tests for email adapter credential isolation under multiplexing.

Verifies that the email adapter reads EMAIL_ADDRESS, EMAIL_PASSWORD,
EMAIL_IMAP_HOST, and EMAIL_SMTP_HOST from the profile-scoped secret
store (agent.secret_scope.get_secret) instead of os.getenv, so that
a secondary profile in a multiplexed gateway does not inherit the
default profile's email credentials via os.environ.

Related issues: #50051, #52307
Related PRs: #51374 (config.py api_server guard), #50094 (config.py scoped env reads)
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from agent import secret_scope as ss


class TestEmailAdapterSecretScope(unittest.TestCase):
    """Verify the email adapter honors the profile secret scope over os.environ."""

    def setUp(self):
        ss.set_multiplex_active(False)

    def tearDown(self):
        ss.set_multiplex_active(False)

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alpha@test.invalid",
        "EMAIL_PASSWORD": "default-pw",
        "EMAIL_IMAP_HOST": "imap.default.com",
        "EMAIL_SMTP_HOST": "smtp.default.com",
    }, clear=False)
    def test_adapter_uses_scoped_credentials_not_environ(self):
        """When a secret scope is installed, the adapter must read from it,
        not from os.environ which may hold another profile's values."""
        from gateway.config import PlatformConfig, Platform
        from plugins.platforms.email.adapter import EmailAdapter

        scoped = {
            "EMAIL_ADDRESS": "beta@test.invalid",
            "EMAIL_PASSWORD": "secondary-pw",
            "EMAIL_IMAP_HOST": "imap.secondary.example",
            "EMAIL_SMTP_HOST": "smtp.secondary.example",
        }
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scoped)
        try:
            cfg = PlatformConfig(enabled=True)
            adapter = EmailAdapter(cfg)
            self.assertEqual(adapter._address, "beta@test.invalid")
            self.assertEqual(adapter._password, "secondary-pw")
            self.assertEqual(adapter._imap_host, "imap.secondary.example")
            self.assertEqual(adapter._smtp_host, "smtp.secondary.example")
        finally:
            ss.reset_secret_scope(token)

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alpha@test.invalid",
        "EMAIL_PASSWORD": "default-pw",
        "EMAIL_IMAP_HOST": "imap.default.com",
        "EMAIL_SMTP_HOST": "smtp.default.com",
    }, clear=False)
    def test_adapter_falls_back_to_environ_without_scope(self):
        """Without a secret scope (single-profile mode), the adapter reads
        from os.environ — backward-compatible with legacy behavior."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        cfg = PlatformConfig(enabled=True)
        adapter = EmailAdapter(cfg)
        self.assertEqual(adapter._address, "alpha@test.invalid")
        self.assertEqual(adapter._password, "default-pw")
        self.assertEqual(adapter._imap_host, "imap.default.com")
        self.assertEqual(adapter._smtp_host, "smtp.default.com")

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alpha@test.invalid",
        "EMAIL_PASSWORD": "default-pw",
        "EMAIL_IMAP_HOST": "imap.default.com",
        "EMAIL_SMTP_HOST": "smtp.default.com",
    }, clear=False)
    def test_check_email_requirements_uses_scope(self):
        """check_email_requirements must also honor the secret scope so that
        a secondary profile with scoped email creds is detected as configured."""
        scoped = {
            "EMAIL_ADDRESS": "beta@test.invalid",
            "EMAIL_PASSWORD": "secondary-pw",
            "EMAIL_IMAP_HOST": "imap.secondary.example",
            "EMAIL_SMTP_HOST": "smtp.secondary.example",
        }
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scoped)
        try:
            from plugins.platforms.email.adapter import check_email_requirements
            self.assertTrue(check_email_requirements())
        finally:
            ss.reset_secret_scope(token)

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alpha@test.invalid",
        "EMAIL_PASSWORD": "default-pw",
        "EMAIL_IMAP_HOST": "imap.default.com",
        "EMAIL_SMTP_HOST": "smtp.default.com",
    }, clear=False)
    def test_adapter_scoped_missing_key_does_not_leak_environ(self):
        """If a key is absent from the scope but present in os.environ,
        the adapter must NOT fall through to os.environ (which would leak
        the default profile's value)."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        scoped = {
            "EMAIL_ADDRESS": "beta@test.invalid",
            # EMAIL_PASSWORD intentionally missing from scope
            "EMAIL_IMAP_HOST": "imap.secondary.example",
            "EMAIL_SMTP_HOST": "smtp.secondary.example",
        }
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scoped)
        try:
            cfg = PlatformConfig(enabled=True)
            adapter = EmailAdapter(cfg)
            self.assertEqual(adapter._address, "beta@test.invalid")
            # Password must NOT be the default profile's "default-pw"
            self.assertNotEqual(adapter._password, "default-pw")
            self.assertEqual(adapter._password, "")
        finally:
            ss.reset_secret_scope(token)

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alpha@test.invalid",
        "EMAIL_PASSWORD": "default-pw",
        "EMAIL_IMAP_HOST": "imap.default.com",
        "EMAIL_SMTP_HOST": "smtp.default.com",
        "EMAIL_ALLOWED_USERS": "gamma@test.invalid",
    }, clear=False)
    def test_allowed_users_uses_scope(self):
        """EMAIL_ALLOWED_USERS must also be read from the secret scope
        so a secondary profile gets its own allowlist, not the default's."""
        scoped = {
            "EMAIL_ADDRESS": "beta@test.invalid",
            "EMAIL_PASSWORD": "secondary-pw",
            "EMAIL_IMAP_HOST": "imap.secondary.example",
            "EMAIL_SMTP_HOST": "smtp.secondary.example",
            "EMAIL_ALLOWED_USERS": "epsilon@test.invalid,delta@test.invalid",
        }
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scoped)
        try:
            # _allowlist_in_effect reads EMAIL_ALLOWED_USERS — verify it
            # sees the scoped value, not the environ value
            from plugins.platforms.email.adapter import EmailAdapter
            self.assertTrue(EmailAdapter._allowlist_in_effect())
        finally:
            ss.reset_secret_scope(token)


if __name__ == "__main__":
    unittest.main()
