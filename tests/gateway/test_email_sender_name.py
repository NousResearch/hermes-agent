"""Outbound From: header carries a configurable display name.

A bare address renders nameless in mail clients and worsens spam-filter
placement; EMAIL_SENDER_NAME / platforms.email.extra.sender_name give the
operator a per-instance display name (default "Hermes"), RFC 2047-encoded
for non-ASCII names via formataddr.
"""

import os
import unittest
from unittest.mock import patch


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter
    cfg = PlatformConfig(enabled=True)
    if extra is not None:
        cfg.extra = extra
    return EmailAdapter(cfg)


class TestSenderNameHeader(unittest.TestCase):

    def test_default_sender_name(self):
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            adapter = _make_adapter()
        self.assertEqual(adapter._sender_name, "Hermes")
        self.assertEqual(adapter._from_header(), "Hermes <hermes@test.com>")

    def test_env_overrides_default(self):
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SENDER_NAME": "Dev Gateway",
        }):
            adapter = _make_adapter()
        self.assertEqual(adapter._from_header(), "Dev Gateway <hermes@test.com>")

    def test_extra_config_used_when_env_unset(self):
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            adapter = _make_adapter(extra={"sender_name": "VPS Bot"})
        self.assertEqual(adapter._from_header(), "VPS Bot <hermes@test.com>")

    def test_non_ascii_name_is_rfc2047_encoded(self):
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SENDER_NAME": "Эркюль",
        }):
            adapter = _make_adapter()
        header = adapter._from_header()
        self.assertIn("hermes@test.com", header)
        self.assertNotIn("Эркюль", header)  # encoded, not raw UTF-8

    def test_reply_uses_display_name(self):
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SENDER_NAME": "Dev Gateway",
        }):
            adapter = _make_adapter()
        msg, _msg_id, _subject = adapter._new_reply("user@test.com", "hi")
        self.assertEqual(msg["From"], "Dev Gateway <hermes@test.com>")


if __name__ == "__main__":
    unittest.main()
