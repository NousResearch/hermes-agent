"""The email adapter's sender identity is exactly one addr-spec, and only its ASCII letters are case-folded.

The identity parsed from From: becomes the session's chat_id, the allowlist key and the reply To:, while sender
authentication only proves the domain.
"""

import os
import unittest
from unittest.mock import patch

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter, _extract_email_address


class TestSenderIdentityParsing(unittest.TestCase):

    def test_bracketed_from_with_more_than_one_address_is_rejected(self):
        # Each of these reached the single-bracket fallback and came back as a multi-recipient "address".
        for raw in ("Doe <victim@example.com,attacker@evil.test>",
                    "Doe <victim@example.com;attacker@evil.test>",
                    "Doe <victim@example.com@evil.test>"):
            self.assertEqual(_extract_email_address(raw), "", raw)
        # The fallback still serves the unquoted display names it exists for.
        self.assertEqual(_extract_email_address("Doe, John <John@Example.com>"), "john@example.com")

    @patch.dict(os.environ, {"EMAIL_ADDRESS": "hermes@example.com", "EMAIL_PASSWORD": "pw",
                             "EMAIL_IMAP_HOST": "imap.example.com", "EMAIL_SMTP_HOST": "smtp.example.com",
                             "EMAIL_ALLOWED_USERS": "kevin@example.com"}, clear=False)
    def test_non_ascii_local_part_does_not_fold_onto_an_allowlisted_address(self):
        adapter = EmailAdapter(PlatformConfig(enabled=True))
        authenticated = {"sender_authenticated": True}
        # U+212A KELVIN SIGN lowercases to ASCII "k" under str.lower().
        spoof = _extract_email_address("Kevin <Kevin@example.com>")
        self.assertNotEqual(spoof, "kevin@example.com")
        self.assertFalse(adapter._sender_accepted(spoof, authenticated))
        # Mixed-case ASCII still folds onto the allowlist entry.
        sender = _extract_email_address("Kevin <Kevin@Example.COM>")
        self.assertEqual(sender, "kevin@example.com")
        self.assertTrue(adapter._sender_accepted(sender, authenticated))


if __name__ == "__main__":
    unittest.main()
