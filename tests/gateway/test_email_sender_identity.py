"""The email adapter's sender identity is exactly one addr-spec, and only its ASCII letters are case-folded.

The identity parsed from From: becomes the session's chat_id, the allowlist key and the reply To:, while sender
authentication only proves the domain.
"""

import email
import os
import unittest
from email.utils import getaddresses
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter, _extract_email_address, _verify_sender_authentication


class TestSenderIdentityParsing(unittest.TestCase):

    def test_bracketed_from_with_more_than_one_address_is_rejected(self):
        # Each of these reached the single-bracket fallback and came back as a multi-recipient "address".
        for raw in ("Doe <victim@example.com,attacker@evil.test>",
                    "Doe <victim@example.com;attacker@evil.test>",
                    "Doe <victim@example.com@evil.test>",
                    "Doe <victim:attacker@evil.test>"):
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


_OPEN_ACCESS_ENV = {"EMAIL_ADDRESS": "hermes@bot.test", "EMAIL_PASSWORD": "pw", "EMAIL_IMAP_HOST": "imap.bot.test",
                    "EMAIL_SMTP_HOST": "smtp.bot.test", "EMAIL_AUTHSERV_ID": "mx.bot.test",
                    "EMAIL_ALLOW_ALL_USERS": "true", "EMAIL_ALLOWED_USERS": "", "GATEWAY_ALLOWED_USERS": ""}


class TestSenderIdentityEndToEnd(unittest.TestCase):
    """parse -> authentication -> sender gate -> reply recipients, with the attacker's own domain passing DMARC."""

    def _reply_recipients(self, from_value: str, auth_domain: str):
        """SMTP recipients of the reply to a From: the adapter accepted, or None when it refuses the message."""
        with patch.dict(os.environ, _OPEN_ACCESS_ENV, clear=False):
            adapter = EmailAdapter(PlatformConfig(enabled=True))
            raw = (f"From: {from_value}\r\nAuthentication-Results: mx.bot.test; dmarc=pass header.from={auth_domain}\r\n"
                   "Subject: hi\r\nMessage-ID: <m1@evil.com>\r\n\r\nbody").encode()
            msg_data = adapter._parse_fetched_message(b"1", raw)
            if msg_data is None or not adapter._sender_accepted(msg_data["sender_addr"], msg_data):
                return None
            smtp = MagicMock()
            with patch.object(adapter, "_connect_smtp", return_value=smtp):
                adapter._send_email(msg_data["sender_addr"], "reply")
            sent = smtp.send_message.call_args[0][0]
            return [a for _, a in getaddresses(sent.get_all("To", []))]

    def test_unterminated_quote_cannot_make_a_two_recipient_identity(self):
        # parseaddr hands the unterminated quoted text back verbatim as the "address".
        for from_value in ('"victim@x.test, attacker@evil.com', 'x:"victim@x.test, attacker@evil.com'):
            self.assertIsNone(self._reply_recipients(from_value, "evil.com"), from_value)

    def test_stray_quote_in_bracketed_address_cannot_become_a_bare_local_part(self):
        # ``victim"@evil.com`` re-parses as the domain-less recipient ``victim``.
        self.assertIsNone(self._reply_recipients('Doe, J <victim"@evil.com>', "evil.com"))

    def test_non_ascii_domain_does_not_authenticate_as_its_ascii_fold(self):
        # U+212A KELVIN SIGN: the identity keeps it, so authentication must not fold it to ``kx.com`` either.
        msg = email.message_from_string("From: Doe <a@\u212ax.com>\r\n"
                                        "Authentication-Results: mx.bot.test; dmarc=pass header.from=kx.com\r\n\r\nb")
        sender = _extract_email_address(msg["From"])
        self.assertEqual(sender, "a@\u212ax.com")
        self.assertFalse(_verify_sender_authentication(msg, sender, authserv_id="mx.bot.test")[0])

    def test_ordinary_sender_still_gets_one_reply_to_itself(self):
        self.assertEqual(self._reply_recipients("Doe, John <John@Evil.com>", "evil.com"), ["john@evil.com"])


if __name__ == "__main__":
    unittest.main()
