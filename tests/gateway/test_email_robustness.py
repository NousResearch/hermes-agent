"""Email adapter robustness against malformed IMAP responses (salvage of #2794).

Validates that:
- Malformed IMAP fetch responses are skipped instead of aborting the batch
  (UIDs are marked seen before fetch, so an abort permanently loses messages)
- Message-ID generation handles a missing '@' in EMAIL_ADDRESS
"""

import os
import unittest
import uuid
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


def _make_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        adapter = EmailAdapter(PlatformConfig(enabled=True))
    return adapter


def _raw_email(sender="user@test.com", subject="Hello"):
    msg = MIMEText("Test body", "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = subject
    msg["Message-ID"] = f"<{uuid.uuid4().hex[:8]}@test.com>"
    return msg.as_bytes()


class TestImapResponseGuard(unittest.TestCase):
    """_fetch_new_messages skips messages with unexpected IMAP structure."""

    def _fetch_with(self, fetch_responses):
        adapter = _make_adapter()
        uids = b" ".join(
            str(i + 1).encode() for i in range(len(fetch_responses))
        )
        fetch_iter = iter(fetch_responses)

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [uids])
            if command == "fetch":
                return next(fetch_iter)
            return ("NO", [])

        mock_imap = MagicMock()
        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            return adapter._fetch_new_messages()

    def test_normal_response_parses(self):
        results = self._fetch_with([("OK", [(b"1 (RFC822 {123}", _raw_email())])])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "user@test.com")

    def test_none_element_skipped(self):
        results = self._fetch_with([("OK", [None])])
        self.assertEqual(results, [])


class TestMessageIdDomain(unittest.TestCase):
    """Message-ID generation tolerates EMAIL_ADDRESS without '@'."""


    def test_address_without_at(self):
        adapter = _make_adapter("not-an-email")
        self.assertEqual(adapter._message_id_domain(), "localhost")


class TestTransportSecurity(unittest.TestCase):
    """platforms.email.extra.imap_security / smtp_security select the transport (#99641)."""

    def _adapter(self, **extra):
        from gateway.config import PlatformConfig

        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "127.0.0.1", "EMAIL_IMAP_PORT": "1143",
            "EMAIL_SMTP_HOST": "127.0.0.1", "EMAIL_SMTP_PORT": "1025",
        }, clear=True):
            from plugins.platforms.email.adapter import EmailAdapter

            return EmailAdapter(PlatformConfig(enabled=True, extra=extra))

    def test_starttls_builds_plain_imap_then_upgrades(self):
        adapter = self._adapter(imap_security="starttls", imap_tls_verify=False)
        imap = MagicMock()
        with patch("imaplib.IMAP4", return_value=imap) as imap_cls, \
             patch("imaplib.IMAP4_SSL") as imap_ssl_cls:
            self.assertIs(adapter._connect_imap(), imap)
        imap_cls.assert_called_once_with("127.0.0.1", 1143, timeout=30)
        imap_ssl_cls.assert_not_called()
        imap.starttls.assert_called_once()

    def test_unknown_mode_falls_back_to_secure_default(self):
        adapter = self._adapter(imap_security="bogus", smtp_security="bogus")
        self.assertEqual(adapter._imap_security, "tls")
        self.assertEqual(adapter._smtp_security, "starttls")  # port 1025 != 465
        # verification stays ON unless explicitly opted out
        self.assertTrue(adapter._imap_tls_verify)
        self.assertTrue(adapter._smtp_tls_verify)


if __name__ == "__main__":
    unittest.main()
