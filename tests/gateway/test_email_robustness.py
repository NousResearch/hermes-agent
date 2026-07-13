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

    def test_empty_list_skipped(self):
        results = self._fetch_with([("OK", [])])
        self.assertEqual(results, [])

    def test_bare_bytes_element_skipped(self):
        # Single bytes item instead of a (header, payload) tuple
        results = self._fetch_with([("OK", [b"not-a-tuple"])])
        self.assertEqual(results, [])

    def test_non_bytes_payload_skipped(self):
        results = self._fetch_with([("OK", [(b"1", None)])])
        self.assertEqual(results, [])

    def test_malformed_does_not_abort_batch(self):
        """A malformed response mid-batch must not lose the messages after it."""
        results = self._fetch_with([
            ("OK", [None]),                                # UID 1 malformed
            ("OK", [(b"2 (RFC822 {123}", _raw_email())]),  # UID 2 fine
        ])
        self.assertEqual(len(results), 1)


class TestMessageIdDomain(unittest.TestCase):
    """Message-ID generation tolerates EMAIL_ADDRESS without '@'."""

    def test_normal_address(self):
        adapter = _make_adapter("hermes@example.org")
        self.assertEqual(adapter._message_id_domain(), "example.org")

    def test_address_without_at(self):
        adapter = _make_adapter("not-an-email")
        self.assertEqual(adapter._message_id_domain(), "localhost")

    def test_address_trailing_at(self):
        adapter = _make_adapter("weird@")
        self.assertEqual(adapter._message_id_domain(), "localhost")


if __name__ == "__main__":
    unittest.main()
