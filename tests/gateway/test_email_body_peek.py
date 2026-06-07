"""Regression tests for IMAP UID FETCH using BODY.PEEK[] instead of RFC822.

Issue #41340: iCloud (and other modern IMAP servers) return metadata-only
responses for (RFC822), causing 'int object has no attribute decode' crash.
Fix: use (BODY.PEEK[]) with defensive shape validation and explicit Seen flag.
"""

import os
import unittest
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


def _make_adapter():
    """Create a minimal EmailAdapter with test config."""
    from gateway.config import PlatformConfig
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from gateway.platforms.email import EmailAdapter
        adapter = EmailAdapter(PlatformConfig(enabled=True))
    return adapter


def _build_rfc822_message(subject="Test", from_addr="sender@example.com", body="Hello"):
    """Build a raw RFC822 bytes payload for testing."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = "hermes@test.com"
    msg["Message-ID"] = "<msg123@example.com>"
    return msg.as_bytes()


class TestFetchNewMessagesBodyPeek(unittest.TestCase):
    """Tests for _fetch_new_messages using BODY.PEEK[] instead of RFC822."""

    def _setup_mock_imap(self, fetch_response):
        """Create a mock IMAP connection with given fetch response shape."""
        mock_imap = MagicMock()
        mock_imap.uid.side_effect = lambda cmd, *args: {
            "search": ("OK", [b"396"]),
            "fetch": ("OK", fetch_response),
            "store": ("OK", [b"Flags updated"]),
        }.get(cmd, ("OK", []))
        return mock_imap

    def test_fetch_uses_body_peek_not_rfc822(self):
        """_fetch_new_messages should use (BODY.PEEK[]) not (RFC822)."""
        adapter = _make_adapter()
        raw_msg = _build_rfc822_message()
        # Simulate a well-formed BODY.PEEK[] response: tuple with (header, literal)
        fetch_resp = [(b"396 (UID 396 BODY[] {234}", raw_msg), b")"]

        mock_imap = self._setup_mock_imap(fetch_resp)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email._send_imap_id"):
            results = adapter._fetch_new_messages()

        # Verify the fetch command used BODY.PEEK[]
        uid_calls = [c for c in mock_imap.uid.call_args_list if c[0][0] == "fetch"]
        self.assertTrue(len(uid_calls) > 0, "Expected at least one fetch call")
        fetch_arg = uid_calls[0][0][2]  # third positional arg is the fetch item
        self.assertIn("BODY.PEEK", fetch_arg,
                      f"Expected BODY.PEEK in fetch item, got: {fetch_arg}")
        self.assertNotIn("RFC822", fetch_arg,
                         f"RFC822 should not be used, got: {fetch_arg}")

    def test_fetch_handles_metadata_only_response_gracefully(self):
        """When server returns metadata-only (no literal), skip without crashing.

        This is the iCloud behavior: (RFC822) returns [b'30 (UID 396)']
        which is a single bytes element, not a tuple. bytes[1] is an int,
        causing 'int object has no attribute decode'.
        With the fix, such responses should be silently skipped.
        """
        adapter = _make_adapter()
        # Metadata-only response: list with a single bytes element, not a tuple
        fetch_resp = [b"396 (UID 396 FLAGS (\\Seen))"]

        mock_imap = self._setup_mock_imap(fetch_resp)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email._send_imap_id"):
            results = adapter._fetch_new_messages()

        # Should NOT crash; should return empty (no parseable messages)
        self.assertEqual(results, [])

    def test_fetch_marks_message_as_seen(self):
        """BODY.PEEK[] doesn't auto-mark as Seen; explicit STORE +FLAGS should follow."""
        adapter = _make_adapter()
        raw_msg = _build_rfc822_message()
        fetch_resp = [(b"396 (UID 396 BODY[] {234}", raw_msg), b")"]

        mock_imap = self._setup_mock_imap(fetch_resp)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email._send_imap_id"):
            results = adapter._fetch_new_messages()

        # Verify STORE +FLAGS (\Seen) was called after fetch
        store_calls = [c for c in mock_imap.uid.call_args_list if c[0][0] == "store"]
        self.assertTrue(len(store_calls) > 0,
                        "Expected a STORE command to mark message as Seen")
        store_args = store_calls[0]
        self.assertIn("+FLAGS", store_args[0],
                      "Expected +FLAGS in STORE command")
        self.assertIn("\\Seen", str(store_args),
                      "Expected \\Seen flag in STORE command")

    def test_fetch_extracts_message_from_body_peek_response(self):
        """Verify full message parsing works with BODY.PEEK[] response shape."""
        adapter = _make_adapter()
        raw_msg = _build_rfc822_message(
            subject="Test Subject",
            from_addr="Alice <alice@example.com>",
            body="Hello World"
        )
        fetch_resp = [(b"396 (UID 396 BODY[] {234}", raw_msg), b")"]

        mock_imap = self._setup_mock_imap(fetch_resp)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email._send_imap_id"):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["subject"], "Test Subject")
        self.assertEqual(results[0]["sender_addr"], "alice@example.com")
        self.assertIn("Hello World", results[0]["body"])

    def test_fetch_error_uses_logger_exception(self):
        """On fetch error, should use logger.exception (not logger.error) for stack trace."""
        adapter = _make_adapter()
        mock_imap = MagicMock()
        mock_imap.login.side_effect = Exception("connection refused")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email._send_imap_id"), \
             patch("gateway.platforms.email.logger") as mock_logger:
            results = adapter._fetch_new_messages()

        # Should have called logger.exception (not just logger.error)
        exception_calls = [c for c in mock_logger.method_calls
                           if "exception" in str(c)]
        self.assertTrue(len(exception_calls) > 0,
                        "Expected logger.exception call for fetch errors, "
                        f"got: {mock_logger.method_calls}")


if __name__ == "__main__":
    unittest.main()
