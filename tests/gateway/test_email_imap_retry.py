"""Invariant tests for transient IMAP failure retry in the Email adapter.

Regression for #80016: a single transient IMAP timeout (a provider blip) must be
retried and absorbed, not surfaced as a fatal error that crashes the gateway
process. Persistent failures must still escalate through the fatal hook so the
gateway's reconnect/backoff learns email is unhealthy.
"""

import os
import unittest
from email.mime.text import MIMEText
from unittest.mock import patch, MagicMock


def _make_adapter():
    from gateway.config import PlatformConfig
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_POLL_INTERVAL": "1",
    }):
        from plugins.platforms.email.adapter import EmailAdapter
        return EmailAdapter(PlatformConfig(enabled=True))


class TestEmailImapRetry(unittest.TestCase):
    def test_transient_timeout_recovers_without_fatal(self):
        """One transient IMAP timeout must be retried and absorbed — it must
        NOT set _last_fetch_failed (#80016 retry contract: the gateway's
        reconnect/backoff handles persistent outages, a one-off blip must not
        crash the process)."""
        adapter = _make_adapter()
        adapter._fetch_retry_delay = 0  # keep the test fast

        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "recovered"
        raw_email["Message-ID"] = "<recovered@test.com>"

        mock_imap = MagicMock()
        calls = []

        def uid_handler(command, *args):
            calls.append(command)
            if len(calls) == 1:
                # First search attempt: transient read timeout.
                raise OSError("read operation timed out")
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(preauthorize=lambda m: True)

        # The retry recovered the message; no fatal error was raised.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["subject"], "recovered")
        self.assertFalse(adapter._last_fetch_failed)

    def test_persistent_failure_still_fatal_after_retries(self):
        """Persistent IMAP failures must still surface through the fatal hook
        after the retry budget is spent (#80016)."""
        adapter = _make_adapter()
        adapter._fetch_retry_delay = 0

        mock_imap = MagicMock()
        mock_imap.uid.side_effect = OSError("read operation timed out")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(preauthorize=lambda m: True)

        self.assertEqual(results, [])
        self.assertTrue(adapter._last_fetch_failed)


if __name__ == "__main__":
    unittest.main()
