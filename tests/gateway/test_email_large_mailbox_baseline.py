"""Baseline behaviour for large mailboxes.

The email adapter used to baseline by enumerating every existing UID with `UID SEARCH ALL`. That
response is a single line, so on a large INBOX it exceeds imaplib's 1 MB read limit and raises
`got more than 1000000 bytes` — the connect then fails on every retry, permanently, without the
mailbox ever becoming usable (observed on a 149k-message INBOX: ~1.2 MB response).

Contract pinned here: the first-connect baseline is derived from the SELECT response's UIDNEXT, so
mailbox size cannot break the connect; and messages that predate the baseline are never fetched,
even though they stay UNSEEN on the server and therefore appear in every UNSEEN search.
"""

import asyncio
import os
import unittest
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import imaplib


def _make_adapter():
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


class TestBaselineSurvivesLargeMailbox(unittest.TestCase):
    def test_baseline_uses_uidnext_instead_of_enumerating_uids(self):
        """A mailbox too large for an ALL-UID search must still connect."""
        adapter = _make_adapter()

        mock_imap = MagicMock()
        mock_imap.response.return_value = ("UIDNEXT", [b"300000"])
        # This is what a 150k-message INBOX does to the old baseline path.
        mock_imap.uid.side_effect = imaplib.IMAP4.error(b"got more than 1000000 bytes")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            ok = adapter._probe_imap(is_reconnect=False)

        self.assertTrue(ok, "connect must not depend on enumerating every UID")
        self.assertEqual(adapter._seen_up_to, 299999)
        mock_imap.uid.assert_not_called()

    def test_baseline_falls_back_to_uid_search_when_uidnext_is_absent(self):
        """Servers that omit UIDNEXT keep the previous behaviour."""
        adapter = _make_adapter()

        mock_imap = MagicMock()
        mock_imap.response.return_value = ("NO", [None])
        mock_imap.uid.return_value = ("OK", [b"11 12 13"])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            ok = adapter._probe_imap(is_reconnect=False)

        self.assertTrue(ok)
        self.assertIsNone(adapter._seen_up_to)
        self.assertEqual(adapter._seen_uids, {b"11", b"12", b"13"})


class TestPrebaselineUnreadMailIsNotFetched(unittest.TestCase):
    def test_only_uids_after_the_watermark_are_fetched(self):
        """Old UNSEEN mail must not be fetched on every poll.

        The adapter never sets \\Seen on the server, so pre-existing unread mail reappears in every
        UNSEEN search; the watermark is what stops it being re-downloaded forever.
        """
        adapter = _make_adapter()
        adapter._seen_up_to = 100

        raw_email = MIMEText("Hello", "plain", "utf-8")
        raw_email["From"] = "user@test.com"
        raw_email["Subject"] = "New"
        raw_email["Message-ID"] = "<new@test.com>"

        mock_imap = MagicMock()
        fetched = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"50 150"])
            if command == "fetch":
                fetched.append(args[0])
                return ("OK", [(b"150", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(fetched, [b"150"], "pre-baseline UID 50 must never be fetched")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "user@test.com")


if __name__ == "__main__":
    unittest.main()
