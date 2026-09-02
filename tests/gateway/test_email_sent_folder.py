"""Tests for the Email adapter's sent-folder copy (IMAP APPEND).

The SMTP path only transmits a message; it does not persist a copy in the
mailbox, so webmail clients (Roundcube, etc.) show an empty Sent folder even
though the mail was delivered. These tests cover the adapter's best-effort
IMAP APPEND of the exact bytes handed to SMTP, with the ``\\Seen`` flag so the
copy renders as read.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from gateway.config import PlatformConfig


class TestSentFolderAppend(unittest.TestCase):
    """Verify sent replies are copied to the Sent folder via IMAP APPEND."""

    def _make_adapter(self, extra_env=None):
        env = {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }
        if extra_env:
            env.update(extra_env)
        with patch.dict(os.environ, env):
            from plugins.platforms.email.adapter import EmailAdapter
            return EmailAdapter(PlatformConfig(enabled=True))

    def test_send_appends_copy_to_sent_folder(self):
        """A successful SMTP send is followed by an IMAP APPEND to Sent."""
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {
            "subject": "Project question",
            "message_id": "<original@test.com>",
        }

        mock_imap = MagicMock()
        mock_imap.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "Sent"'])

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            adapter._send_email("user@test.com", "Here is the answer.", None)

            # SMTP transmit happened.
            mock_server.send_message.assert_called_once()
            # IMAP APPEND happened with the Sent folder and \Seen flag.
            mock_imap.append.assert_called_once()
            args, _kwargs = mock_imap.append.call_args
            self.assertEqual(args[0], "Sent")
            self.assertEqual(args[1], "\\Seen")
            self.assertIsInstance(args[3], bytes)

    def test_sent_folder_explicit_env(self):
        """EMAIL_SENT_FOLDER overrides auto-detection."""
        adapter = self._make_adapter()
        # _sent_folder() reads the env var at call time, so keep it set here.
        with patch.dict(os.environ, {"EMAIL_SENT_FOLDER": "Custom Sent"}):
            self.assertEqual(adapter._sent_folder(), "Custom Sent")

    def test_sent_folder_explicit_config(self):
        """platforms.email.sent_folder (config.extra) overrides auto-detection."""
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra={"sent_folder": "Config Sent"}))
        self.assertEqual(adapter._sent_folder(), "Config Sent")

    def test_sent_folder_autodetect(self):
        """Auto-detect picks the first matching common folder name."""
        adapter = self._make_adapter()
        mock_imap = MagicMock()
        mock_imap.list.return_value = (
            "OK",
            [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Sent Items"'],
        )
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            self.assertEqual(adapter._sent_folder(), "Sent Items")

    def test_append_failure_does_not_fail_send(self):
        """A failed APPEND is best-effort and must not raise out of send."""
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {
            "subject": "Project question",
            "message_id": "<original@test.com>",
        }

        mock_imap = MagicMock()
        mock_imap.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "Sent"'])
        mock_imap.append.side_effect = Exception("append boom")

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            # Must not raise even though APPEND failed.
            adapter._send_email("user@test.com", "Here is the answer.", None)
            mock_server.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
