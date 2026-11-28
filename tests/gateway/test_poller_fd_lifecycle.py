"""Poller socket lifecycle — proxied fd-leak regressions (#79889).

On macOS (256 soft fd limit) a gateway routing weixin/email pollers through
a local HTTP proxy leaked one TCP socket per failed poll/connect cycle until
``[Errno 24] Too many open files`` crashed the gateway. Two code-side gaps:

1. email: ``imaplib.IMAP4.logout()`` only swallows ``OSError``; on a broken
   connection ``LOGOUT`` raises ``IMAP4.abort`` *before* the internal
   ``shutdown()``, so the socket stayed open. And ``connect()`` had no
   try/finally at all — a login/select failure abandoned the connected
   socket entirely.
2. weixin: repeated poll failures through a proxy strand sockets in the
   aiohttp connector (now covered in the weixin plugin's own tests).
"""

import asyncio
import imaplib
import os
import unittest
from unittest.mock import MagicMock, patch


def _make_email_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


class TestCloseImap(unittest.TestCase):
    """_close_imap must guarantee socket teardown."""


    def test_logout_abort_falls_back_to_shutdown(self):
        from plugins.platforms.email.adapter import _close_imap

        imap = MagicMock()
        imap.logout.side_effect = imaplib.IMAP4.abort("socket error: EOF")
        _close_imap(imap)
        imap.shutdown.assert_called_once()

    def test_shutdown_failure_is_swallowed(self):
        from plugins.platforms.email.adapter import _close_imap

        imap = MagicMock()
        imap.logout.side_effect = imaplib.IMAP4.abort("broken")
        imap.shutdown.side_effect = OSError("already closed")
        _close_imap(imap)  # must not raise


class TestEmailConnectClosesSocket(unittest.TestCase):
    """connect() must close the IMAP socket on every path, incl. failures."""

    def test_login_failure_still_closes_socket(self):
        adapter = _make_email_adapter()
        mock_imap = MagicMock()
        mock_imap.login.side_effect = imaplib.IMAP4.error("AUTHENTICATIONFAILED")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            result = asyncio.run(adapter.connect())

        self.assertFalse(result)
        # The failed handle must have been torn down (logout attempted;
        # abort fallback covered by TestCloseImap).
        mock_imap.logout.assert_called_once()



class TestFetchClosesSocketOnBrokenLogout(unittest.TestCase):
    def test_fetch_logout_abort_falls_back_to_shutdown(self):
        adapter = _make_email_adapter()
        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])
        mock_imap.logout.side_effect = imaplib.IMAP4.abort("EOF")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(results, [])
        mock_imap.shutdown.assert_called_once()
        # A teardown failure is not a fetch failure.
        self.assertFalse(adapter._last_fetch_failed)


if __name__ == "__main__":
    unittest.main()
