"""Send-only email mode tests.

Kept in a dedicated file so the send-only PR does not add hunks to the shared
``test_email.py`` (which upstream edits frequently and caused merge conflicts).

Send-only = EMAIL_ADDRESS + EMAIL_SMTP_HOST only:
  * empty EMAIL_IMAP_HOST  -> no IMAP probe / no inbound poll loop;
  * empty EMAIL_PASSWORD   -> SMTP is used without AUTH (internal relay).
A full IMAP+SMTP setup is unchanged.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_adapter(env):
    """Construct an EmailAdapter with ``env`` applied (reads config at __init__)."""
    with patch.dict(os.environ, env, clear=False):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        return EmailAdapter(PlatformConfig(enabled=True))


_SEND_ONLY_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "",
    "EMAIL_IMAP_HOST": "",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


class TestSendOnlyMode(unittest.TestCase):
    def test_smtp_probe_connects_without_login_when_no_password(self):
        """No password: the SMTP probe connects and quits but never authenticates."""
        adapter = _make_adapter(_SEND_ONLY_ENV)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            self.assertTrue(adapter._probe_smtp())
        mock_server.login.assert_not_called()
        mock_server.quit.assert_called_once()

    def test_connect_aborts_when_smtp_host_missing(self):
        """A missing SMTP host still returns False with a non-retryable config error so
        the gateway stops reconnecting instead of looping (#40715)."""
        adapter = _make_adapter({
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "",
        })
        result = asyncio.run(adapter.connect())
        self.assertFalse(result)
        self.assertTrue(adapter.has_fatal_error)
        self.assertEqual(adapter.fatal_error_code, "email_missing_configuration")
        self.assertFalse(adapter.fatal_error_retryable)
        self.assertIn("EMAIL_SMTP_HOST", adapter.fatal_error_message or "")

    def test_minimal_config_enables_platform(self):
        """EMAIL_ADDRESS + EMAIL_SMTP_HOST alone enable the platform; PASSWORD/IMAP are optional."""
        from plugins.platforms.email.adapter import check_email_requirements
        with patch.dict(os.environ, _SEND_ONLY_ENV, clear=False):
            self.assertTrue(check_email_requirements())


class TestSendOnlyStandaloneTool(unittest.TestCase):
    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    })
    def test_send_email_tool_no_auth_skips_login(self):
        """The standalone send path (send-only, no EMAIL_PASSWORD) sends without SMTP AUTH."""
        from plugins.platforms.email.adapter import _standalone_send as _email_send

        async def _send_email(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            result = asyncio.run(
                _send_email({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "user@test.com", "Hello")
            )

        self.assertTrue(result["success"])
        mock_server.login.assert_not_called()
        mock_server.send_message.assert_called_once()


_AUTH_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


class TestSmtpUsername(unittest.TestCase):
    """EMAIL_SMTP_USERNAME separates the SMTP AUTH login from the From address."""

    def test_smtp_username_defaults_to_address(self):
        adapter = _make_adapter(_AUTH_ENV)
        self.assertEqual(adapter._smtp_username, "hermes@test.com")

    def test_smtp_username_override_used_in_probe(self):
        adapter = _make_adapter({**_AUTH_ENV, "EMAIL_SMTP_USERNAME": "relay-login@test.com"})
        self.assertEqual(adapter._smtp_username, "relay-login@test.com")
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            self.assertTrue(adapter._probe_smtp())
        mock_server.login.assert_called_once_with("relay-login@test.com", "secret")

    def test_smtp_username_override_used_in_send(self):
        from email.mime.multipart import MIMEMultipart
        adapter = _make_adapter({**_AUTH_ENV, "EMAIL_SMTP_USERNAME": "relay-login@test.com"})
        with patch.object(adapter, "_connect_smtp") as mock_connect:
            mock_server = MagicMock()
            mock_connect.return_value = mock_server
            adapter._smtp_send(MIMEMultipart())
        mock_server.login.assert_called_once_with("relay-login@test.com", "secret")

    def test_standalone_send_uses_smtp_username_from_env(self):
        from plugins.platforms.email.adapter import _standalone_send as _email_send

        async def _send(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        env = {**_AUTH_ENV, "EMAIL_SMTP_USERNAME": "relay-login@test.com", "EMAIL_SMTP_PORT": "587"}
        with patch.dict(os.environ, env, clear=False), patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            result = asyncio.run(_send({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "u@test.com", "Hi"))
        self.assertTrue(result["success"])
        mock_server.login.assert_called_once_with("relay-login@test.com", "secret")

    def test_standalone_send_defaults_login_to_address(self):
        from plugins.platforms.email.adapter import _standalone_send as _email_send

        async def _send(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        with patch.dict(os.environ, {**_AUTH_ENV, "EMAIL_SMTP_PORT": "587"}, clear=False), patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            result = asyncio.run(_send({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "u@test.com", "Hi"))
        self.assertTrue(result["success"])
        mock_server.login.assert_called_once_with("hermes@test.com", "secret")


if __name__ == "__main__":
    unittest.main()
