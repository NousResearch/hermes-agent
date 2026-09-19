import imaplib
import smtplib
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter


@pytest.fixture
def adapter(monkeypatch):
    for key, value in {
        "EMAIL_ADDRESS": "bot@example.test", "EMAIL_PASSWORD": "test-password",
        "EMAIL_IMAP_HOST": "imap.example.test", "EMAIL_SMTP_HOST": "smtp.example.test",
    }.items():
        monkeypatch.setenv(key, value)
    return EmailAdapter(PlatformConfig(enabled=True))


@pytest.mark.parametrize("error, retryable", [
    (imaplib.IMAP4.error(b"[AUTHENTICATIONFAILED] Invalid credentials"), False),
    (imaplib.IMAP4.abort("connection closed"), True),
    (OSError("network unreachable"), True),
    (imaplib.IMAP4.error("[UNAVAILABLE] Try again later"), True),
    (imaplib.IMAP4.error("Too many simultaneous connections"), True),
])
def test_imap_failure_classification(monkeypatch, adapter, error, retryable):
    imap = MagicMock()
    imap.login.side_effect = error
    monkeypatch.setattr(adapter, "_connect_imap", lambda: imap)
    assert adapter._probe_imap(False) is False
    assert adapter.fatal_error_retryable is retryable
    imap.logout.assert_called_once()
    if not retryable:
        assert adapter.fatal_error_code == "email_auth_error"


@pytest.mark.parametrize("code, retryable", [(535, False), (454, True)])
@pytest.mark.parametrize("quit_fails", [False, True])
def test_smtp_auth_failure_classification(monkeypatch, adapter, code, retryable, quit_fails):
    smtp = MagicMock()
    smtp.login.side_effect = smtplib.SMTPAuthenticationError(code, b"authentication failed")
    if quit_fails:
        smtp.quit.side_effect = OSError("connection closed")
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: smtp)
    assert adapter._probe_smtp() is False
    assert adapter.fatal_error_retryable is retryable
    assert adapter.fatal_error_code == "email_auth_error"
    if quit_fails:
        smtp.close.assert_called_once()
