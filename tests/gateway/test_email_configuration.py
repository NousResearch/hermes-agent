import pytest

from gateway.config import PlatformConfig
from gateway.platforms import email as email_platform


@pytest.mark.parametrize("blank", ["", "   "])
def test_email_requirements_treat_blank_values_as_missing(monkeypatch, blank):
    for name in (
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
    ):
        monkeypatch.setenv(name, blank)

    assert email_platform.check_email_requirements() is False


@pytest.mark.asyncio
async def test_email_connect_fails_fast_without_retryable_blank_host(monkeypatch):
    monkeypatch.setenv("EMAIL_ADDRESS", "agent@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "app-password")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "")
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.example.com")

    def fail_if_called(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("blank IMAP host should be rejected before connecting")

    monkeypatch.setattr(email_platform.imaplib, "IMAP4_SSL", fail_if_called)

    adapter = email_platform.EmailAdapter(PlatformConfig(enabled=True))

    assert await adapter.connect() is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "email_missing_configuration"
    assert adapter.fatal_error_retryable is False
    assert "EMAIL_IMAP_HOST" in (adapter.fatal_error_message or "")
