"""SMTP authentication redaction and connection cleanup regressions for Email."""

import smtplib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.email import adapter as email_adapter


SMTP_USERNAME = "smtp-user@auth.invalid"
SMTP_PASSWORD = "smtp-secret-synthetic"
LEGACY_ADDRESS = "legacy@test.invalid"
LEGACY_PASSWORD = "legacy-secret-synthetic"
RAW_AUTH_RESPONSE = (
    f"Authentication failed with SMTP password {SMTP_PASSWORD} "
    f"and legacy password {LEGACY_PASSWORD}"
)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("EMAIL_ADDRESS", LEGACY_ADDRESS)
    monkeypatch.setenv("EMAIL_PASSWORD", LEGACY_PASSWORD)
    monkeypatch.setenv("EMAIL_IMAP_HOST", "imap.test.invalid")
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.test.invalid")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("EMAIL_SMTP_USERNAME", SMTP_USERNAME)
    monkeypatch.setenv("EMAIL_SMTP_PASSWORD", SMTP_PASSWORD)
    return email_adapter.EmailAdapter(PlatformConfig(enabled=True))


@pytest.fixture
def auth_failing_smtp():
    smtp = MagicMock()
    smtp.login.side_effect = smtplib.SMTPAuthenticationError(
        535, RAW_AUTH_RESPONSE.encode()
    )
    return smtp


def _assert_sanitized(*surfaces):
    for surface in surfaces:
        value = str(surface or "")
        assert "SMTP authentication failed" in value
        assert "535" in value
        assert "EMAIL_SMTP_USERNAME" in value
        assert "EMAIL_SMTP_PASSWORD" in value
        assert "EMAIL_ADDRESS" in value
        assert "EMAIL_PASSWORD" in value
        for exposed in (
            SMTP_USERNAME,
            LEGACY_ADDRESS,
            SMTP_PASSWORD,
            LEGACY_PASSWORD,
            RAW_AUTH_RESPONSE,
        ):
            assert exposed not in value


@pytest.mark.asyncio
async def test_startup_auth_failure_survives_quit_failure_and_closes(
    adapter, auth_failing_smtp, monkeypatch, caplog
):
    imap = MagicMock()
    imap.uid.return_value = ("OK", [b""])
    monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", lambda *a, **k: imap)
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: auth_failing_smtp)
    auth_failing_smtp.quit.side_effect = RuntimeError(
        f"quit reflected {SMTP_PASSWORD} and {LEGACY_PASSWORD}"
    )

    with caplog.at_level("ERROR", logger=email_adapter.__name__):
        connected = await adapter.connect()

    assert connected is False
    assert adapter.fatal_error_code == "email_auth_error"
    assert adapter.fatal_error_retryable is False
    auth_failing_smtp.quit.assert_called_once_with()
    auth_failing_smtp.close.assert_called_once_with()
    _assert_sanitized(adapter.fatal_error_message, caplog.text)


@pytest.mark.asyncio
async def test_normal_reply_auth_failure_redacts_result_and_logs(
    adapter, auth_failing_smtp, monkeypatch, caplog
):
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: auth_failing_smtp)

    with caplog.at_level("ERROR", logger=email_adapter.__name__):
        result = await adapter.send("recipient@test.invalid", "hello")

    assert result.success is False
    auth_failing_smtp.quit.assert_called_once_with()
    _assert_sanitized(result.error, caplog.text)


@pytest.mark.asyncio
async def test_multi_attachment_auth_failure_redacts_fallback_logs(
    adapter, auth_failing_smtp, monkeypatch, caplog, tmp_path
):
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-image")
    fallback = AsyncMock()
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: auth_failing_smtp)
    monkeypatch.setattr(BasePlatformAdapter, "send_multiple_images", fallback)

    with caplog.at_level("ERROR", logger=email_adapter.__name__):
        await adapter.send_multiple_images(
            "recipient@test.invalid",
            [(image.as_uri(), "image caption")],
        )

    auth_failing_smtp.quit.assert_called_once_with()
    fallback.assert_awaited_once()
    _assert_sanitized(caplog.text)


@pytest.mark.asyncio
async def test_single_document_auth_failure_redacts_result_and_logs(
    adapter, auth_failing_smtp, monkeypatch, caplog, tmp_path
):
    document = tmp_path / "document.txt"
    document.write_text("content")
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: auth_failing_smtp)

    with caplog.at_level("ERROR", logger=email_adapter.__name__):
        result = await adapter.send_document(
            "recipient@test.invalid", str(document), "document caption"
        )

    assert result.success is False
    auth_failing_smtp.quit.assert_called_once_with()
    _assert_sanitized(result.error, caplog.text)


def _registered_standalone_sender():
    ctx = MagicMock()
    email_adapter.register(ctx)
    return ctx.register_platform.call_args.kwargs["standalone_sender_fn"]


@pytest.mark.asyncio
async def test_registered_standalone_auth_failure_cleans_up_and_preserves_sanitized_result(
    adapter, auth_failing_smtp, monkeypatch
):
    del adapter  # Fixture supplies the scoped SMTP and legacy credentials.
    auth_failing_smtp.quit.side_effect = RuntimeError(
        f"quit reflected {SMTP_PASSWORD} and {LEGACY_PASSWORD}"
    )
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", lambda *a, **k: auth_failing_smtp)

    result = await _registered_standalone_sender()(
        SimpleNamespace(extra={}), "recipient@test.invalid", "hello"
    )

    assert result.get("success") is not True
    auth_failing_smtp.quit.assert_called_once_with()
    auth_failing_smtp.close.assert_called_once_with()
    _assert_sanitized(result.get("error"))


@pytest.mark.asyncio
async def test_registered_standalone_generic_failure_still_cleans_up(
    adapter, monkeypatch
):
    del adapter  # Fixture supplies the scoped SMTP and legacy credentials.
    smtp = MagicMock()
    smtp.send_message.side_effect = RuntimeError("generic send failed")
    smtp.quit.side_effect = RuntimeError("generic quit failed")
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", lambda *a, **k: smtp)

    result = await _registered_standalone_sender()(
        SimpleNamespace(extra={}), "recipient@test.invalid", "hello"
    )

    assert result.get("success") is not True
    assert "generic send failed" in result.get("error", "")
    smtp.quit.assert_called_once_with()
    smtp.close.assert_called_once_with()
