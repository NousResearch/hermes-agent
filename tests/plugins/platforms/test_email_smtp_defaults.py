"""Regression for #150806: the SMTP TLS default must follow the port, not a fixed list.

Implicit TLS ("tls") on 465 and on the ports that end in 465 (10465, 4465, ...): a provider
that mirrors implicit TLS on an alternate port answers a plaintext connect with no banner, so
the default STARTTLS probe waits out ``SMTP_CONNECT_TIMEOUT`` twice and the probe fails.

Every other port keeps STARTTLS — the submission ports (25, 587) and the documented local
relays (1025) need it — and an explicit ``EMAIL_SMTP_SECURITY`` always wins.

No mail server and no network connection is used: ``smtplib`` is patched and the transport
class each mode must pick is asserted.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter, _standalone_send


@pytest.mark.parametrize(
    ("port", "security", "expected"),
    [
        # Implicit TLS: 465 itself and the alternate ports that end in 465.
        (465, None, "tls"),
        (10465, None, "tls"),
        (4465, None, "tls"),
        # Everything else keeps STARTTLS — including the submission ports and the
        # documented local relays.
        (587, None, "starttls"),
        (25, None, "starttls"),
        (1025, None, "starttls"),
        # An explicit setting always wins over the port-derived default.
        (10465, "starttls", "starttls"),
        (587, "tls", "tls"),
        (10465, "plain", "plain"),
    ],
)
@pytest.mark.parametrize("standalone", [False, True])
def test_smtp_transport_default_and_override(monkeypatch, port, security, expected, standalone):
    monkeypatch.setenv("EMAIL_ADDRESS", "hermes@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", str(port))
    if security is None:
        monkeypatch.delenv("EMAIL_SMTP_SECURITY", raising=False)
    else:
        monkeypatch.setenv("EMAIL_SMTP_SECURITY", security)

    with patch("smtplib.SMTP") as plain_smtp, patch("smtplib.SMTP_SSL") as tls_smtp:
        if standalone:
            result = asyncio.run(_standalone_send(
                SimpleNamespace(extra={}), "recipient@example.com", "hello",
            ))
            assert result["success"] is True
        else:
            adapter = EmailAdapter(PlatformConfig(enabled=True))
            adapter._connect_smtp()

    if expected == "tls":
        tls_smtp.assert_called_once()
        plain_smtp.assert_not_called()
    else:
        plain_smtp.assert_called_once()
        tls_smtp.assert_not_called()
        if expected == "starttls":
            plain_smtp.return_value.starttls.assert_called_once()
        else:
            plain_smtp.return_value.starttls.assert_not_called()
