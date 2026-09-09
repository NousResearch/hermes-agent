"""Tests for hermes_cli/auth_codex_ssl_helper.py SSL/TLS middlebox hint logic.

Ensures _ssl_tls_middlebox_hint:
1. Returns a hint for ssl.SSLError / SSLEOFError / 'UNEXPECTED_EOF_WHILE_READING' / 'handshake timed out'
2. Returns None for non-SSL exceptions (ConnectError without SSL in message, httpx.TimeoutException)
"""

from __future__ import annotations

import ssl
import pytest


def test_ssl_error_recognized():
    """ssl.SSLError subclass → hint returned."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = ssl.SSLEOFError(
        "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
    )
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is not None
    assert "post-quantum hybrid groups" in hint
    assert "X25519MLKEM768" in hint
    assert "OPENSSL_CONF" in hint


def test_unexpected_eof_message_recognized():
    """Raw Exception with 'UNEXPECTED_EOF_WHILE_READING' in message → hint."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = Exception("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is not None
    assert "middlebox" in hint


def test_ssl_handshake_timeout_recognized():
    """Exception message 'handshake operation timed out' → hint."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = ssl.SSLError("_ssl.c:999: The handshake operation timed out")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is not None
    assert "TLS 1.3" in hint


def test_ssl_alert_recognized():
    """Exception message 'tlsv1 alert unknown ca' → hint."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = Exception("tlsv1 alert unknown ca")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is not None


def test_non_ssl_connect_error_no_hint():
    """Plain ConnectError without SSL signature → None."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = Exception("[Errno 61] Connection refused")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is None


def test_read_timeout_no_hint():
    """Timeout without SSL signature → None."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = Exception("ReadTimeout: Request timeout after 15.0s")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is None


def test_http_status_error_no_hint():
    """HTTP 429/500 status error → None."""
    from hermes_cli.auth_codex_ssl_helper import _ssl_tls_middlebox_hint

    exc = Exception("HTTP 429: Too Many Requests")
    hint = _ssl_tls_middlebox_hint(exc)

    assert hint is None
