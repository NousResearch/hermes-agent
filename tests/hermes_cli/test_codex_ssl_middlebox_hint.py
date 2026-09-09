"""Codex device-login SSL EOF / handshake-timeout middlebox+PQ hint (#106384).

OpenSSL 3.5+ may offer X25519MLKEM768; some middleboxes drop the larger TLS 1.3
ClientHello. Hermes must preserve the underlying SSL error and append a
workaround hint — never silently force TLS 1.2 or disable PQ globally.
"""

from __future__ import annotations

import ssl

import httpx
import pytest

from hermes_cli.auth_codex import (
    _codex_login_post,
    _codex_poll_authorization_code,
    _pq_middlebox_ssl_hint,
)
from hermes_cli.auth_constants import AuthError


_SSL_EOF_MSG = (
    "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol "
    "(_ssl.c:1016)"
)
_HANDSHAKE_TIMEOUT_SSL_C = "handshake operation timed out (_ssl.c:997)"
_PQ_MARKERS = ("OPENSSL_CONF", "X25519MLKEM768", "ML-KEM")


def _boom_client(exc: BaseException):
    def factory(**kwargs):
        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, *a, **k):
                raise exc

        return _Client()

    return factory


def _assert_pq_hint(msg: str) -> None:
    assert any(marker in msg for marker in _PQ_MARKERS), msg


def _assert_no_pq_hint(msg: str) -> None:
    assert "OPENSSL_CONF" not in msg
    assert "X25519MLKEM768" not in msg
    assert "ML-KEM" not in msg
    assert "middlebox" not in msg.lower()


def test_codex_login_post_appends_pq_middlebox_hint_on_ssl_eof(monkeypatch):
    class Boom(ssl.SSLError):
        pass

    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(Boom(1, _SSL_EOF_MSG)),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Failed to request device code", "device_code_request_failed"),
        )
    msg = str(exc_info.value)
    assert "UNEXPECTED_EOF_WHILE_READING" in msg
    _assert_pq_hint(msg)
    assert exc_info.value.code == "device_code_request_failed"


def test_codex_login_post_no_hint_on_plain_connect_error(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(httpx.ConnectError("Failed to resolve 'auth.openai.com'")),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Failed to request device code", "device_code_request_failed"),
        )
    msg = str(exc_info.value)
    assert "Failed to resolve" in msg
    _assert_no_pq_hint(msg)


def test_codex_login_post_hints_on_ssleoferror_message(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(ssl.SSLEOFError(8, "EOF occurred in violation of protocol")),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Token exchange failed", "token_exchange_failed"),
        )
    msg = str(exc_info.value)
    assert "EOF occurred in violation of protocol" in msg
    _assert_pq_hint(msg)


def test_codex_login_post_hints_on_ssl_c_handshake_timeout(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(ssl.SSLError(1, _HANDSHAKE_TIMEOUT_SSL_C)),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Failed to request device code", "device_code_request_failed"),
        )
    msg = str(exc_info.value)
    assert "handshake operation timed out" in msg
    assert "_ssl.c:" in msg
    _assert_pq_hint(msg)


def test_codex_login_post_hints_on_wrapped_httpx_ssl_eof(monkeypatch):
    ssl_err = ssl.SSLError(1, _SSL_EOF_MSG)
    wrapped = httpx.ConnectError("Connection failed")
    wrapped.__cause__ = ssl_err
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(wrapped),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Failed to request device code", "device_code_request_failed"),
        )
    msg = str(exc_info.value)
    assert "UNEXPECTED_EOF_WHILE_READING" in msg or "Connection failed" in msg
    _assert_pq_hint(msg)


def test_codex_login_post_no_hint_on_ambiguous_timeout(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(httpx.ConnectTimeout("timed out")),
    )
    with pytest.raises(AuthError) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/x",
            failure=("Failed to request device code", "device_code_request_failed"),
        )
    _assert_no_pq_hint(str(exc_info.value))


def test_codex_poll_ssl_eof_shaped_through_transport_helper(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        _boom_client(ssl.SSLError(1, _SSL_EOF_MSG)),
    )
    monkeypatch.setattr("hermes_cli.auth_codex.time.sleep", lambda *_a, **_k: None)
    with pytest.raises(AuthError) as exc_info:
        _codex_poll_authorization_code(
            "https://auth.openai.com",
            device_auth_id="dev",
            user_code="CODE",
            poll_interval=0,
        )
    msg = str(exc_info.value)
    assert "UNEXPECTED_EOF_WHILE_READING" in msg
    _assert_pq_hint(msg)
    assert exc_info.value.code == "device_code_poll_error"


def test_codex_poll_http_status_error_has_no_pq_hint(monkeypatch):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            return httpx.Response(500, request=httpx.Request("POST", "https://auth.openai.com/x"))

    monkeypatch.setattr(
        "hermes_cli.auth_codex._codex_http_client",
        lambda **kwargs: _Client(),
    )
    monkeypatch.setattr("hermes_cli.auth_codex.time.sleep", lambda *_a, **_k: None)
    with pytest.raises(AuthError) as exc_info:
        _codex_poll_authorization_code(
            "https://auth.openai.com",
            device_auth_id="dev",
            user_code="CODE",
            poll_interval=0,
        )
    msg = str(exc_info.value)
    assert "status 500" in msg
    _assert_no_pq_hint(msg)
    assert exc_info.value.code == "device_code_poll_error"


def test_pq_hint_on_sslerror_handshake_timed_out_without_ssl_c():
    hint = _pq_middlebox_ssl_hint(ssl.SSLError("TLS handshake timed out"))
    assert hint
    _assert_pq_hint(hint)
    assert "x25519:secp256r1:secp384r1:x448" in hint
    assert "does not auto-rewrite" in hint


def test_pq_hint_absent_on_cert_verify_failed():
    hint = _pq_middlebox_ssl_hint(
        ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))
    assert hint == ""


def test_pq_hint_absent_on_plain_oserror():
    hint = _pq_middlebox_ssl_hint(OSError("[Errno 8] nodename nor servname provided"))
    assert hint == ""
