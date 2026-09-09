"""Regression tests: Codex device-login transport errors keep the underlying SSL detail.

On networks whose middlebox rejects the larger TLS 1.3 ClientHello that OpenSSL 3.5+ sends
(post-quantum hybrid groups), every device-login HTTP call dies with ``SSLEOFError`` /
handshake timeouts while curl still works (#106384). Previously:

* ``_codex_login_post`` swallowed the exception chain (no ``from exc``) and gave no hint;
* the polling loop let the raw ``httpx`` error escape with no ``AuthError`` shaping at all.

Both paths must keep the original SSL text, chain the cause, and append the OPENSSL_CONF
workaround hint so the failure stops masquerading as a Codex outage.
"""

import httpx
import pytest

from hermes_cli import auth_codex
from hermes_cli.auth import AuthError


_SSL_EOF_MESSAGE = (
    "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)")


class _RaisingClient:
    """Context-manager httpx.Client stand-in whose ``post`` always raises *exc*."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, url, **kwargs):
        raise self._exc


def _patch_client(monkeypatch, exc: BaseException) -> None:
    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kwargs: _RaisingClient(exc))


def test_login_post_ssl_error_keeps_detail_and_hint(monkeypatch):
    _patch_client(monkeypatch, httpx.ConnectError(_SSL_EOF_MESSAGE))

    with pytest.raises(AuthError) as excinfo:
        auth_codex._codex_login_post(
            "https://auth.openai.com/api/accounts/deviceauth/usercode",
            failure=("Failed to request device code", "device_code_request_failed"))

    err = excinfo.value
    assert err.code == "device_code_request_failed"
    assert _SSL_EOF_MESSAGE in str(err)
    assert "OPENSSL_CONF" in str(err)
    assert "#106384" in str(err)
    # ``raise ... from exc`` keeps the underlying transport error inspectable.
    assert isinstance(err.__cause__, httpx.ConnectError)


def test_login_post_non_ssl_error_has_no_interop_hint(monkeypatch):
    _patch_client(monkeypatch, httpx.ConnectError("Connection refused"))

    with pytest.raises(AuthError) as excinfo:
        auth_codex._codex_login_post(
            "https://auth.openai.com/api/accounts/deviceauth/usercode",
            failure=("Failed to request device code", "device_code_request_failed"))

    assert "Connection refused" in str(excinfo.value)
    assert "OPENSSL_CONF" not in str(excinfo.value)


def test_poll_authorization_code_ssl_error_surfaced_as_auth_error(monkeypatch):
    _patch_client(monkeypatch, httpx.ConnectError(_SSL_EOF_MESSAGE))

    with pytest.raises(AuthError) as excinfo:
        auth_codex._codex_poll_authorization_code(
            "https://auth.openai.com", device_auth_id="da", user_code="uc", poll_interval=0)

    err = excinfo.value
    assert err.code == "device_code_poll_error"
    assert _SSL_EOF_MESSAGE in str(err)
    assert "OPENSSL_CONF" in str(err)
    assert isinstance(err.__cause__, httpx.ConnectError)
