"""Codex device-login transport errors keep the underlying SSL detail and add a hint (#106384).

OpenSSL 3.5+ advertises post-quantum hybrid groups; some middleboxes drop the resulting larger
TLS 1.3 ClientHello, so every device-login POST dies with ``SSLEOFError`` / handshake timeouts
while curl still works. Both transport paths (``_codex_login_post`` and the poll loop, which
previously let the raw ``httpx`` error escape unshaped) must surface a typed ``AuthError`` that
keeps the original text and appends the OPENSSL_CONF / TLS 1.2 hint — and only for SSL errors.
"""

import ssl

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


def _login_post():
    return auth_codex._codex_login_post(
        "https://auth.openai.com/api/accounts/deviceauth/usercode",
        failure=("Failed to request device code", "device_code_request_failed"))


def _poll():
    return auth_codex._codex_poll_authorization_code(
        "https://auth.openai.com", device_auth_id="da", user_code="uc", poll_interval=0)


@pytest.mark.parametrize(
    "call, code",
    [(_login_post, "device_code_request_failed"), (_poll, "device_code_poll_error")],
    ids=["login_post", "poll"])
def test_ssl_transport_error_keeps_detail_and_adds_hint(monkeypatch, call, code):
    exc = ssl.SSLEOFError(8, _SSL_EOF_MESSAGE)
    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kw: _RaisingClient(exc))

    with pytest.raises(AuthError) as excinfo:
        call()

    err = excinfo.value
    assert err.code == code
    assert "UNEXPECTED_EOF_WHILE_READING" in str(err)
    assert "OPENSSL_CONF" in str(err)
    assert err.__cause__ is exc


def test_plain_timeout_has_no_ssl_hint(monkeypatch):
    exc = httpx.ConnectTimeout("timed out")
    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kw: _RaisingClient(exc))

    with pytest.raises(AuthError) as excinfo:
        _login_post()

    assert "timed out" in str(excinfo.value)
    assert "OPENSSL_CONF" not in str(excinfo.value)
