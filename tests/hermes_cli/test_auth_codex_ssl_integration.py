"""Integration tests: Codex login SSL failures _codex_login_post / _codex_poll_authorization_code."""

from __future__ import annotations

import ssl
import pytest
from unittest.mock import MagicMock


def test_codex_login_post_ssl_failure_includes_hint(monkeypatch):
    """_codex_login_post with SSL transport exception appends the PQ-middlebox hint."""
    from hermes_cli.auth_codex import _codex_login_post

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            raise ssl.SSLEOFError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred")

    monkeypatch.setattr(
        "hermes_cli.auth_codex.httpx.Client", lambda *a, **k: _FakeClient()
    )

    with pytest.raises(Exception) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/api/accounts/deviceauth/usercode",
            failure=("Failed to request device code", "device_code_request_failed"),
            json={"client_id": "test"},
        )

    msg = str(exc_info.value)
    assert "Failed to request device code" in msg
    assert "[SSL: UNEXPECTED_EOF_WHILE_READING]" in msg
    assert "post-quantum hybrid groups" in msg
    assert "OPENSSL_CONF" in msg


def test_codex_login_post_non_ssl_error_no_hint(monkeypatch):
    """_codex_login_post with plain ConnectError: no hint appended."""
    from hermes_cli.auth_codex import _codex_login_post

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            raise Exception("[Errno 61] Connection refused")

    monkeypatch.setattr(
        "hermes_cli.auth_codex.httpx.Client", lambda *a, **k: _FakeClient()
    )

    with pytest.raises(Exception) as exc_info:
        _codex_login_post(
            "https://auth.openai.com/api/accounts/deviceauth/usercode",
            failure=("Failed to request device code", "device_code_request_failed"),
            json={"client_id": "test"},
        )

    msg = str(exc_info.value)
    assert "Connection refused" in msg
    assert "post-quantum" not in msg
    assert "OPENSSL_CONF" not in msg


def test_codex_poll_ssl_failure_includes_hint(monkeypatch):
    """_codex_poll_authorization_code with SSL transport error → hint."""
    from hermes_cli.auth_codex import _codex_poll_authorization_code

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            raise ssl.SSLError("_ssl.c:999: The handshake operation timed out")

    monkeypatch.setattr(
        "hermes_cli.auth_codex.httpx.Client", lambda *a, **k: _FakeClient()
    )

    with pytest.raises(Exception) as exc_info:
        _codex_poll_authorization_code(
            "https://auth.openai.com",
            device_auth_id="dummy",
            user_code="CODE",
            poll_interval=1,
        )

    msg = str(exc_info.value)
    assert "Device auth poll transport failure" in msg
    assert "handshake operation timed out" in msg
    assert "middlebox" in msg


def test_codex_poll_non_ssl_error_no_hint(monkeypatch):
    """_codex_poll_authorization_code with plain ReadTimeout → no hint."""
    from hermes_cli.auth_codex import _codex_poll_authorization_code

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            raise Exception("ReadTimeout: exceeded 15.0s")

    monkeypatch.setattr(
        "hermes_cli.auth_codex.httpx.Client", lambda *a, **k: _FakeClient()
    )

    with pytest.raises(Exception) as exc_info:
        _codex_poll_authorization_code(
            "https://auth.openai.com",
            device_auth_id="dummy",
            user_code="CODE",
            poll_interval=1,
        )

    msg = str(exc_info.value)
    assert "ReadTimeout" in msg
    assert "post-quantum" not in msg
