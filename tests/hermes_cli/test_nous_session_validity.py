"""Tests for the local-only Nous session classifier exposed on /api/status."""

import base64
import json
import time

import hermes_cli.auth as auth
from hermes_cli.auth import (
    NOUS_SESSION_TERMINAL,
    NOUS_SESSION_UNKNOWN,
    NOUS_SESSION_VALID,
    get_nous_session_validity,
)


def _invoke_jwt(*, seconds: int = 3600) -> str:
    def _encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return ".".join(
        (
            _encode({"alg": "none", "typ": "JWT"}),
            _encode(
                {
                    "sub": "test-user",
                    "scope": auth.DEFAULT_NOUS_SCOPE,
                    "exp": int(time.time() + seconds),
                }
            ),
            "signature",
        )
    )


def _fail_if_live_auth_is_used(*args, **kwargs):
    raise AssertionError("session validity must not resolve or refresh credentials")


def _block_live_auth(monkeypatch):
    monkeypatch.setattr(auth, "get_nous_auth_status", _fail_if_live_auth_is_used)
    monkeypatch.setattr(
        auth,
        "resolve_nous_runtime_credentials",
        _fail_if_live_auth_is_used,
    )


def test_valid_when_local_invoke_jwt_is_usable(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": _invoke_jwt(),
            "refresh_token": "rt",
            "scope": auth.DEFAULT_NOUS_SCOPE,
        },
    )
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_VALID


def test_terminal_on_persisted_quarantine_marker(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "last_auth_error": {
                "relogin_required": True,
                "code": "invalid_grant",
            },
        },
    )
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_TERMINAL


# ── get_nous_auth_status_local — refresh-free display snapshot ──


def test_local_status_not_logged_in_after_terminal_quarantine(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "last_auth_error": {
                "relogin_required": True,
                "code": "invalid_grant",
            },
        },
    )
    _block_live_auth(monkeypatch)

    status = auth.get_nous_auth_status_local()
    assert status["logged_in"] is False
    assert status["relogin_required"] is True
    assert status["error_code"] == "invalid_grant"


def test_local_status_repeated_polling_never_uses_live_auth(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": _invoke_jwt(),
            "refresh_token": "rt",
            "scope": auth.DEFAULT_NOUS_SCOPE,
        },
    )
    _block_live_auth(monkeypatch)

    assert all(
        auth.get_nous_auth_status_local()["logged_in"] for _ in range(10)
    )
