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


def test_repeated_status_checks_never_use_live_auth_resolution(monkeypatch):
    state = {
        "access_token": _invoke_jwt(),
        "refresh_token": "rt",
        "scope": auth.DEFAULT_NOUS_SCOPE,
    }
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda provider: state)
    _block_live_auth(monkeypatch)

    assert [get_nous_session_validity() for _ in range(10)] == [
        NOUS_SESSION_VALID
    ] * 10


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


def test_stale_quarantine_marker_ignored_after_relogin(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": _invoke_jwt(),
            "refresh_token": "new-rt",
            "scope": auth.DEFAULT_NOUS_SCOPE,
            "last_auth_error": {
                "relogin_required": True,
                "code": "invalid_grant",
            },
        },
    )
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_VALID


def test_expiring_token_is_unknown_without_refreshing(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": _invoke_jwt(seconds=30),
            "refresh_token": "rt",
            "scope": auth.DEFAULT_NOUS_SCOPE,
        },
    )
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_invalid_token_is_unknown_without_refreshing(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": "not-a-jwt",
            "refresh_token": "rt",
        },
    )
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_no_provider_state_is_unknown(monkeypatch):
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda provider: None)
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_provider_state_exception_is_unknown(monkeypatch):
    def _boom(provider):
        raise RuntimeError("disk error")

    monkeypatch.setattr(auth, "get_provider_auth_state", _boom)
    _block_live_auth(monkeypatch)

    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN
