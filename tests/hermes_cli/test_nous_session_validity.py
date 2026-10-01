
import auth.providers.nous_status as _auth_auth_providers_nous_status

from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

import auth.constants as _auth_auth_constants
import auth.providers.nous as _auth_auth_providers_nous
"""Tests for the local-only Nous session classifier exposed on /api/status."""
import auth.provider_state as auth_provider_state

import base64
import json
import time

import hermes_cli.auth as auth
import hermes_cli.auth_nous as auth_nous


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
                    "scope": _auth_auth_constants.DEFAULT_NOUS_SCOPE,
                    "exp": int(time.time() + seconds),
                }
            ),
            "signature",
        )
    )


def _fail_if_live_auth_is_used(*args, **kwargs):
    raise AssertionError("session validity must not resolve or refresh credentials")


def _block_live_auth(monkeypatch):
    monkeypatch.setattr(_auth_auth_providers_nous_status, "get_nous_auth_status", _fail_if_live_auth_is_used)
    monkeypatch.setattr(
        _auth_auth_providers_nous,
        "resolve_nous_runtime_credentials",
        _fail_if_live_auth_is_used,
    )
    monkeypatch.setattr(
        _auth_auth_providers_nous,
        "resolve_nous_runtime_credentials",
        _fail_if_live_auth_is_used,
    )






# ── get_nous_auth_status_local — refresh-free display snapshot ──


def test_local_status_not_logged_in_after_terminal_quarantine(monkeypatch):
    monkeypatch.setattr(
        auth_provider_state,
        "get_provider_auth_state",
        lambda provider: {
            "last_auth_error": {
                "relogin_required": True,
                "code": "invalid_grant",
            },
        },
    )
    _block_live_auth(monkeypatch)

    status = _auth_auth_providers_nous_status.get_nous_auth_status_local(environment=_phase6_auth_environment())
    assert status["logged_in"] is False
    assert status["relogin_required"] is True
    assert status["error_code"] == "invalid_grant"


def test_local_status_repeated_polling_never_uses_live_auth(monkeypatch):
    monkeypatch.setattr(
        auth_provider_state,
        "get_provider_auth_state",
        lambda provider: {
            "access_token": _invoke_jwt(),
            "refresh_token": "rt",
            "scope": _auth_auth_constants.DEFAULT_NOUS_SCOPE,
        },
    )
    _block_live_auth(monkeypatch)

    assert all(
        _auth_auth_providers_nous_status.get_nous_auth_status_local(environment=_phase6_auth_environment())["logged_in"] for _ in range(10)
    )
