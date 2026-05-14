"""Tests for xAI Coding Plan OAuth credentials imported from the Grok CLI."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from hermes_cli.auth import (
    AuthError,
    _is_xai_token_expiring,
    get_xai_coding_plan_auth_status,
    resolve_xai_coding_plan_credentials,
)


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def _write_xai_auth_state(hermes_home: Path, *, token: str = "access", expires_at: str = "2099-01-01T00:00:00Z") -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "xai-coding-plan",
                "providers": {
                    "xai-coding-plan": {
                        "access_token": token,
                        "refresh_token": "refresh",
                        "expires_at": expires_at,
                        "base_url": "https://api.x.ai/v1",
                    }
                },
            }
        )
    )


def test_xai_token_expiry_uses_stored_expiry_and_jwt_claims():
    assert _is_xai_token_expiring("access", expires_at="2000-01-01T00:00:00Z") is True
    assert _is_xai_token_expiring("access", expires_at="2099-01-01T00:00:00Z") is False
    assert _is_xai_token_expiring(_jwt_with_exp(int(time.time()) - 10)) is True
    assert _is_xai_token_expiring(_jwt_with_exp(int(time.time()) + 3600)) is False


def test_resolve_xai_credentials_raises_when_expired_refresh_fails(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_xai_auth_state(
        hermes_home,
        token="expired-token",
        expires_at="2000-01-01T00:00:00Z",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.auth._import_grok_auth_tokens", lambda: None)

    with pytest.raises(AuthError) as exc:
        resolve_xai_coding_plan_credentials()

    assert exc.value.code == "xai_token_expired"
    assert exc.value.relogin_required is True


def test_xai_auth_status_uses_single_probe_for_valid_stored_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_xai_auth_state(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.auth._import_grok_auth_tokens",
        lambda: (_ for _ in ()).throw(AssertionError("should not re-import fresh token")),
    )

    calls = {"count": 0}

    def _verify(token: str) -> bool:
        calls["count"] += 1
        assert token == "access"
        return True

    monkeypatch.setattr("hermes_cli.auth._verify_xai_token", _verify)

    status = get_xai_coding_plan_auth_status()

    assert status == {"logged_in": True}
    assert calls["count"] == 1
