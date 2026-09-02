"""Regression: Desktop/profile auth.json symlink must share the xAI lock.

Profile homes that symlink auth.json to the root store previously flocked a
distinct profiles/<name>/auth.json.lock while writing the same inode. Desktop
spawns --profile <name> plus --profile default; the loser of a single-use xAI
refresh then quarantined (wiped) the shared grant and broke CLI too.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.auth import (
    AuthError,
    _auth_lock_path,
    _canonical_auth_path,
    resolve_xai_oauth_runtime_credentials,
)


def _seed(hermes_home: Path, *, access: str = "dead-access", refresh: str = "dead-refresh") -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "xai-oauth",
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": access,
                            "refresh_token": refresh,
                            "token_type": "Bearer",
                        },
                        "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
                        "auth_mode": "oauth_device_code",
                    }
                },
            },
            indent=2,
        )
    )
    return auth_file


@pytest.mark.require_symlinks
def test_auth_lock_path_follows_symlink(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _seed(root)
    profile = tmp_path / "profiles" / "jn-design"
    profile.mkdir(parents=True)
    (profile / "auth.json").symlink_to(root / "auth.json")

    monkeypatch.setenv("HERMES_HOME", str(profile))
    lock = _auth_lock_path()
    assert lock == (root / "auth.json").resolve().with_suffix(".lock")
    assert _canonical_auth_path(profile / "auth.json") == (root / "auth.json").resolve()


def test_quarantine_adopts_peer_rotation_instead_of_wiping(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    auth_file = _seed(hermes_home, refresh="rt-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _peer_then_fail(tokens, **kwargs):
        payload = json.loads(auth_file.read_text())
        payload["providers"]["xai-oauth"]["tokens"] = {
            "access_token": "peer-access",
            "refresh_token": "rt-new",
            "token_type": "Bearer",
        }
        auth_file.write_text(json.dumps(payload, indent=2))
        raise AuthError(
            "xAI token refresh failed. Response: invalid_grant",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _peer_then_fail)

    creds = resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert creds["api_key"] == "peer-access"

    raw = json.loads(auth_file.read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    assert tokens["refresh_token"] == "rt-new"
    assert tokens["access_token"] == "peer-access"
    assert "last_auth_error" not in raw["providers"]["xai-oauth"]


def test_quarantine_still_clears_tokens_when_grant_is_dead(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    auth_file = _seed(hermes_home, refresh="rt-dead")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _terminal(tokens, **kwargs):
        raise AuthError(
            "xAI token refresh failed. Response: invalid_grant",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _terminal)

    creds = resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert creds["api_key"] == "dead-access"
    raw = json.loads(auth_file.read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    assert tokens["access_token"] == "dead-access"
    assert tokens["refresh_token"] == "rt-dead"
    assert "last_auth_error" not in raw["providers"]["xai-oauth"]


def test_quarantine_keeps_unexpired_jwt_after_invalid_grant(tmp_path, monkeypatch):
    import base64
    import time

    exp = int(time.time()) + 3600
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).rstrip(b"=").decode()
    access = f"h.{payload}.s"
    hermes_home = tmp_path / "hermes"
    auth_file = _seed(hermes_home, access=access, refresh="rt-spent")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _terminal(tokens, **kwargs):
        raise AuthError(
            "xAI token refresh failed. Response: invalid_grant",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _terminal)

    creds = resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert creds["api_key"] == access
    raw = json.loads(auth_file.read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    assert tokens["access_token"] == access
    assert tokens["refresh_token"] == "rt-spent"
    assert "last_auth_error" not in raw["providers"]["xai-oauth"]
