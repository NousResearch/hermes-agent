"""A profile with no local Nous state must self-heal from the shared cross-profile grant.

Regression for the Nous credential loss seen on a multi-profile install: a profile whose local
``providers.nous`` section had been cleared (logout, quarantine of a dead grant, a freshly cloned
profile) raised "Hermes is not logged into Nous Portal." from
``resolve_nous_runtime_credentials()`` even while a live grant sat in
``<hermes-root>/shared/nous_auth.json``. The credential pool classifies that error as transient and
benches the only Nous credential for an hour, and the CLI then offers the first-run provider wizard
on a machine that is signed in.
"""

import base64
import json
import time
from datetime import datetime, timezone

import pytest

import hermes_cli.auth as auth_mod
from hermes_cli.auth_constants import AuthError

SHARED_REFRESH_TOKEN = "shared-grant-refresh-token"


def _jwt_part(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _invoke_jwt(*, seconds: int = 3600) -> str:
    return (
        f"{_jwt_part({'alg': 'none', 'typ': 'JWT'})}."
        f"{_jwt_part({'sub': 'test-user', 'scope': 'inference:invoke', 'exp': int(time.time() + seconds)})}.sig"
    )


def _iso(seconds: int) -> str:
    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).isoformat()


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """A profile home with NO ``providers.nous`` state plus a redirected shared store."""
    home = tmp_path / "home"
    home.mkdir()
    shared_dir = tmp_path / "shared"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))

    def write_shared_grant(access_token: str) -> None:
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "nous_auth.json").write_text(json.dumps({
            "_schema": 1,
            "access_token": access_token,
            "refresh_token": SHARED_REFRESH_TOKEN,
            "token_type": "Bearer",
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
            "client_id": "hermes-cli",
            "portal_base_url": "https://portal.nousresearch.com",
            "inference_base_url": "https://inference-api.nousresearch.com/v1",
            "obtained_at": _iso(-60),
            "expires_at": _iso(3600),
        }))

    return home, write_shared_grant


def test_empty_local_state_adopts_the_shared_grant(profile_home):
    """The shared grant is adopted (and persisted) instead of reporting a logout."""
    home, write_shared_grant = profile_home
    token = _invoke_jwt()
    write_shared_grant(token)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["provider"] == "nous"
    assert creds["api_key"] == token
    persisted = json.loads((home / "auth.json").read_text())["providers"]["nous"]
    assert persisted["refresh_token"] == SHARED_REFRESH_TOKEN
    assert persisted["agent_key"] == token


def test_empty_local_state_without_a_shared_grant_still_requires_a_login(profile_home):
    """No shared grant → the original relogin error is unchanged (no silent empty session)."""
    with pytest.raises(AuthError, match="not logged into Nous Portal"):
        auth_mod.resolve_nous_runtime_credentials()
