"""The shared Nous store never rolls a profile back to an OLDER, already-redeemed token pair.

Refresh tokens are single-use with reuse detection: a profile that adopts a stale shared copy
replays a redeemed refresh token on its next refresh and is forced to sign in again.
"""

import base64
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

import hermes_cli.auth as auth_mod
from hermes_cli import auth_nous


def _jwt(tag: str, seconds: int) -> str:
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    claims = {"sub": "user-1", "scope": auth_mod.DEFAULT_NOUS_SCOPE, "exp": int(time.time() + seconds),
              "tag": tag}
    return ".".join([b64({"alg": "none"}), b64(claims), "sig"])


def _iso(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _state(tag: str, refresh_token: str, *, issued: int, ttl: int = 3000) -> dict:
    token = _jwt(tag, issued + ttl)
    return {
        "portal_base_url": auth_mod.DEFAULT_NOUS_PORTAL_URL, "client_id": "hermes-cli",
        "token_type": "Bearer", "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        "access_token": token, "refresh_token": refresh_token, "obtained_at": _iso(issued),
        "expires_in": ttl, "expires_at": _iso(issued + ttl),
        "agent_key": token, "agent_key_expires_at": _iso(issued + ttl)}


@pytest.fixture
def homes(tmp_path, monkeypatch):
    hermes_home, shared = tmp_path / "hermes", tmp_path / "shared"
    hermes_home.mkdir()
    shared.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared))
    return hermes_home, shared


def _seed(hermes_home, profile_state, shared_state):
    (hermes_home / "auth.json").write_text(json.dumps(
        {"version": 1, "active_provider": "nous", "providers": {"nous": profile_state}}))
    auth_nous._write_shared_nous_state(shared_state, renewal=False)


def _profile_refresh_token(hermes_home) -> str:
    return json.loads((hermes_home / "auth.json").read_text())["providers"]["nous"]["refresh_token"]


def test_older_shared_state_never_replaces_profile_and_is_healed_from_it(homes, monkeypatch):
    hermes_home, _ = homes
    # Profile refreshed RT1 -> RT2 but the shared write was lost: shared still holds RT1/K1.
    _seed(hermes_home, _state("K2", "RT2", issued=0), _state("K1", "RT1", issued=-1000))
    posted = []

    def fake_refresh(*, client, portal_base_url, client_id, refresh_token):
        posted.append(refresh_token)
        return {"access_token": _jwt("K3", 3600), "refresh_token": "RT3", "expires_in": 3600,
                "token_type": "Bearer", "scope": auth_mod.DEFAULT_NOUS_SCOPE}

    monkeypatch.setattr(auth_nous, "_refresh_access_token", fake_refresh)
    monkeypatch.setattr(auth_mod, "_refresh_access_token", fake_refresh, raising=False)

    # A valid-token read takes the merge path without refreshing: the profile keeps its own pair
    # and the stale shared copy is brought up to date from it.
    auth_mod.resolve_nous_access_token()
    assert _profile_refresh_token(hermes_home) == "RT2"
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RT2"

    # The next refresh redeems the profile's live token, never the redeemed RT1.
    auth_mod.resolve_nous_runtime_credentials(force_refresh=True)
    assert posted == ["RT2"]
    assert _profile_refresh_token(hermes_home) == "RT3"


def test_newer_shared_state_is_adopted_and_a_stale_renewal_cannot_overwrite_it(homes):
    hermes_home, _ = homes
    newer, older = _state("K2", "RT2", issued=0), _state("K1", "RT1", issued=-1000)
    _seed(hermes_home, older, newer)

    # Control: a sibling's NEWER rotation in the shared store is still adopted.
    profile = dict(older)
    with auth_nous._nous_shared_store_lock():
        assert auth_nous._merge_shared_nous_oauth_state(profile) is True
    assert profile["refresh_token"] == "RT2"

    # A renewal write carrying the older pair leaves the newer shared copy in place.
    auth_nous._write_shared_nous_state(older, renewal=True)
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RT2"
