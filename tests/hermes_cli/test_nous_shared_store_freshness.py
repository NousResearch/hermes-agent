"""The shared Nous store never rolls a profile back to an OLDER, already-redeemed token pair.

Refresh tokens are single-use with reuse detection: a profile that adopts a stale shared copy
replays a redeemed refresh token on its next refresh and is forced to sign in again.
"""

import base64
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

import hermes_agent.hermes_cli.auth as auth_mod
from hermes_agent.hermes_cli import auth_nous


def _jwt(tag: str, seconds: int, *, issued: int | None = None, sub: str = "user-1") -> str:
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    claims = {"sub": sub, "scope": auth_mod.DEFAULT_NOUS_SCOPE, "exp": int(time.time() + seconds),
              "tag": tag}
    if issued is not None:
        claims["iat"] = int(time.time() + issued)
    return ".".join([b64({"alg": "none"}), b64(claims), "sig"])


def _iso(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _state(tag: str, refresh_token: str, *, issued: int, ttl: int = 3000, sub: str = "user-1",
           token: str | None = None) -> dict:
    token = token or _jwt(tag, issued + ttl, issued=issued, sub=sub)
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


def _single_use_portal(monkeypatch, ttl: int) -> list:
    """Fake refresh POST: single-use refresh tokens with reuse detection, issuing *ttl*-second keys."""
    posted: list = []

    def fake_refresh(*, client, portal_base_url, client_id, refresh_token):
        if refresh_token in posted:
            raise auth_mod.AuthError("Refresh token reuse detected", provider="nous", code="invalid_grant",
                                     relogin_required=True)
        posted.append(refresh_token)
        n = len(posted)
        return {"access_token": _jwt(f"K-new{n}", ttl, issued=0), "refresh_token": f"RT-new{n}",
                "expires_in": ttl, "token_type": "Bearer", "scope": auth_mod.DEFAULT_NOUS_SCOPE}

    monkeypatch.setattr(auth_nous, "_refresh_access_token", fake_refresh)
    monkeypatch.setattr(auth_mod, "_refresh_access_token", fake_refresh, raising=False)
    return posted


def test_a_later_rotation_with_a_shorter_ttl_is_still_the_newer_pair(homes, monkeypatch):
    """Expiry is not issue order: the Portal sets access TTLs per client and per issue."""
    hermes_home, _ = homes
    in_sync = _state("K1", "RT1", issued=-60, ttl=3600)
    _seed(hermes_home, in_sync, in_sync)
    posted = _single_use_portal(monkeypatch, ttl=900)  # the next rotation issues 15-minute keys

    auth_mod.resolve_nous_runtime_credentials(force_refresh=True)
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RT-new1"
    auth_mod.resolve_nous_runtime_credentials()
    auth_mod.resolve_nous_runtime_credentials(force_refresh=True)
    assert posted == ["RT1", "RT-new1"]  # the redeemed RT1 is never replayed
    assert _profile_refresh_token(hermes_home) == "RT-new2"

    # A sibling's later, shorter-lived rotation is adopted over this profile's longer-lived pair.
    _seed(hermes_home, _state("K2", "RT2", issued=-600, ttl=3600), _state("K3", "RT3", issued=-5, ttl=900))
    auth_mod.resolve_nous_runtime_credentials()
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RT3"  # never rolled back to RT2
    auth_mod.resolve_nous_runtime_credentials(force_refresh=True)
    assert posted[-1] == "RT3"


def test_two_rotations_in_one_iat_second_order_by_obtained_at_not_expiry(homes):
    """Equal whole-second ``iat``: the sub-second ``obtained_at`` decides, never the shorter TTL."""
    hermes_home, _ = homes
    iat = 2_000_000_000  # fixed clock: both rotations stamped in the same JWT second

    def rotation(tag, refresh_token, *, offset, ttl):
        def b64(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
        token = ".".join([b64({"alg": "none"}), b64(
            {"sub": "user-1", "scope": auth_mod.DEFAULT_NOUS_SCOPE, "iat": iat, "exp": iat + ttl,
             "tag": tag}), "sig"])
        state = _state(tag, refresh_token, issued=0, ttl=ttl, token=token)
        state["obtained_at"] = datetime.fromtimestamp(iat + offset, UTC).isoformat()
        return state

    first = rotation("K1", "RT1", offset=0.2, ttl=3600)
    later = rotation("K2", "RT2", offset=0.7, ttl=900)  # the later rotation carries a shorter TTL
    _seed(hermes_home, first, later)

    profile = dict(first)
    with auth_nous._nous_shared_store_lock():
        assert auth_nous._merge_shared_nous_oauth_state(profile) is True
    assert profile["refresh_token"] == "RT2"
    auth_nous._write_shared_nous_state(first, renewal=True)  # the spent pair never restores
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RT2"


def test_unknown_identities_never_push_over_the_shared_store(homes):
    hermes_home, _ = homes
    # No JWT ``sub`` on either side: unknown identities, so a newer profile is not pushed over shared.
    newer, older = (_state("A", "RTA", issued=-10, token="opaqueA"), _state("B", "RTB", issued=-600, token="opaqueB"))
    _seed(hermes_home, newer, older)
    with auth_nous._nous_shared_store_lock():
        assert auth_nous._merge_shared_nous_oauth_state(dict(newer)) is False
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RTB"


def test_a_profile_never_adopts_another_accounts_newer_shared_login(homes):
    """Profiles are islands: account B's shared login is not account A's next refresh token."""
    hermes_home, _ = homes
    _seed(hermes_home, _state("KA", "RTA", issued=-600, sub="uA"), _state("KB", "RTB", issued=-10, sub="uB"))
    auth_mod.resolve_nous_access_token()
    assert _profile_refresh_token(hermes_home) == "RTA"
    assert auth_nous._read_shared_nous_state()["refresh_token"] == "RTB"


def test_a_quarantined_profile_recovers_from_a_siblings_relogin(homes):
    """A quarantined profile has no tokens (so no known identity) and must still adopt the fresh login."""
    hermes_home, _ = homes
    quarantined = {k: v for k, v in _state("K-old", "RT-old", issued=-7200).items()
                   if k not in {"access_token", "refresh_token", "expires_at", "expires_in", "obtained_at",
                                "agent_key", "agent_key_expires_at"}}
    quarantined["last_auth_error"] = {"provider": "nous", "code": "invalid_grant", "reason": "test"}
    _seed(hermes_home, quarantined, _state("K-login", "RT-login", issued=-5))
    auth_mod.resolve_nous_access_token()
    assert _profile_refresh_token(hermes_home) == "RT-login"
