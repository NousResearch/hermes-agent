"""Invariants for the declarative OAuth PKCE helper an out-of-tree provider plugin plugs into
``ProviderProfile.auth_handler`` / ``refresh_credential``."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import urlopen

import pytest

import hermes_cli.auth
import providers
from hermes_cli import auth_oauth_pkce_plugin as pkce
from hermes_cli.auth_constants import AuthError
from providers import register_provider
from providers.base import ProviderProfile
from tests.providers.fake_pkce_idp import FakeIdP

PROVIDER = "example-pkce"


@pytest.fixture
def idp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    server = FakeIdP().start()
    yield server
    server.stop()


def _config(idp: FakeIdP, **overrides) -> pkce.OAuthPKCEConfig:
    base = pkce.OAuthPKCEConfig(client_id="hermes-example", authorize_url=f"{idp.base}/authorize",
                                token_url=f"{idp.base}/token", scopes=("inference",), timeout_seconds=5)
    return replace(base, **overrides)


def _browser_hits(url: str) -> bool:
    with urlopen(url, timeout=5) as response:  # the fake IdP 302s straight to the loopback callback
        assert response.status == 200
    return True


def test_pkce_handler_add_status_refresh_logout_against_fake_idp(idp, monkeypatch, tmp_path, request):
    from agent.credential_pool import load_pool

    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler, refresh = pkce.pkce_auth_handler(cfg), pkce.pkce_refresh_credential(cfg)
    # Registered like a real plugin so the credential pool finds ``refresh_credential`` through the seam.
    register_provider(ProviderProfile(name=PROVIDER, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                      auth_handler=handler, refresh_credential=refresh))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(PROVIDER, None),
                                  hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None)))
    args = SimpleNamespace(provider=PROVIDER, no_browser=False)

    assert handler("add", args) is True
    exchange = idp.token_requests[-1]
    assert exchange["grant_type"] == "authorization_code" and exchange["code_verifier"]
    assert exchange["redirect_uri"].startswith("http://127.0.0.1:")
    rows = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER]
    assert [(r["auth_type"], r["source"], r["oauth_pkce"]["client_id"]) for r in rows] == [
        ("oauth", pkce.POOL_SOURCE, "hermes-example")]
    assert rows[0]["refresh_token"] in idp.refresh_tokens and rows[0]["expires_at_ms"]
    assert handler("status", args) is True
    assert handler("refresh", args) is False  # declined → the pool's generic refresh owns it

    # Pool-driven rotation through the real refresh path; the old single-use refresh token is gone.
    before = rows[0]["refresh_token"]
    pool = load_pool(PROVIDER)
    rotated = pool.try_refresh_matching(credential_id=rows[0]["id"])
    assert rotated is not None and rotated.refresh_token != before and rotated.access_token != rows[0]["access_token"]
    assert idp.token_requests[-1] == {"grant_type": "refresh_token", "refresh_token": before, "scope": "inference",
                                      "client_id": "hermes-example"}
    assert before not in idp.refresh_tokens
    # A peer that already rotated on disk is adopted without spending the refresh token again.
    stale = load_pool(PROVIDER).entries()[0]
    stale.refresh_token = before
    assert refresh(stale)["refresh_token"] == rotated.refresh_token
    assert idp.token_requests[-1]["refresh_token"] == before  # no new POST

    assert handler("logout", args) is True
    assert load_pool(PROVIDER).entries() == []

    # Negative: a callback whose state is not ours is rejected and never reaches the token endpoint.
    idp.override_state = "forged"
    posts = len(idp.token_requests)
    with pytest.raises(AuthError) as excinfo:
        handler("add", args)
    assert excinfo.value.code == "oauth_state_mismatch" and len(idp.token_requests) == posts


def test_token_url_off_authorize_allowlist_refused_before_any_request(idp, monkeypatch):
    from hermes_cli.auth_constants import httpx

    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("token endpoint must not be contacted")))
    cfg = _config(idp, token_url="https://token.attacker.example/token")
    with pytest.raises(AuthError) as login_err:
        pkce.login(PROVIDER, cfg, open_browser=False)
    entry = SimpleNamespace(provider=PROVIDER, id="abc123", refresh_token="rt", access_token="at", expires_at_ms=None)
    with pytest.raises(AuthError) as refresh_err:
        pkce.pkce_refresh_credential(cfg)(entry)
    assert {login_err.value.code, refresh_err.value.code} == {"oauth_token_host_rejected"}
    with pytest.raises(AuthError, match="HTTPS"):
        pkce.validate_config(PROVIDER, _config(idp, authorize_url="http://auth.example.com/authorize"))


def test_spent_refresh_token_is_grant_dead_and_marks_the_pool_row_dead(idp, monkeypatch, tmp_path, request):
    """RFC 6749 invalid_grant on refresh is terminal. The pool must DEAD the row, not EXHAUST it."""
    from agent.credential_pool import STATUS_DEAD, load_pool

    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler, refresh = pkce.pkce_auth_handler(cfg), pkce.pkce_refresh_credential(cfg)
    register_provider(ProviderProfile(name=PROVIDER, auth_type="oauth_external",
                                      base_url="https://example.invalid/v1",
                                      auth_handler=handler, refresh_credential=refresh))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(PROVIDER, None),
                                  hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None)))
    args = SimpleNamespace(provider=PROVIDER, no_browser=False)
    assert handler("add", args) is True
    rows = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER]
    spent = rows[0]["refresh_token"]
    idp.refresh_tokens.discard(spent)

    with pytest.raises(AuthError) as excinfo:
        refresh(SimpleNamespace(provider=PROVIDER, id=rows[0]["id"], refresh_token=spent,
                                access_token=rows[0]["access_token"], expires_at_ms=None))
    assert excinfo.value.code == "invalid_grant" and excinfo.value.relogin_required is True

    pool = load_pool(PROVIDER)
    result = pool.try_refresh_matching(credential_id=rows[0]["id"])
    assert result is None or result.last_status == STATUS_DEAD
    disk = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER][0]
    assert disk["last_status"] == STATUS_DEAD


def test_alias_login_stores_the_row_under_the_canonical_profile_name(idp, monkeypatch, tmp_path, request):
    """hermes auth add <alias> must write the pool under ProviderProfile.name, not the typed alias."""
    from agent.credential_pool import load_pool

    alias = "example-pkce-alias"
    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler = pkce.pkce_auth_handler(cfg)
    register_provider(ProviderProfile(
        name=PROVIDER, aliases=(alias,), auth_type="oauth_external",
        base_url="https://example.invalid/v1",
        auth_handler=handler, refresh_credential=pkce.pkce_refresh_credential(cfg)))
    request.addfinalizer(lambda: (
        providers._REGISTRY.pop(PROVIDER, None),
        providers._ALIASES.pop(alias, None),
        hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None),
        hermes_cli.auth.PROVIDER_REGISTRY.pop(alias, None)))
    args = SimpleNamespace(provider=alias, no_browser=False)
    assert handler("add", args) is True
    pool = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]
    assert PROVIDER in pool and alias not in pool
    assert handler("status", args) is True
    assert load_pool(PROVIDER).entries()[0].provider == PROVIDER
