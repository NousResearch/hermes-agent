"""The Nous plane credential resolves under its own profile's secret scope when the gateway multiplexes.

The Remote Config poller is a bare thread with no profile scope. Without a binding, the resolver's
read of ``HERMES_PORTAL_BASE_URL`` fails closed, the override counts as absent, the stored
non-production ``portal_base_url`` fails the host allowlist and the refresh token is sent to the
production Portal (``invalid_grant`` -> the login is quarantined).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from hermes_agent.agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
from plugins.config_backends.remote import credentials

STAGING = "https://portal.staging-nousresearch.com"
OTHER = "https://portal.other-nousresearch.com"


def _profile(home: Path, portal: str) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"HERMES_PORTAL_BASE_URL={portal}\n", encoding="utf-8")
    nous = {"access_token": "expired-access", "refresh_token": f"refresh-for-{portal}",
            "client_id": "hermes-cli-vps", "portal_base_url": portal,
            "expires_at": "2000-01-01T00:00:00+00:00"}
    (home / "auth.json").write_text(json.dumps(
        {"version": 1, "active_provider": "nous", "providers": {"nous": nous}}), encoding="utf-8")
    return home


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    home = _profile(tmp_path / "hermes", STAGING)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    return home


@pytest.fixture
def refreshes(monkeypatch):
    from hermes_agent.hermes_cli import auth, auth_nous

    seen: list[tuple[str, str]] = []

    def fake_refresh(*, client, portal_base_url, client_id, refresh_token):
        seen.append((portal_base_url, refresh_token))
        return {"access_token": f"access-from-{portal_base_url}", "refresh_token": refresh_token,
                "expires_in": 3600}

    monkeypatch.setattr(auth, "_RESOLVE_TOKEN_CACHE", {})
    monkeypatch.setattr(auth, "_refresh_access_token", fake_refresh)
    monkeypatch.setattr(auth_nous, "_refresh_access_token", fake_refresh)
    return seen


def _in_fresh_thread(fn):
    """Run ``fn`` on a bare thread, as the poller does: no ContextVar reaches it."""
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # surfaced to the test below
            out["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "token resolution hung"
    if "error" in out:
        raise out["error"]
    return out["value"]


def _no_unscoped_warning(caplog) -> None:
    text = caplog.text
    assert "unreadable" not in text, text
    assert "ignoring invalid portal_base_url" not in text, text


def test_poller_thread_refreshes_launch_profile_against_its_own_portal(launch_home, refreshes, caplog):
    set_multiplex_active(True)
    caplog.set_level(logging.WARNING)

    token = _in_fresh_thread(lambda: credentials.plane_token(launch_home))

    assert refreshes == [(STAGING, f"refresh-for-{STAGING}")]
    assert token == f"access-from-{STAGING}"
    _no_unscoped_warning(caplog)


def test_poller_thread_refreshes_secondary_profile_against_its_own_portal(launch_home, refreshes, caplog):
    secondary = _profile(launch_home / "profiles" / "work", OTHER)
    set_multiplex_active(True)
    caplog.set_level(logging.WARNING)

    token = _in_fresh_thread(lambda: credentials.plane_token(secondary))

    assert refreshes == [(OTHER, f"refresh-for-{OTHER}")]
    assert token == f"access-from-{OTHER}"
    _no_unscoped_warning(caplog)


def test_another_profiles_bound_scope_does_not_route_this_profiles_token(launch_home, refreshes, caplog):
    """A caller bound to the launch profile resolving the secondary's token routes on the
    secondary's override, never the bound one; the caller's scope is back afterwards."""
    from hermes_agent.agent.secret_scope import current_secret_scope_home

    secondary = _profile(launch_home / "profiles" / "work", OTHER)
    set_multiplex_active(True)
    caplog.set_level(logging.WARNING)

    bound = set_secret_scope({"HERMES_PORTAL_BASE_URL": STAGING}, profile_home=str(launch_home))
    try:
        credentials.plane_token(secondary)
        assert current_secret_scope_home() == str(launch_home)
    finally:
        reset_secret_scope(bound)

    assert refreshes == [(OTHER, f"refresh-for-{OTHER}")]
    _no_unscoped_warning(caplog)


def test_scope_is_reset_when_resolution_fails(launch_home, monkeypatch):
    from hermes_agent.agent.secret_scope import current_secret_scope
    from hermes_agent.hermes_cli import auth

    set_multiplex_active(True)

    def boom(**_kw):
        assert current_secret_scope() is not None  # bound during the resolution
        raise RuntimeError("network down")

    monkeypatch.setattr(auth, "resolve_nous_access_token", boom)

    def call():
        with pytest.raises(credentials.PlaneCredentialError):
            credentials.plane_token(launch_home)
        return current_secret_scope()

    assert _in_fresh_thread(call) is None


def test_home_and_scope_are_restored_when_the_scope_build_fails(launch_home, monkeypatch):
    from hermes_agent.agent.secret_scope import current_secret_scope_home
    from hermes_agent.hermes_constants import get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override
    from hermes_agent.tui_gateway import launch_profile_policy

    secondary = _profile(launch_home / "profiles" / "work", OTHER)
    set_multiplex_active(True)

    def boom(_home):
        raise RuntimeError("unreadable managed .env")

    monkeypatch.setattr(launch_profile_policy, "owner_secret_scope", boom)
    home_token = set_hermes_home_override(str(launch_home))
    scope_token = set_secret_scope({}, profile_home=str(launch_home))
    try:
        with pytest.raises(credentials.PlaneCredentialError):
            credentials.plane_token(secondary)
        assert get_hermes_home_override() == str(launch_home)
        assert current_secret_scope_home() == str(launch_home)
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


def test_launch_profile_keeps_its_env_only_override_frozen_at_activation(tmp_path, monkeypatch, refreshes):
    """A launch Portal override with no .env line (systemd / ``op run``) is the launch profile's
    own: it survives later live-env mutation because the launch scope reads the frozen env."""
    home = _profile(tmp_path / "hermes", STAGING)
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", STAGING)
    from hermes_agent.tui_gateway.launch_profile_policy import activate_multi_profile_hosting
    activate_multi_profile_hosting()  # freezes the launch env
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", OTHER)  # poisoned afterwards

    _in_fresh_thread(lambda: credentials.plane_token(home))

    assert refreshes == [(STAGING, f"refresh-for-{STAGING}")]


def test_secondary_never_borrows_the_launch_profiles_env_override(launch_home, monkeypatch, refreshes):
    """A secondary with no Portal override of its own routes on its stored (allowlisted) URL, never
    on the launch profile's process-env override."""
    secondary = _profile(launch_home / "profiles" / "work", "https://portal.nousresearch.com")
    (secondary / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", STAGING)
    set_multiplex_active(True)

    _in_fresh_thread(lambda: credentials.plane_token(secondary))

    assert refreshes == [("https://portal.nousresearch.com", "refresh-for-https://portal.nousresearch.com")]


def test_single_profile_binds_nothing(launch_home, refreshes, monkeypatch):
    """Not multiplexing: no scope is bound and the process env still routes (unchanged behaviour)."""
    from hermes_agent.agent.secret_scope import current_secret_scope
    from hermes_agent.hermes_cli import auth

    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", OTHER)
    real = auth.resolve_nous_access_token
    scopes = []

    def spy(**kw):
        scopes.append(current_secret_scope())
        return real(**kw)

    monkeypatch.setattr(auth, "resolve_nous_access_token", spy)
    credentials.plane_token(launch_home)
    assert scopes == [None]
    assert refreshes[0][0] == OTHER
