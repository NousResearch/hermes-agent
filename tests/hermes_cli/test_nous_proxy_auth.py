"""Nous proxy authentication regressions."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter


def test_nous_proxy_uses_global_auth_when_profile_has_no_nous_state(tmp_path, monkeypatch):
    """Regression for #98098: proxy readiness follows profile/global auth fallback."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_home = tmp_path / ".hermes"
    profile_home = global_home / "profiles" / "proxy-worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    (profile_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    (global_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "nous": {"access_token": "global-access", "refresh_token": "global-refresh"},
        },
    }))

    assert NousPortalAdapter().is_authenticated()


def test_nous_proxy_prefers_profile_auth_over_global_fallback(tmp_path, monkeypatch):
    """A profile's Nous state continues to shadow the global fallback."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_home = tmp_path / ".hermes"
    profile_home = global_home / "profiles" / "proxy-worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    (profile_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "nous": {"access_token": "profile-access", "refresh_token": "profile-refresh"},
        },
    }))
    (global_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "nous": {"access_token": "global-access", "refresh_token": "global-refresh"},
        },
    }))

    assert NousPortalAdapter()._read_state()["access_token"] == "profile-access"
