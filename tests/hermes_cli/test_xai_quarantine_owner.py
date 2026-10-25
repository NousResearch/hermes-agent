"""Terminal failures must not resurrect a root grant from its pooled aliases."""

import json
from pathlib import Path

import pytest

from agent.credential_pool import load_pool
from hermes_cli import auth


@pytest.mark.parametrize("via_pool", [False, True], ids=["singleton", "pool"])
def test_terminal_root_grant_cannot_be_resolved_from_pool(tmp_path, monkeypatch, via_pool):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    profile_store = {"version": 1, "active_provider": "openrouter", "providers": {}}
    (profile / "auth.json").write_text(json.dumps(profile_store), encoding="utf-8")
    dead = {"access_token": "dead-access", "refresh_token": "dead-refresh"}
    healthy = {"id": "independent", "source": "manual:device_code", "auth_type": "oauth",
               "access_token": "healthy-access", "refresh_token": "healthy-refresh", "priority": 3}
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"xai-oauth": {"tokens": dead,
            "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"}}},
        "credential_pool": {"xai-oauth": [
            {"id": "singleton", "source": "device_code", "auth_type": "oauth", "priority": 0, **dead},
            {"id": "alias", "source": "manual:device_code", "auth_type": "oauth", "priority": 1, **dead},
            {"id": "access-alias", "source": "manual", "auth_type": "api_key", "priority": 2,
             "access_token": "dead-access"},
            healthy,
        ]},
    }), encoding="utf-8")
    posted = []

    def revoked(_access, refresh, **_kwargs):
        posted.append(refresh)
        raise auth.AuthError("revoked", provider="xai-oauth", code="xai_refresh_failed", relogin_required=True)

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", revoked)
    if via_pool:
        pool = load_pool("xai-oauth")
        assert pool.select().access_token == "dead-access"
        assert pool.try_refresh_current() is None
        assert pool.select().access_token == "healthy-access"
    else:
        with pytest.raises(auth.AuthError, match="revoked"):
            auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert posted == ["dead-refresh"]
    root_store = json.loads((root / "auth.json").read_text(encoding="utf-8"))
    assert root_store["active_provider"] == "openai-codex"
    assert not root_store["providers"]["xai-oauth"]["tokens"].get("refresh_token")
    assert json.loads((profile / "auth.json").read_text(encoding="utf-8")) == profile_store
    assert [row["id"] for row in auth.read_credential_pool("xai-oauth")] == ["independent"]
    assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "healthy-access"
    assert load_pool("xai-oauth").select().access_token == "healthy-access"
