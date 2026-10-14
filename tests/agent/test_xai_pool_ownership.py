"""xAI rotating grants keep one authority across sibling profiles (PR #67304)."""

import json
from pathlib import Path

import pytest

from agent.credential_pool import load_pool
from hermes_cli import auth


@pytest.mark.parametrize("storage", ["singleton", "pool", "both"])
@pytest.mark.parametrize("profile_state", [None, {"auth_mode": "oauth_device_code"}])
def test_two_pool_rotations_reach_sibling_without_forking(tmp_path, monkeypatch, profile_state, storage):
    root = tmp_path / ".hermes"
    profiles = [root / "profiles" / name for name in ("one", "two")]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for profile in profiles:
        profile.mkdir(parents=True)
        (profile / "auth.json").write_text(json.dumps({
            "version": 1, "active_provider": "openrouter",
            "providers": {"xai-oauth": profile_state} if profile_state else {},
        }), encoding="utf-8")
    pair = {"access_token": "access-0", "refresh_token": "refresh-0"}
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"xai-oauth": {
            "tokens": pair if storage != "pool" else {}, "last_auth_error": {"code": "old-error"},
        }},
        "credential_pool": {"xai-oauth": [{
            "id": "root-grant", "source": "device_code", "auth_type": "oauth", **pair,
        }] if storage != "singleton" else []},
    }), encoding="utf-8")
    posted = []

    def rotate(_access_token, refresh_token, **_kwargs):
        posted.append(refresh_token)
        n = len(posted)
        return {"access_token": f"access-{n}", "refresh_token": f"refresh-{n}",
                "last_refresh": f"2026-10-01T00:0{n}:00Z"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", rotate)
    monkeypatch.setenv("HERMES_HOME", str(profiles[0]))
    for generation in (1, 2):
        pool = load_pool("xai-oauth")
        assert pool.select() is not None
        assert pool.try_refresh_current().access_token == f"access-{generation}"
    assert posted == ["refresh-0", "refresh-1"]
    root_store = json.loads((root / "auth.json").read_text(encoding="utf-8"))
    assert root_store["active_provider"] == "openai-codex"
    assert root_store["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "refresh-2"
    assert "last_auth_error" not in root_store["providers"]["xai-oauth"]
    for profile in profiles:
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert load_pool("xai-oauth").select().access_token == "access-2"
        assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "access-2"
        store = json.loads((profile / "auth.json").read_text(encoding="utf-8"))
        assert store["active_provider"] == "openrouter"
        assert store["providers"].get("xai-oauth") == profile_state
        assert not store.get("credential_pool", {}).get("xai-oauth")
