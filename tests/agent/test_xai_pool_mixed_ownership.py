"""PR #67304: a profile singleton must not own an independent borrowed pool grant."""

import json
from pathlib import Path

import pytest

from agent.credential_pool import load_pool
from hermes_cli import auth


def _store(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def mixed_grants(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profiles = [root / "profiles" / name for name in ("one", "two")]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for profile in profiles:
        profile.mkdir(parents=True)
        (profile / "auth.json").write_text(json.dumps({
            "version": 1, "active_provider": "openrouter",
            "providers": {"xai-oauth": {"tokens": {
                "access_token": f"{profile.name}-access-A", "refresh_token": f"{profile.name}-refresh-A",
            }}},
        }), encoding="utf-8")
    root_store = {
        "version": 1, "active_provider": "openai-codex",
        "providers": {"xai-oauth": {"tokens": {
            "access_token": "access-C", "refresh_token": "refresh-C",
        }}},
        "credential_pool": {"xai-oauth": [
            {"id": "borrowed-B", "source": "manual:device_code", "auth_type": "oauth", "priority": 0,
             "access_token": "access-B0", "refresh_token": "refresh-B0"},
            {"id": "independent-C", "source": "manual:device_code", "auth_type": "oauth", "priority": 1,
             "access_token": "access-C", "refresh_token": "refresh-C"},
        ]},
    }
    (root / "auth.json").write_text(json.dumps(root_store), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profiles[0]))
    return root, profiles


def test_borrowed_pool_rotations_lock_and_update_root_not_profile_singleton(mixed_grants, monkeypatch):
    root, profiles = mixed_grants
    root_path = root / "auth.json"
    root_before = _store(root_path)
    profile_before = [(profile / "auth.json").read_bytes() for profile in profiles]
    posted = []
    root_locked = []

    def rotate(access_token, refresh_token, **_kwargs):
        root_locked.append(getattr(auth._auth_lock_holder_for(root_path), "depth", 0) > 0)
        posted.append((access_token, refresh_token))
        n = len(posted)
        return {"access_token": f"access-B{n}", "refresh_token": f"refresh-B{n}",
                "last_refresh": f"2026-10-01T00:0{n}:00Z"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", rotate)
    for n, profile in enumerate(profiles, start=1):
        monkeypatch.setenv("HERMES_HOME", str(profile))
        pool = load_pool("xai-oauth")
        refreshed = pool.try_refresh_matching(credential_id="borrowed-B")
        assert refreshed is not None
        assert refreshed.access_token == f"access-B{n}"
        reloaded = load_pool("xai-oauth")
        assert next(row for row in reloaded.entries() if row.id == "borrowed-B").refresh_token == f"refresh-B{n}"
        assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == f"{profile.name}-access-A"
    assert posted == [("access-B0", "refresh-B0"), ("access-B1", "refresh-B1")]
    assert root_locked == [True, True]
    for profile, before in zip(profiles, profile_before):
        assert (profile / "auth.json").read_bytes() == before
    after = _store(root_path)
    assert after["providers"] == root_before["providers"]
    assert after["active_provider"] == root_before["active_provider"]
    independent = after["credential_pool"]["xai-oauth"][1]
    assert independent.items() >= root_before["credential_pool"]["xai-oauth"][1].items()
    assert after["credential_pool"]["xai-oauth"][0]["refresh_token"] == "refresh-B2"


def test_terminal_borrowed_grant_stays_quarantined_without_removing_independent_grants(mixed_grants, monkeypatch):
    root, profiles = mixed_grants
    root_path = root / "auth.json"
    before = [_store(profile / "auth.json") for profile in profiles]
    root_before = _store(root_path)
    pool = load_pool("xai-oauth")
    monkeypatch.setenv("HERMES_HOME", str(profiles[1]))
    waiting_pool = load_pool("xai-oauth")
    monkeypatch.setenv("HERMES_HOME", str(profiles[0]))
    posted = []

    def revoked(_access_token, refresh_token, **_kwargs):
        posted.append(refresh_token)
        raise auth.AuthError("revoked", provider="xai-oauth", code="xai_refresh_failed", relogin_required=True)

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", revoked)
    assert pool.try_refresh_matching(credential_id="borrowed-B") is None
    assert "one-access-A" in {row.access_token for row in pool.entries()}
    assert "access-C" in {row.access_token for row in pool.entries()}
    for profile, profile_before in zip(profiles, before):
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert "borrowed-B" not in {row.id for row in load_pool("xai-oauth").entries()}
        assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == f"{profile.name}-access-A"
        assert _store(profile / "auth.json") == profile_before
    # A pool loaded BEFORE quarantine must also adopt the removal, not POST a revoked grant.
    assert waiting_pool.try_refresh_matching(credential_id="borrowed-B") is None
    assert posted == ["refresh-B0"]
    after = _store(root_path)
    assert after["providers"] == root_before["providers"]
    assert after["active_provider"] == root_before["active_provider"]
    assert [row["id"] for row in after["credential_pool"]["xai-oauth"]] == ["independent-C"]
    assert after["credential_pool"]["xai-oauth"][0]["refresh_token"] == "refresh-C"
