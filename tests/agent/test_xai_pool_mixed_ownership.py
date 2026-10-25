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


def test_device_code_seeding_keeps_profile_and_borrowed_root_grants_separate(mixed_grants, monkeypatch):
    root, profiles = mixed_grants
    root_path = root / "auth.json"
    store = _store(root_path)
    store["providers"] = {}
    borrowed = store["credential_pool"]["xai-oauth"][0]
    borrowed["source"] = "device_code"
    store["credential_pool"]["xai-oauth"] = [borrowed]
    root_path.write_text(json.dumps(store), encoding="utf-8")
    profile_path = profiles[0] / "auth.json"
    posted = []

    def rotate(access_token, refresh_token, **_kwargs):
        posted.append((access_token, refresh_token))
        return {"access_token": access_token + "-rotated", "refresh_token": refresh_token + "-rotated",
                "last_refresh": "2026-10-01T00:01:00Z"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", rotate)
    for suffix in ("", "-rotated"):
        pool = load_pool("xai-oauth")
        by_id = {row.id: row for row in pool.entries()}
        assert by_id["borrowed-B"].refresh_token == "refresh-B0" + suffix
        local = next(row for row in pool.entries() if row.id != "borrowed-B")
        assert local.refresh_token == "one-refresh-A" + suffix
        assert pool.select().id == (local.id if suffix else "borrowed-B")
        assert _store(root_path)["credential_pool"]["xai-oauth"][0]["refresh_token"] == "refresh-B0" + suffix
        if suffix:
            break
        assert pool.try_refresh_current().refresh_token == "refresh-B0-rotated"
        assert _store(profile_path)["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "one-refresh-A"
        root_after_rotation = root_path.read_bytes()
        assert pool.try_refresh_matching(credential_id=local.id).refresh_token == "one-refresh-A-rotated"
        assert root_path.read_bytes() == root_after_rotation
        # A status flush must not materialize borrowed rows in the profile or swap their tokens.
        assert pool.mark_exhausted_and_rotate(credential_id="borrowed-B", status_code=429).id == local.id
    assert posted == [("access-B0", "refresh-B0"), ("one-access-A", "one-refresh-A")]
    assert _store(profile_path)["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "one-refresh-A-rotated"
    assert not _store(profile_path).get("credential_pool", {}).get("xai-oauth")
    assert _store(profile_path)["active_provider"] == "openrouter"
    assert _store(root_path)["active_provider"] == "openai-codex"
    monkeypatch.setenv("HERMES_HOME", str(profiles[1]))
    sibling = load_pool("xai-oauth")
    assert next(row for row in sibling.entries() if row.id == "borrowed-B").refresh_token == "refresh-B0-rotated"
    assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "two-access-A"


def test_same_grant_profile_singleton_and_root_pool_keep_one_owner(mixed_grants, monkeypatch):
    root, profiles = mixed_grants
    root_path = root / "auth.json"
    profile_path = profiles[0] / "auth.json"
    store = _store(root_path)
    store["providers"] = {}
    borrowed = store["credential_pool"]["xai-oauth"][0]
    borrowed["source"] = "device_code"
    store["credential_pool"]["xai-oauth"] = [borrowed]
    root_path.write_text(json.dumps(store), encoding="utf-8")
    profile = _store(profile_path)
    profile["providers"]["xai-oauth"]["tokens"] = {
        "access_token": borrowed["access_token"], "refresh_token": borrowed["refresh_token"],
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    posted = []

    def rotate(access_token, refresh_token, **_kwargs):
        posted.append(refresh_token)
        return {"access_token": access_token + "-rotated", "refresh_token": refresh_token + "-rotated"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", rotate)
    for suffix in ("", "-rotated"):
        pool = load_pool("xai-oauth")
        assert [(row.id, row.refresh_token) for row in pool.entries()] == [("borrowed-B", "refresh-B0" + suffix)]
        assert not _store(profile_path)["providers"].get("xai-oauth")
        assert pool.try_refresh_matching(credential_id="borrowed-B").refresh_token == "refresh-B0" + suffix + "-rotated"
    assert posted == ["refresh-B0", "refresh-B0-rotated"]
    assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "access-B0-rotated-rotated"
    assert _store(profile_path)["active_provider"] == "openrouter"


def test_same_owner_singleton_aliases_still_deduplicate(mixed_grants, monkeypatch):
    _, profiles = mixed_grants
    home = profiles[0]
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "auth.json"
    store = _store(path)
    tokens = store["providers"]["xai-oauth"]["tokens"]
    store["credential_pool"] = {"xai-oauth": [
        {"id": row_id, "source": "device_code", "auth_type": "oauth", **tokens}
        for row_id in ("original", "alias")
    ]}
    store["credential_pool"]["xai-oauth"].append({
        "id": "independent", "source": "manual:device_code", "auth_type": "oauth",
        "access_token": "other-access", "refresh_token": "other-refresh",
    })
    path.write_text(json.dumps(store), encoding="utf-8")
    posted = []

    def rotate(access_token, refresh_token, **_kwargs):
        posted.append(refresh_token)
        return {"access_token": access_token + "-rotated", "refresh_token": refresh_token + "-rotated"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", rotate)
    pool = load_pool("xai-oauth")
    assert {row.id for row in pool.entries()} == {"original", "independent"}
    assert pool.try_refresh_matching(credential_id="original").refresh_token == tokens["refresh_token"] + "-rotated"
    reloaded = {row.id: row for row in load_pool("xai-oauth").entries()}
    assert set(reloaded) == {"original", "independent"}
    assert reloaded["original"].refresh_token == tokens["refresh_token"] + "-rotated"
    assert reloaded["independent"].refresh_token == "other-refresh"
    assert posted == [tokens["refresh_token"]]
    assert {row["id"] for row in _store(path)["credential_pool"]["xai-oauth"]} == {"original", "independent"}


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
