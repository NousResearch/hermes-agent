"""Regression tests: xAI OAuth root credential_pool write-back on rotation.

Bug (split-brain observed on a two-profile host, 2026-09-15): when the xAI grant lives
ONLY in the global root's ``credential_pool`` (device-code pool entry, no
``providers.xai-oauth`` block anywhere), a profile resolving it via fallback and
refreshing wrote the rotated chain into a PROFILE providers-shadow via
``_save_xai_oauth_tokens``. That shadow then won every later read — root's pool entry
kept the consumed single-use refresh token and a stale access token, so the sibling
gateway (resolving from root) died on 403 ``bad-credentials`` once the stale access
token expired, and root's next refresh attempt failed with a reused refresh token.

The fix: when the profile has no own grant and root's pool carries a usable one, the
rotation writes back into the ROOT pool entry (mirroring the #74339 pool-refresh fix,
``_sync_device_code_entry_to_auth_store``) instead of creating a shadow.

These tests drive the real ``_save_xai_oauth_tokens`` against real on-disk stores
(profile + root under ``tmp_path``).
"""

import json

import pytest

import hermes_cli.auth_xai as AX
from hermes_cli import auth as A


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Profile store WITHOUT own xai-oauth grant + root store whose grant lives only in credential_pool."""
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    # Keep the pytest seat belt away from these tmp stores.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def test_rotation_writes_back_to_root_pool_entry(profile_and_root):
    """A profile refreshing a root-pool-resolved grant must update the ROOT pool entry,
    not create a profile providers-shadow."""
    profile_path, root_path = profile_and_root

    _write_store(
        profile_path,
        {"version": 1, "providers": {}, "credential_pool": {}},
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "root-entry",
                        "label": "main",
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                        "request_count": 7,
                    }
                ]
            },
        },
    )

    AX._save_xai_oauth_tokens(
        {"access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer"},
        discovery={"token_endpoint": "https://auth.x.ai/oauth2/token"},
        set_active=False,
    )

    root = _read_store(root_path)
    entry = root["credential_pool"]["xai-oauth"][0]
    assert entry["access_token"] == "new-access"
    assert entry["refresh_token"] == "new-refresh"
    # Non-token fields survive the rotation.
    assert entry["id"] == "root-entry"
    assert entry["request_count"] == 7

    profile = _read_store(profile_path)
    assert "xai-oauth" not in (profile.get("providers") or {}), (
        "rotation must not create a profile providers-shadow that shadows root forever"
    )


def test_rotation_still_writes_profile_when_profile_owns_grant(profile_and_root):
    """Sanity: when the PROFILE owns the grant (own providers block), rotation stays local."""
    profile_path, root_path = profile_and_root

    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {"tokens": {"access_token": "p-old-a", "refresh_token": "p-old-r"}}
            },
            "credential_pool": {},
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "xai-oauth": [
                    {"id": "other-account", "access_token": "other-a", "refresh_token": "other-r"}
                ]
            },
        },
    )

    AX._save_xai_oauth_tokens(
        {"access_token": "p-new-a", "refresh_token": "p-new-r"}, set_active=False
    )

    profile = _read_store(profile_path)
    tokens = profile["providers"]["xai-oauth"]["tokens"]
    assert tokens["access_token"] == "p-new-a"
    root = _read_store(root_path)
    assert root["credential_pool"]["xai-oauth"][0]["access_token"] == "other-a", (
        "an independent root pool entry (different account) must not be overwritten"
    )


def test_rotation_without_any_grant_writes_profile_local(profile_and_root):
    """No grant anywhere: keep the classic behavior (profile-local write)."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {}})
    _write_store(root_path, {"version": 1, "providers": {}, "credential_pool": {}})

    AX._save_xai_oauth_tokens(
        {"access_token": "lone-a", "refresh_token": "lone-r"}, set_active=False
    )

    profile = _read_store(profile_path)
    assert profile["providers"]["xai-oauth"]["tokens"]["access_token"] == "lone-a"
