"""xAI OAuth has one canonical owner across all Hermes profiles."""
from __future__ import annotations

import base64
import json
import stat
import time

from hermes_cli import auth
from agent import credential_pool as cp


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _state(access, refresh):
    return {
        "auth_mode": "oauth_device_code",
        "tokens": {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
        },
        "last_refresh": "2026-07-31T07:01:13.681197Z",
    }


def _jwt_with_exp(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_profile_xai_state_never_shadows_canonical_root(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    _write(
        profile_path,
        {
            "providers": {"xai-oauth": _state("stale-access", "stale-refresh")},
            "credential_pool": {
                "xai-oauth": [{"access_token": "pool-stale", "refresh_token": "pool-stale-r"}]
            },
        },
    )
    _write(root_path, {"providers": {"xai-oauth": _state("root-access", "root-refresh")}})
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "seatbelt-home"))

    resolved = auth._read_xai_oauth_tokens()

    assert resolved["tokens"]["access_token"] == "root-access"
    assert resolved["tokens"]["refresh_token"] == "root-refresh"


def test_xai_save_updates_only_root_and_derives_pool_atomically(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    profile = {"providers": {"anthropic": {"keep": True}}, "credential_pool": {}}
    _write(profile_path, profile)
    _write(
        root_path,
        {
            "providers": {
                "xai-oauth": {
                    **_state("old-access", "old-refresh"),
                    "last_auth_error": {"code": "xai_refresh_failed"},
                }
            },
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "canonical",
                        "source": "device_code",
                        "access_token": "split-access",
                        "refresh_token": "split-refresh",
                        "last_status": "error",
                        "last_error_code": "invalid_grant",
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "seatbelt-home"))

    auth._save_xai_oauth_tokens(
        {"access_token": "new-access", "refresh_token": "new-refresh"},
        last_refresh="2026-08-01T00:00:00Z",
    )

    assert _read(profile_path) == profile
    root = _read(root_path)
    provider = root["providers"]["xai-oauth"]
    pool = root["credential_pool"]["xai-oauth"]
    assert len(pool) == 1
    assert pool[0]["access_token"] == provider["tokens"]["access_token"]
    assert pool[0]["refresh_token"] == provider["tokens"]["refresh_token"]
    assert "last_auth_error" not in provider
    assert "last_error_code" not in pool[0]
    assert stat.S_IMODE(root_path.stat().st_mode) == 0o600


def test_xai_pool_write_cannot_override_canonical_provider_tokens(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    _write(profile_path, {"providers": {}, "credential_pool": {}})
    _write(root_path, {"providers": {"xai-oauth": _state("root-access", "root-refresh")}})
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "seatbelt-home"))

    auth.write_credential_pool(
        "xai-oauth",
        [
            {
                "id": "stale",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "stale-access",
                "refresh_token": "stale-refresh",
            }
        ],
    )

    root = _read(root_path)
    entry = root["credential_pool"]["xai-oauth"][0]
    assert entry["access_token"] == "root-access"
    assert entry["refresh_token"] == "root-refresh"
    assert "xai-oauth" not in _read(profile_path)["credential_pool"]


def test_pool_refresh_holds_canonical_lock_and_commits_pair_atomically(
    tmp_path, monkeypatch
):
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    _write(profile_path, {"providers": {}, "credential_pool": {}})
    _write(
        root_path,
        {
            "providers": {"xai-oauth": _state("old-access", "old-refresh")},
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "canonical",
                        "label": "device_code",
                        "priority": 0,
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "seatbelt-home"))
    observed = {"root_lock_held": False}

    def fake_refresh(access_token, refresh_token, **_kwargs):
        holder = auth._auth_lock_holder_for(root_path)
        observed["root_lock_held"] = getattr(holder, "depth", 0) > 0
        assert access_token == "old-access"
        assert refresh_token == "old-refresh"
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2026-08-01T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_refresh)
    pool = cp.load_pool("xai-oauth")
    entry = pool.entries()[0]

    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert observed["root_lock_held"] is True
    root = _read(root_path)
    provider_tokens = root["providers"]["xai-oauth"]["tokens"]
    pool_tokens = root["credential_pool"]["xai-oauth"][0]
    assert provider_tokens["access_token"] == "rotated-access"
    assert provider_tokens["refresh_token"] == "rotated-refresh"
    assert pool_tokens["access_token"] == provider_tokens["access_token"]
    assert pool_tokens["refresh_token"] == provider_tokens["refresh_token"]
    assert "xai-oauth" not in _read(profile_path)["providers"]


def test_pool_refresh_adopts_fresh_auth_store_token_without_post(
    tmp_path, monkeypatch
):
    """A lock waiter must reuse the winner's fresh pair, not spend it again."""
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    fresh_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _write(profile_path, {"providers": {}, "credential_pool": {}})
    _write(
        root_path,
        {"providers": {"xai-oauth": _state(fresh_access, "fresh-refresh")}},
    )
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "seatbelt-home"))

    stale = cp.PooledCredential(
        provider="xai-oauth",
        id="canonical",
        label="device_code",
        priority=0,
        source="device_code",
        auth_type=cp.AUTH_TYPE_OAUTH,
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    pool = cp.CredentialPool("xai-oauth", [stale])
    refresh_calls = {"count": 0}

    def unexpected_refresh(*_args, **_kwargs):
        refresh_calls["count"] += 1
        raise AssertionError("fresh auth-store token must skip refresh POST")

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", unexpected_refresh)

    refreshed = pool._refresh_entry(stale, force=False)

    assert refreshed is not None
    assert refreshed.access_token == fresh_access
    assert refreshed.refresh_token == "fresh-refresh"
    assert refresh_calls["count"] == 0
