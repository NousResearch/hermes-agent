"""Canonical shared xAI OAuth store — issue #65394.

Unlike the Nous shared store (profile is source of truth + best-effort mirror),
the xAI shared store itself is authoritative: one grant family, one lock, one
serialized refresher, no per-profile forking of the rotating refresh token.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli import auth
from hermes_cli.auth import AuthError


def _jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"{header}.{payload}.sig"


@pytest.fixture
def shared_env(tmp_path, monkeypatch):
    """Enable shared xAI mode with an isolated shared dir + profile auth."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    # Keep seat belts off real home.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    # Ensure gate reads as enabled.
    assert auth._xai_shared_auth_enabled() is True
    return {
        "shared_dir": shared_dir,
        "hermes_home": hermes_home,
        "store": shared_dir / "xai_oauth.json",
        "profile_auth": hermes_home / "auth.json",
    }


def _write_shared(env, *, access="at-1", refresh="rt-1", generation=1, **extra):
    payload = {
        "_schema": 1,
        "generation": generation,
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "auth_mode": "oauth_device_code",
        "last_refresh": "2026-07-01T00:00:00Z",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
        **extra,
    }
    env["store"].write_text(json.dumps(payload), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Seat belt / gate / path
# ---------------------------------------------------------------------------


def test_shared_mode_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    assert auth._xai_shared_auth_enabled() is False


def test_shared_mode_via_providers_list(monkeypatch):
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.setenv("HERMES_SHARED_AUTH_PROVIDERS", "nous,xai-oauth")
    assert auth._xai_shared_auth_enabled() is True


def test_pytest_seat_belt_refuses_real_shared_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    # Point the shared dir at the platform-native real path so the seat belt
    # has something dangerous to refuse (conftest normally isolates HERMES_HOME).
    from hermes_constants import _get_platform_default_hermes_home

    real_shared = _get_platform_default_hermes_home() / "shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(real_shared))
    with pytest.raises(RuntimeError, match="Refusing to touch real user shared xAI"):
        auth._xai_shared_store_path()


def test_write_shared_creates_0600_file(shared_env):
    written = auth._write_shared_xai_state(
        {
            "access_token": "at",
            "refresh_token": "rt",
            "auth_mode": "oauth_device_code",
        }
    )
    path = shared_env["store"]
    assert path.is_file()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    assert written["generation"] == 1
    assert written["refresh_token"] == "rt"


def test_write_shared_bumps_generation(shared_env):
    auth._write_shared_xai_state(
        {"access_token": "a1", "refresh_token": "r1"}
    )
    second = auth._write_shared_xai_state(
        {"access_token": "a2", "refresh_token": "r2"}
    )
    assert second["generation"] == 2


def test_persist_failure_is_loud(shared_env, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "open", boom)
    with pytest.raises(AuthError) as exc:
        auth._write_shared_xai_state(
            {"access_token": "a", "refresh_token": "r"}
        )
    assert exc.value.code == "xai_shared_persist_failed"


# ---------------------------------------------------------------------------
# Read / save / no profile fork
# ---------------------------------------------------------------------------


def test_read_prefers_shared_not_profile(shared_env):
    _write_shared(shared_env, access="shared-at", refresh="shared-rt")
    # Poison the profile with a different RT — must be ignored.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "profile-at",
                            "refresh_token": "profile-rt",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    data = auth._read_xai_oauth_tokens(_lock=False)
    assert data["tokens"]["access_token"] == "shared-at"
    assert data["tokens"]["refresh_token"] == "shared-rt"
    assert data["auth_store"] == "shared"


def test_save_writes_shared_and_strips_profile_tokens(shared_env):
    auth._save_xai_oauth_tokens(
        {"access_token": "new-at", "refresh_token": "new-rt", "token_type": "Bearer"},
        discovery={"token_endpoint": "https://auth.x.ai/oauth/token"},
    )
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "new-rt"
    assert shared["access_token"] == "new-at"
    assert shared["generation"] >= 1

    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    state = profile["providers"]["xai-oauth"]
    assert state.get("source") == auth.XAI_SHARED_SOURCE
    assert "tokens" not in state
    assert state.get("refresh_token") is None


def test_write_through_disabled_when_shared_active(shared_env, monkeypatch):
    root = shared_env["hermes_home"].parent / "root" / "auth.json"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(json.dumps({"version": 1, "providers": {}}), encoding="utf-8")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root)
    # Should no-op entirely.
    auth._write_through_xai_oauth_to_global_root(
        {"tokens": {"access_token": "a", "refresh_token": "r"}}
    )
    root_store = json.loads(root.read_text(encoding="utf-8"))
    assert "xai-oauth" not in root_store.get("providers", {})


# ---------------------------------------------------------------------------
# Generation compare / concurrent adopt
# ---------------------------------------------------------------------------


def test_force_refresh_adopts_winner_without_second_post(shared_env, monkeypatch):
    _write_shared(
        shared_env,
        access=_jwt(int(time.time()) + 30),
        refresh="rt-stale",
        generation=3,
    )
    posts = {"n": 0}

    def fake_pure(access, refresh, **kwargs):
        posts["n"] += 1
        return {
            "access_token": "should-not-run",
            "refresh_token": "should-not-run",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)

    # Simulate: caller still holds the rejected old access token, but another
    # process already rotated the shared store to a new generation.
    _write_shared(
        shared_env,
        access="winner-at",
        refresh="winner-rt",
        generation=4,
    )
    creds = auth.resolve_xai_oauth_runtime_credentials(
        force_refresh=True,
        rejected_access_token=_jwt(int(time.time()) + 30),
        expected_generation=3,
    )
    assert posts["n"] == 0
    assert creds["api_key"] == "winner-at"
    assert creds["generation"] == 4


def test_refresh_posts_once_and_persists(shared_env, monkeypatch):
    old_at = _jwt(int(time.time()) - 10)  # expired
    _write_shared(shared_env, access=old_at, refresh="rt-1", generation=1)
    posts = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T01:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    monkeypatch.setattr(
        auth,
        "_xai_oauth_discovery",
        lambda *_a, **_k: {"token_endpoint": "https://auth.x.ai/oauth/token"},
    )

    creds = auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=True)
    assert posts == ["rt-1"]
    assert creds["api_key"] == "new-at"
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "new-rt"
    assert shared["generation"] == 2


def test_concurrent_waiters_second_adopts(shared_env, monkeypatch):
    old_at = _jwt(int(time.time()) - 5)
    _write_shared(shared_env, access=old_at, refresh="rt-only-once", generation=1)
    posts = []
    barrier = threading.Barrier(2)
    hold = threading.Event()

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        # Hold the lock (we're inside pure only after lock acquired by resolve)
        # Simulate slow network so the second waiter queues on the flock.
        hold.wait(timeout=2.0)
        return {
            "access_token": "rotated-at",
            "refresh_token": "rotated-rt",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T02:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    monkeypatch.setattr(
        auth,
        "_xai_oauth_discovery",
        lambda *_a, **_k: {"token_endpoint": "https://auth.x.ai/oauth/token"},
    )

    results = [None, None]
    errors = [None, None]

    def worker(idx):
        try:
            barrier.wait(timeout=5)
            if idx == 0:
                # First through does the refresh.
                results[idx] = auth.resolve_xai_oauth_runtime_credentials(
                    force_refresh=True,
                    rejected_access_token=old_at,
                    expected_generation=1,
                )
                hold.set()
            else:
                # Give winner a head start to acquire the lock.
                time.sleep(0.05)
                results[idx] = auth.resolve_xai_oauth_runtime_credentials(
                    force_refresh=True,
                    rejected_access_token=old_at,
                    expected_generation=1,
                )
        except Exception as exc:  # pragma: no cover
            errors[idx] = exc
            hold.set()

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)
    assert errors == [None, None]
    # Exactly one POST of the single-use RT.
    assert posts == ["rt-only-once"]
    assert results[0]["api_key"] == "rotated-at"
    assert results[1]["api_key"] == "rotated-at"


# ---------------------------------------------------------------------------
# Quarantine compare-and-clear
# ---------------------------------------------------------------------------


def test_quarantine_skips_when_generation_changed(shared_env):
    _write_shared(shared_env, access="a", refresh="rt-old", generation=5)
    # Simulate loser trying to clear with stale RT while winner already rotated.
    _write_shared(shared_env, access="a2", refresh="rt-new", generation=6)
    cleared = auth._clear_shared_xai_state(
        "test",
        terminal_error={"code": "xai_refresh_failed", "message": "nope"},
        only_if_refresh_token="rt-old",
        only_if_generation=5,
    )
    assert cleared is False
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "rt-new"
    assert shared["generation"] == 6


def test_quarantine_clears_when_still_canonical(shared_env):
    _write_shared(shared_env, access="a", refresh="rt-dead", generation=2)
    cleared = auth._clear_shared_xai_state(
        "test",
        terminal_error={
            "provider": "xai-oauth",
            "code": "xai_refresh_failed",
            "message": "invalid_grant",
            "relogin_required": True,
        },
        only_if_refresh_token="rt-dead",
        only_if_generation=2,
    )
    assert cleared is True
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert not shared.get("refresh_token")
    assert shared.get("last_auth_error", {}).get("code") == "xai_refresh_failed"


# ---------------------------------------------------------------------------
# Status / profile disable / migration
# ---------------------------------------------------------------------------


def test_status_points_at_shared_path(shared_env):
    _write_shared(
        shared_env,
        access=_jwt(int(time.time()) + 3600),
        refresh="rt",
        generation=1,
    )
    status = auth.get_xai_oauth_auth_status()
    assert status["logged_in"] is True
    assert status["shared_mode"] is True
    assert str(shared_env["store"]) in status["auth_store"]
    assert status["source"] == auth.XAI_SHARED_SOURCE
    # Never leak full RT in status.
    assert "refresh_token" not in status


def test_profile_disable_blocks_resolve(shared_env):
    _write_shared(shared_env, access="a", refresh="r")
    auth.disable_profile_xai_shared_auth()
    with pytest.raises(AuthError) as exc:
        auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.code == "xai_shared_profile_disabled"
    # Canonical grant remains.
    assert shared_env["store"].is_file()


def test_migrate_shared_strips_legacy(shared_env, monkeypatch):
    # Seed legacy profile tokens.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "legacy-at",
                            "refresh_token": "legacy-rt",
                        },
                        "last_refresh": "2026-06-01T00:00:00Z",
                        "auth_mode": "oauth_device_code",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    written = auth.migrate_xai_oauth_to_shared_store(source="profile", strip_legacy=True)
    assert written["refresh_token"] == "legacy-rt"
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "legacy-rt"
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    state = profile["providers"]["xai-oauth"]
    assert "tokens" not in state
    assert state.get("source") == auth.XAI_SHARED_SOURCE


def test_merge_shared_updates_local_dict(shared_env):
    _write_shared(shared_env, access="sa", refresh="sr", generation=9)
    local = {"access_token": "la", "refresh_token": "lr", "generation": 1}
    assert auth._merge_shared_xai_state(local) is True
    assert local["refresh_token"] == "sr"
    assert local["generation"] == 9


# ---------------------------------------------------------------------------
# A1/A3/A4/A5 — multi-profile legacy pool strip + sole ownership
# ---------------------------------------------------------------------------


def _seed_legacy_pool_auth(path: Path, *, access: str, refresh: str, manual: bool = False):
    source = "manual:device_code" if manual else "device_code"
    payload = {
        "version": 1,
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": access,
                    "refresh_token": refresh,
                },
                "auth_mode": "oauth_device_code",
            }
        },
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": f"{source}-{refresh[:8]}",
                    "source": source,
                    "auth_type": "oauth",
                    "access_token": access,
                    "refresh_token": refresh,
                    "priority": 0,
                }
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strip_covers_all_profiles_and_manuals(shared_env, tmp_path, monkeypatch):
    """A1/A4: strip removes RTs from EVERY profile + root + manual pool rows."""
    # Make default hermes root == tmp home so profile enumeration is in-sandbox.
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()

    # Active profile (HERMES_HOME)
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="p-at", refresh="p-rt", manual=False
    )
    # Second named profile with a manual RT
    other = profiles_root / "coder" / "auth.json"
    _seed_legacy_pool_auth(other, access="c-at", refresh="c-rt", manual=True)
    # Root auth.json with device_code RT
    root_auth = hermes_root / "auth.json"
    _seed_legacy_pool_auth(root_auth, access="r-at", refresh="r-rt", manual=False)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    audit = auth._strip_legacy_xai_oauth_secrets(include_global_root=True, fail_loud=True)
    assert audit  # something was cleaned

    for path in (shared_env["profile_auth"], other, root_auth):
        store = json.loads(path.read_text(encoding="utf-8"))
        assert not auth._auth_store_holds_durable_xai_refresh_token(store), path
        pool = store.get("credential_pool", {}).get("xai-oauth", [])
        for entry in pool:
            assert not entry.get("refresh_token")
            assert not str(entry.get("source") or "").startswith("manual")


def test_migrate_with_legacy_pools_leaves_no_fork(shared_env, tmp_path, monkeypatch):
    """Pre-populated device_code + same-RT pool fork is gone after migrate.

    R2: a *different* live RT (true multi-grant) is refused as ambiguous —
    this test uses a same-identity pool fork so promote is well-defined.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="legacy-at", refresh="legacy-rt"
    )
    # Same-identity pool fork (duplicate RT) — still a sole live grant.
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    store["credential_pool"]["xai-oauth"].append(
        {
            "id": "manual-fork",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "legacy-at",
            "refresh_token": "legacy-rt",
            "priority": 1,
        }
    )
    shared_env["profile_auth"].write_text(json.dumps(store), encoding="utf-8")

    written = auth.migrate_xai_oauth_to_shared_store(source="profile", strip_legacy=True)
    assert written["refresh_token"] == "legacy-rt"

    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)
    pool = profile.get("credential_pool", {}).get("xai-oauth", [])
    assert all(not e.get("refresh_token") for e in pool)
    assert all(not str(e.get("source") or "").startswith("manual") for e in pool)


def test_strip_fails_loud_when_residual_rt_remains(shared_env, monkeypatch):
    """A1: migration/login must fail if a durable RT cannot be removed."""
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="a", refresh="rt-stuck"
    )

    real_save = auth._save_auth_store

    def save_but_restore_rt(store, target_path=None):
        # Pretend write succeeded but leave a residual RT (poisoned write).
        path = real_save(store, target_path=target_path)
        poisoned = json.loads(path.read_text(encoding="utf-8"))
        poisoned.setdefault("providers", {})["xai-oauth"] = {
            "tokens": {"access_token": "a", "refresh_token": "rt-stuck"}
        }
        path.write_text(json.dumps(poisoned), encoding="utf-8")
        return path

    monkeypatch.setattr(auth, "_save_auth_store", save_but_restore_rt)
    with pytest.raises(AuthError) as exc:
        auth._strip_legacy_xai_oauth_secrets(include_global_root=False, fail_loud=True)
    assert exc.value.code == "xai_shared_strip_incomplete"


def test_logout_preserves_disable_marker(shared_env):
    """B1: hermes logout --provider xai-oauth keeps shared_disabled marker."""
    _write_shared(shared_env, access="a", refresh="r", generation=1)
    auth._write_profile_xai_shared_reference(enabled=True, generation=1)

    # Simulate logout_command shared-mode profile path (clear then disable).
    auth.clear_provider_auth("xai-oauth")
    auth.disable_profile_xai_shared_auth()

    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    state = profile["providers"]["xai-oauth"]
    assert state.get("enabled") is False
    assert state.get("shared_disabled") is True
    # Canonical grant still present.
    assert shared_env["store"].is_file()
    with pytest.raises(AuthError) as exc:
        auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.code == "xai_shared_profile_disabled"


def test_logout_command_shared_profile_path(shared_env):
    """B1 end-to-end via logout_command."""
    from types import SimpleNamespace

    _write_shared(shared_env, access="a", refresh="r")
    auth._write_profile_xai_shared_reference(enabled=True)

    args = SimpleNamespace(
        provider="xai-oauth",
        reset_config=False,
        global_logout=False,
        shared=False,
        **{"global": False},
    )
    auth.logout_command(args)

    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    state = profile["providers"]["xai-oauth"]
    assert state.get("shared_disabled") is True
    assert state.get("enabled") is False
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared.get("refresh_token") == "r"


def test_load_pool_rewrites_device_code_and_manual(shared_env):
    """A3/A4: load_pool under shared mode discards local RTs."""
    from agent.credential_pool import load_pool

    _write_shared(shared_env, access="shared-at", refresh="shared-rt", generation=2)
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="old-at", refresh="old-rt", manual=False
    )
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    store["credential_pool"]["xai-oauth"].append(
        {
            "id": "manual-1",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "manual-at",
            "refresh_token": "manual-rt",
            "priority": 1,
        }
    )
    shared_env["profile_auth"].write_text(json.dumps(store), encoding="utf-8")

    pool = load_pool("xai-oauth")
    assert pool.has_credentials()
    for entry in pool.entries():
        assert not entry.refresh_token
        assert str(entry.source) == auth.XAI_SHARED_SOURCE

    # Persisted profile must not hold RTs either.
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)


def test_manual_row_does_not_pure_refresh_under_shared(shared_env, monkeypatch):
    """A4: legacy manual entry cannot pure-refresh a local RT outside the lock."""
    from agent.credential_pool import CredentialPool, PooledCredential, AUTH_TYPE_OAUTH

    _write_shared(shared_env, access="shared-at", refresh="shared-rt", generation=1)
    pure_calls = []

    def fake_pure(access, refresh, **kwargs):
        pure_calls.append(refresh)
        return {
            "access_token": "should-not",
            "refresh_token": "should-not",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    monkeypatch.setattr(
        auth,
        "_xai_oauth_discovery",
        lambda *_a, **_k: {"token_endpoint": "https://auth.x.ai/oauth/token"},
    )

    entry = PooledCredential(
        id="manual-1",
        provider="xai-oauth",
        label="manual-1",
        source="manual:device_code",
        auth_type=AUTH_TYPE_OAUTH,
        access_token="manual-at",
        refresh_token="manual-rt",
        priority=0,
    )
    pool = CredentialPool("xai-oauth", [entry])
    # Force refresh path
    refreshed = pool._refresh_entry(entry, force=True)
    assert pure_calls == []  # must not pure-refresh local RT
    assert refreshed is not None
    assert refreshed.refresh_token is None
    assert refreshed.source == auth.XAI_SHARED_SOURCE


def test_write_shared_persists_id_token(shared_env):
    """D3: id_token from refresh is persisted in the canonical store."""
    written = auth._write_shared_xai_state(
        {
            "access_token": "at",
            "refresh_token": "rt",
            "id_token": "id-tok-1",
        }
    )
    assert written.get("id_token") == "id-tok-1"
    on_disk = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert on_disk.get("id_token") == "id-tok-1"


def test_save_strips_root_rt(shared_env, tmp_path, monkeypatch):
    """A5: shared save clears root providers.xai-oauth refresh_token."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    root_auth = hermes_root / "auth.json"
    _seed_legacy_pool_auth(root_auth, access="root-at", refresh="root-rt")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    auth._save_xai_oauth_tokens(
        {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "token_type": "Bearer",
            "id_token": "id-1",
        }
    )
    root = json.loads(root_auth.read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(root)
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "new-rt"
    assert shared.get("id_token") == "id-1"


def test_login_refuses_silent_replace_when_disabled(shared_env, monkeypatch):
    """B2: disabled profile login does not silently clobber the fleet grant."""
    from types import SimpleNamespace

    _write_shared(shared_env, access="fleet-at", refresh="fleet-rt", generation=3)
    auth.disable_profile_xai_shared_auth()

    # Non-interactive decline
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    login_called = {"n": 0}

    def boom_login(**kwargs):
        login_called["n"] += 1
        raise AssertionError("device login must not run without confirmation")

    monkeypatch.setattr(auth, "_xai_oauth_device_code_login", boom_login)
    auth._login_xai_oauth(
        SimpleNamespace(timeout=5, no_browser=True),
        auth.PROVIDER_REGISTRY["xai-oauth"],
        force_new_login=True,
    )
    assert login_called["n"] == 0
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "fleet-rt"


def test_persistence_guard_strips_device_code_rt(shared_env):
    """A2: write_credential_pool under shared mode cannot persist a device_code RT."""
    from hermes_cli.auth import write_credential_pool

    write_credential_pool(
        "xai-oauth",
        [
            {
                "id": "dc-1",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "at",
                "refresh_token": "rt-should-die",
                "priority": 0,
            }
        ],
    )
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    entries = store["credential_pool"]["xai-oauth"]
    assert len(entries) == 1
    assert not entries[0].get("refresh_token")
    assert entries[0]["source"] == auth.XAI_SHARED_SOURCE


# ---------------------------------------------------------------------------
# Round-3 F1–F7: fail-closed / fail-loud / auto-promote
# ---------------------------------------------------------------------------


def test_f1_load_pool_empty_shared_promotes_device_code_rt(shared_env):
    """F1: empty shared + sole device_code RT → promote, never lose the RT."""
    from agent.credential_pool import load_pool

    assert not shared_env["store"].exists()
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="solo-at", refresh="solo-rt", manual=False
    )

    pool = load_pool("xai-oauth")
    assert pool.has_credentials()

    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "solo-rt"
    assert shared["access_token"] == "solo-at"

    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)
    for entry in pool.entries():
        assert not entry.refresh_token
        assert entry.source == auth.XAI_SHARED_SOURCE


def test_f1_load_pool_empty_shared_promotes_manual_rt(shared_env):
    """F1: empty shared + sole manual RT → promote, never lose the RT."""
    from agent.credential_pool import load_pool

    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="man-at", refresh="man-rt", manual=True
    )
    # Pool-only manual (providers tokens also present via seed helper).
    pool = load_pool("xai-oauth")
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "man-rt"
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)
    assert pool.has_credentials()


def test_f1_resolve_empty_shared_promotes_local_rt(shared_env):
    """F1: resolve with empty shared + local RT promotes rather than failing."""
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="r-at", refresh="r-rt", manual=False
    )
    assert not shared_env["store"].exists()
    creds = auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert creds["api_key"] == "r-at"
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["refresh_token"] == "r-rt"


def test_f1_poisoned_shared_write_preserves_local_rt(shared_env, monkeypatch):
    """F1: if shared write fails, local RT must remain and error surfaces."""
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="keep-at", refresh="keep-rt"
    )

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "open", boom)
    with pytest.raises(AuthError):
        auth.ensure_shared_xai_grant_from_local(strip_legacy=True)

    assert not shared_env["store"].exists()
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert auth._auth_store_holds_durable_xai_refresh_token(profile)
    tokens = profile["providers"]["xai-oauth"]["tokens"]
    assert tokens["refresh_token"] == "keep-rt"


def test_f2_resolve_http_fail_closed_profile_disabled(shared_env, monkeypatch):
    """F2: shared mode + profile disabled + surviving manual row → not returned."""
    from tools.xai_http import has_xai_credentials, resolve_xai_http_credentials

    _write_shared(shared_env, access="shared-at", refresh="shared-rt")
    auth.disable_profile_xai_shared_auth()
    # Plant a surviving manual-style row that legacy would have picked.
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    store.setdefault("credential_pool", {})["xai-oauth"] = [
        {
            "id": "manual-surviving",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "manual-at",
            "refresh_token": "manual-rt",
            "priority": 0,
        }
    ]
    shared_env["profile_auth"].write_text(json.dumps(store), encoding="utf-8")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    assert has_xai_credentials() is False
    creds = resolve_xai_http_credentials()
    assert creds.get("api_key") in ("", None)
    assert creds.get("source") != auth.XAI_SHARED_SOURCE
    assert creds.get("api_key") != "manual-at"


def test_f2_resolve_http_fail_closed_empty_canonical(shared_env, monkeypatch):
    """F2: empty shared + no promotable grant + leftover access-only pool row."""
    from tools.xai_http import has_xai_credentials, resolve_xai_http_credentials

    # Access-only pool row (no RT) — cannot promote; must not win under shared.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "orphan",
                            "source": "manual:device_code",
                            "auth_type": "oauth",
                            "access_token": "orphan-at",
                            "priority": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert has_xai_credentials() is False
    creds = resolve_xai_http_credentials()
    assert creds.get("api_key") != "orphan-at"


def test_f2_proxy_fail_closed_profile_disabled(shared_env):
    """F2: proxy does not return a legacy pool row when profile is disabled."""
    from hermes_cli.proxy.adapters.xai import XAIGrokAdapter

    _write_shared(shared_env, access="shared-at", refresh="shared-rt")
    auth.disable_profile_xai_shared_auth()
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    store.setdefault("credential_pool", {})["xai-oauth"] = [
        {
            "id": "manual-1",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "manual-at",
            "refresh_token": "manual-rt",
            "priority": 0,
        }
    ]
    shared_env["profile_auth"].write_text(json.dumps(store), encoding="utf-8")

    adapter = XAIGrokAdapter()
    assert adapter.is_authenticated() is False
    with pytest.raises(RuntimeError, match="disabled"):
        adapter.get_credential()


def test_f3a_enumeration_failure_fails_strip(shared_env, monkeypatch):
    """F3a: profile enumeration failure fails migration/strip (no silent omit)."""

    def boom():
        raise OSError("profiles root unreadable")

    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", boom)
    with pytest.raises(AuthError) as exc:
        auth._strip_legacy_xai_oauth_secrets(include_global_root=True, fail_loud=True)
    assert exc.value.code == "xai_shared_profile_enum_failed"


def test_f3b_disable_marker_write_failure_raises(shared_env, monkeypatch):
    """F3b: disable marker write failure raises — no false success."""
    _write_shared(shared_env, access="a", refresh="r")

    def boom(*_a, **_k):
        raise OSError("cannot write auth.json")

    monkeypatch.setattr(auth, "_save_auth_store", boom)
    with pytest.raises(AuthError) as exc:
        auth.disable_profile_xai_shared_auth()
    assert exc.value.code in {
        "xai_shared_reference_write_failed",
        "xai_shared_disable_failed",
    }
    # Marker must not falsely claim disabled.
    assert auth._profile_xai_shared_disabled() is False


def test_f3c_global_logout_surfaces_strip_failure(shared_env, monkeypatch, capsys):
    """F3c: global logout does not report success when fleet strip fails."""
    from types import SimpleNamespace

    _write_shared(shared_env, access="a", refresh="r")
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="left", refresh="left-rt"
    )

    def fail_strip(**_k):
        raise AuthError(
            "residual rt",
            provider="xai-oauth",
            code="xai_shared_strip_incomplete",
        )

    monkeypatch.setattr(auth, "_strip_legacy_xai_oauth_secrets", fail_strip)
    args = SimpleNamespace(
        provider="xai-oauth",
        reset_config=False,
        global_logout=True,
        shared=False,
        **{"global": False},
    )
    with pytest.raises(SystemExit) as exc:
        auth.logout_command(args)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "Logged out" not in out


def test_f4a_enable_gate_off_no_strip(monkeypatch, tmp_path):
    """F4a: enable with gate OFF does not strip multi-profile RTs."""
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    auth_path = hermes_home / "auth.json"
    _seed_legacy_pool_auth(auth_path, access="local-at", refresh="local-rt")

    with pytest.raises(AuthError) as exc:
        auth.enable_profile_xai_shared_auth()
    assert exc.value.code == "xai_shared_not_enabled"

    store = json.loads(auth_path.read_text(encoding="utf-8"))
    assert auth._auth_store_holds_durable_xai_refresh_token(store)


def test_f4a_enable_empty_shared_no_strip(shared_env):
    """F4a: enable with empty shared store refuses and leaves local RT."""
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="local-at", refresh="local-rt"
    )
    with pytest.raises(AuthError) as exc:
        auth.enable_profile_xai_shared_auth()
    assert exc.value.code == "xai_shared_empty"
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert auth._auth_store_holds_durable_xai_refresh_token(profile)


def test_f4b_disable_gate_off_raises(monkeypatch, tmp_path):
    """F4b: disable-shared requires gate ON; does not strip tokens when off."""
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    auth_path = hermes_home / "auth.json"
    _seed_legacy_pool_auth(auth_path, access="local-at", refresh="local-rt")

    with pytest.raises(AuthError) as exc:
        auth.disable_profile_xai_shared_auth()
    assert exc.value.code == "xai_shared_not_enabled"
    store = json.loads(auth_path.read_text(encoding="utf-8"))
    assert auth._auth_store_holds_durable_xai_refresh_token(store)


def test_f4b_write_reference_gate_off_does_not_strip(monkeypatch, tmp_path):
    """F4b: reference writer must not strip tokens when shared mode is off."""
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    auth_path = hermes_home / "auth.json"
    _seed_legacy_pool_auth(auth_path, access="local-at", refresh="local-rt")

    # Direct writer call with gate off should preserve RTs.
    auth._write_profile_xai_shared_reference(enabled=False)
    store = json.loads(auth_path.read_text(encoding="utf-8"))
    assert auth._auth_store_holds_durable_xai_refresh_token(store)


def test_f7_migrate_preserves_id_token(shared_env):
    """F7: migration copies tokens.id_token into the shared store."""
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "legacy-at",
                            "refresh_token": "legacy-rt",
                            "id_token": "legacy-id-token",
                        },
                        "last_refresh": "2026-06-01T00:00:00Z",
                        "auth_mode": "oauth_device_code",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    written = auth.migrate_xai_oauth_to_shared_store(source="profile", strip_legacy=True)
    assert written.get("id_token") == "legacy-id-token"
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared.get("id_token") == "legacy-id-token"


# ---------------------------------------------------------------------------
# Round-4: quarantine / race / upgrade / corrupt / availability / 429 / remove
# ---------------------------------------------------------------------------


def test_r1_quarantine_then_promote_no_resurrection(shared_env):
    """R1/G6: tombstoned canonical + dead local must NOT resurrect."""
    # Terminal quarantine of the shared grant.
    _write_shared(shared_env, access="old-at", refresh="old-rt", generation=3)
    cleared = auth._clear_shared_xai_state(
        "invalid_grant",
        terminal_error={
            "provider": "xai-oauth",
            "code": "invalid_grant",
            "message": "refresh token revoked",
            "relogin_required": True,
        },
        only_if_refresh_token="old-rt",
        only_if_generation=3,
    )
    assert cleared is True
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert not shared.get("refresh_token")
    assert shared.get("last_auth_error", {}).get("code") == "invalid_grant"

    # Plant a local row that looks like a grant (even a dead one).
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="zombie-at", refresh="zombie-rt"
    )
    store = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    store["credential_pool"]["xai-oauth"][0]["last_status"] = "dead"
    store["credential_pool"]["xai-oauth"][0]["last_error_reason"] = "invalid_grant"
    # Also mark the providers singleton as terminal.
    store["providers"]["xai-oauth"]["last_auth_error"] = {
        "code": "invalid_grant",
        "relogin_required": True,
    }
    shared_env["profile_auth"].write_text(json.dumps(store), encoding="utf-8")

    result = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert result is None
    # Canonical stays tombstoned — no resurrection.
    after = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert not after.get("refresh_token")
    assert after.get("last_auth_error", {}).get("code") == "invalid_grant"
    assert after.get("access_token", "") in ("", None)


def test_r1_live_local_also_refused_when_quarantined(shared_env):
    """R1: even a *live* local grant must not resurrect a tombstoned store."""
    _write_shared(shared_env, access="a", refresh="r", generation=1)
    auth._clear_shared_xai_state(
        "invalid_grant",
        terminal_error={
            "code": "invalid_grant",
            "message": "dead",
            "relogin_required": True,
        },
        only_if_refresh_token="r",
        only_if_generation=1,
    )
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="live-at", refresh="live-rt"
    )
    assert auth.ensure_shared_xai_grant_from_local(strip_legacy=True) is None
    after = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert not after.get("refresh_token")
    assert after.get("last_auth_error")


def test_r2_rejects_dead_and_ambiguous_local(shared_env):
    """R2: dead rows rejected; multiple distinct live RTs are ambiguous."""
    # Dead-only pool: not promotable.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "dead-1",
                            "source": "device_code",
                            "auth_type": "oauth",
                            "access_token": "d-at",
                            "refresh_token": "d-rt",
                            "last_status": "dead",
                            "last_error_reason": "invalid_grant",
                            "priority": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    store = auth._load_auth_store()
    assert auth._xai_oauth_state_from_store(store, sole_live=True) is None

    # Two distinct live RTs → ambiguous.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "a1",
                            "refresh_token": "rt-1",
                        }
                    }
                },
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "other",
                            "source": "manual:device_code",
                            "auth_type": "oauth",
                            "access_token": "a2",
                            "refresh_token": "rt-2",
                            "priority": 1,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    store = auth._load_auth_store()
    with pytest.raises(AuthError) as exc:
        auth._xai_oauth_state_from_store(store, sole_live=True)
    assert exc.value.code == "xai_promote_ambiguous_local"


def test_r3_logout_during_promote_does_not_resurrect(shared_env, monkeypatch):
    """R3: concurrent local logout wins — promotion must not write removed grant.

    Sequence: ensure probe elects live grant → migrate elects live grant →
    concurrent logout clears local → recheck under profile lock sees identity
    gone → refuse write.
    """
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="race-at", refresh="race-rt"
    )
    assert not shared_env["store"].exists()

    real_elect = auth._elect_sole_promotable_xai_under_lock
    calls = {"n": 0}

    def elect_then_logout(path):
        calls["n"] += 1
        state = real_elect(path)
        # After migrate's first elect (call 2: ensure probe is call 1), clear
        # the local source so the recheck (call 3) observes logout.
        if calls["n"] == 2 and state is not None:
            shared_env["profile_auth"].write_text(
                json.dumps({"version": 1, "providers": {}}),
                encoding="utf-8",
            )
        return state

    monkeypatch.setattr(auth, "_elect_sole_promotable_xai_under_lock", elect_then_logout)
    result = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert result is None
    # Canonical must not hold the removed grant.
    if shared_env["store"].exists():
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert shared.get("refresh_token") not in {"race-rt"}
        assert not auth._xai_shared_state_has_usable_tokens(shared)
    # Local remains cleared.
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)
    assert calls["n"] >= 3  # probe + elect + recheck


def test_r4_upgrade_state_sweeps_all_profiles(shared_env, tmp_path, monkeypatch):
    """R4: populated canonical + dormant per-profile RTs → first consume sweeps."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    # Pre-existing canonical grant (deploy upgrade state).
    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=7)
    # Dormant per-profile RTs left from earlier rounds.
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="p-at", refresh="p-rt"
    )
    other = profiles_root / "worker" / "auth.json"
    _seed_legacy_pool_auth(other, access="w-at", refresh="w-rt", manual=True)
    root_auth = hermes_root / "auth.json"
    _seed_legacy_pool_auth(root_auth, access="r-at", refresh="r-rt")

    # Marker must not exist yet.
    marker = auth._xai_sole_owner_marker_path()
    assert not marker.exists()

    result = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert auth._xai_shared_state_has_usable_tokens(result)
    assert result["refresh_token"] == "canon-rt"
    assert marker.is_file()

    for path in (shared_env["profile_auth"], other, root_auth):
        store = json.loads(path.read_text(encoding="utf-8"))
        assert not auth._auth_store_holds_durable_xai_refresh_token(store), path


def test_r5_corrupt_auth_store_fails_sole_owner_audit(shared_env, tmp_path, monkeypatch):
    """R5: unreadable auth.json holding a live RT must FAIL the audit."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    # Write a valid seed then corrupt the file bytes while keeping it present.
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="c-at", refresh="c-rt"
    )
    # Overwrite with non-JSON so parse fails (live RT bytes still on disk).
    shared_env["profile_auth"].write_bytes(
        b'{"version":1,"providers":{"xai-oauth":{"tokens":'
        b'{"access_token":"c-at","refresh_token":"c-rt"}}}'  # truncated / corrupt
        b"NOT-JSON-TRAILER"
    )

    with pytest.raises(AuthError) as exc:
        auth._strip_legacy_xai_oauth_secrets(include_global_root=True, fail_loud=True)
    assert exc.value.code in {
        "xai_shared_strip_incomplete",
        "auth_store_unreadable",
    }
    # Must not certify clean.
    assert "unreadable" in str(exc.value).lower() or "corrupt" in str(exc.value).lower() or "incomplete" in str(exc.value).lower()


def test_r9_shared_mode_429_no_pool_rotation(shared_env, monkeypatch):
    """R9: shared-mode proxy 429 stays on canonical policy (no wrong-ref mark)."""
    from hermes_cli.proxy.adapters.xai import XAIGrokAdapter
    from hermes_cli.proxy.adapters.base import UpstreamCredential

    _write_shared(shared_env, access="shared-at", refresh="shared-rt", generation=1)
    adapter = XAIGrokAdapter()
    rotate_calls = []

    class FakePool:
        def mark_exhausted_and_rotate(self, **kwargs):
            rotate_calls.append(kwargs)
            return None

        def try_refresh_current(self):
            return None

    adapter._pool = FakePool()  # type: ignore[assignment]
    retry = adapter.get_retry_credential(
        failed_credential=UpstreamCredential(
            bearer="shared-at",
            base_url="https://api.x.ai/v1",
            expires_at=None,
        ),
        status_code=429,
    )
    assert retry is None
    assert rotate_calls == []


def test_r10_global_logout_leaves_non_promotable_tombstone(shared_env):
    """R1/R10: global logout tombstones the store (no auto-promote resurrection)."""
    _write_shared(shared_env, access="a", refresh="r", generation=2)
    cleared = auth._clear_shared_xai_state("global_logout")
    assert cleared is True
    assert shared_env["store"].is_file()
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert not shared.get("refresh_token")
    assert shared.get("logged_out") is True or shared.get("last_auth_error")
    assert auth._shared_xai_state_is_quarantined(shared)

    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="z-at", refresh="z-rt"
    )
    assert auth.ensure_shared_xai_grant_from_local(strip_legacy=True) is None
    after = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert after.get("refresh_token") in ("", None)


# ---------------------------------------------------------------------------
# Round-5: H1 race window / H2 cross-store sole-live / H3 verifiable marker /
# H4 fail-closed logout / H5 parent fsync / H6 gate-off byte-identical
# ---------------------------------------------------------------------------


def test_h1_concurrent_real_logout_cannot_install_removed_grant(shared_env, monkeypatch):
    """H1: REAL concurrent clear_provider_auth vs unpatched promote.

    A second thread runs the real logout path (``clear_provider_auth``), which
    acquires the profile auth lock the normal way — no lock-bypass monkeypatch
    of the code under test. Explicit ordering relative to the canonical commit:

    - If promotion committed ``race-rt``, logout must NOT have completed before
      that commit (proves ``logout clears source → stale promotion commits`` is
      impossible under the dual-lock critical section).
    - If promote did not install a grant, the canonical store must NOT hold
      the removed ``race-rt``.
    """
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="race-at", refresh="race-rt"
    )
    assert not shared_env["store"].exists()

    barrier = threading.Barrier(2)
    outcomes = {"promote": "unset", "logout": "unset", "promote_exc": None}
    order: list = []
    order_lock = threading.Lock()

    real_write = auth._write_shared_xai_state

    def write_recording(*args, **kwargs):
        with order_lock:
            order.append("commit_start")
        result = real_write(*args, **kwargs)
        with order_lock:
            order.append("commit_done")
        return result

    monkeypatch.setattr(auth, "_write_shared_xai_state", write_recording)

    def do_promote():
        barrier.wait(timeout=5.0)
        try:
            result = auth.migrate_xai_oauth_to_shared_store(
                source="profile", strip_legacy=True
            )
            outcomes["promote"] = result
        except Exception as exc:
            outcomes["promote"] = None
            outcomes["promote_exc"] = exc

    def do_logout():
        barrier.wait(timeout=5.0)
        try:
            outcomes["logout"] = auth.clear_provider_auth("xai-oauth")
        except Exception as exc:
            outcomes["logout"] = exc
        with order_lock:
            order.append("logout_done")

    t_promote = threading.Thread(target=do_promote, name="h1-promote")
    t_logout = threading.Thread(target=do_logout, name="h1-logout")
    t_promote.start()
    t_logout.start()
    t_promote.join(timeout=30.0)
    t_logout.join(timeout=30.0)
    assert not t_promote.is_alive() and not t_logout.is_alive()

    store = shared_env["store"]
    shared = None
    if store.is_file():
        shared = json.loads(store.read_text(encoding="utf-8"))

    promote_ok = (
        isinstance(outcomes["promote"], dict)
        and outcomes["promote"].get("refresh_token") == "race-rt"
    )
    if promote_ok:
        assert shared is not None
        assert shared.get("refresh_token") == "race-rt"
        assert auth._xai_shared_state_has_usable_tokens(shared)
        # R7-4: promotion committed ⇒ logout did not complete before commit.
        assert "commit_done" in order, order
        assert "logout_done" in order, order
        assert order.index("commit_done") < order.index("logout_done"), order
    else:
        # Promote lost the race or aborted — must not install removed grant.
        if shared is not None:
            assert shared.get("refresh_token") != "race-rt"
            assert not (
                auth._xai_shared_state_has_usable_tokens(shared)
                and shared.get("refresh_token") == "race-rt"
            )
        # Race abort / empty source are acceptable; lock failures are not.
        if outcomes["promote_exc"] is not None:
            assert isinstance(outcomes["promote_exc"], AuthError)
            assert outcomes["promote_exc"].code in {
                "xai_promote_local_race",
                "xai_migrate_source_empty",
            }


def test_h1_profile_lock_held_through_canonical_write(shared_env, monkeypatch):
    """H1: foreign LOCK_NB probe — profile-source lock held across write."""
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="hold-at", refresh="hold-rt"
    )
    held_during_write = {"value": None}
    real_write = auth._write_shared_xai_state

    def write_probe(state, **kwargs):
        # Non-blocking foreign-thread probe against the REAL lock file.
        result_box = {"acquired": None}

        def try_acquire():
            lock_path = shared_env["profile_auth"].with_suffix(".lock")
            try:
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
            except OSError:
                result_box["acquired"] = True
                return
            try:
                if hasattr(auth, "fcntl") and auth.fcntl is not None:
                    try:
                        auth.fcntl.flock(fd, auth.fcntl.LOCK_EX | auth.fcntl.LOCK_NB)
                        result_box["acquired"] = True
                        auth.fcntl.flock(fd, auth.fcntl.LOCK_UN)
                    except (OSError, BlockingIOError):
                        result_box["acquired"] = False
                else:
                    result_box["acquired"] = None
            finally:
                os.close(fd)

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=2.0)
        held_during_write["value"] = result_box["acquired"] is False
        return real_write(state, **kwargs)

    monkeypatch.setattr(auth, "_write_shared_xai_state", write_probe)
    written = auth.migrate_xai_oauth_to_shared_store(
        source="profile", strip_legacy=True
    )
    assert written.get("refresh_token") == "hold-rt"
    assert held_during_write["value"] is True


def test_h2_distinct_profile_and_root_rts_are_ambiguous(shared_env, tmp_path, monkeypatch):
    """H2: live RT-A in profile + live RT-B in root → ambiguous (not profile-first)."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    # Ensure global-root path resolves to hermes_root/auth.json (distinct from
    # the active profile under HERMES_HOME).
    root_auth = hermes_root / "auth.json"
    _seed_legacy_pool_auth(root_auth, access="root-at", refresh="root-rt-B")
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt-A"
    )
    # Force _global_auth_file_path to the distinct root.
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)

    with pytest.raises(AuthError) as exc:
        auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert exc.value.code == "xai_promote_ambiguous_local"
    # Must not have silently installed the profile grant.
    if shared_env["store"].exists():
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert shared.get("refresh_token") not in {"prof-rt-A", "root-rt-B"}


def test_h2_unreadable_store_fails_election(shared_env, tmp_path, monkeypatch):
    """H2 fail-closed: unreadable/corrupt store fails election (never omit)."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    root_auth = hermes_root / "auth.json"
    root_auth.write_text("{not-json", encoding="utf-8")
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt"
    )
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)

    with pytest.raises(AuthError) as exc:
        auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert exc.value.code in {
        "xai_auth_store_unreadable",
        "auth_store_unreadable",
    }
    # Must not have silently promoted the readable profile grant.
    if shared_env["store"].exists():
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert shared.get("refresh_token") != "prof-rt"


def test_h2_availability_probe_no_profile_first_short_circuit(
    shared_env, tmp_path, monkeypatch
):
    """H2: has_xai_credentials / is_authenticated match promoter on ambiguity."""
    from hermes_cli.proxy.adapters.xai import XAIGrokAdapter
    from tools.xai_http import has_xai_credentials

    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    root_auth = hermes_root / "auth.json"
    _seed_legacy_pool_auth(root_auth, access="root-at", refresh="root-rt-B")
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt-A"
    )
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)
    # Empty/never-initialized shared so probes fall into local promote check.
    assert not shared_env["store"].exists()

    assert has_xai_credentials() is False
    adapter = XAIGrokAdapter()
    assert adapter.is_authenticated() is False


def test_h3_stale_marker_reaudits_restored_profile_rt(
    shared_env, tmp_path, monkeypatch
):
    """H3 exact window: marker present, then restore dormant profile RT → re-audit.

    Existence-only markers used to certify a dirty fleet forever. After a
    valid marker is written, introducing/restoring a profile RT must invalidate
    the marker digest and force a fail-loud fleet strip on next consume.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=7)
    # First consume: sweep + write verifiable marker.
    result = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert auth._xai_shared_state_has_usable_tokens(result)
    marker = auth._xai_sole_owner_marker_path()
    assert marker.is_file()
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload.get("fleet_digest")
    digest_before = marker_payload["fleet_digest"]

    # Exact dirtying window: restore a dormant profile RT after marker commit.
    other = profiles_root / "restored" / "auth.json"
    _seed_legacy_pool_auth(other, access="restored-at", refresh="restored-rt")
    assert auth._auth_store_holds_durable_xai_refresh_token(
        json.loads(other.read_text(encoding="utf-8"))
    )

    # Marker must no longer validate against the dirty fleet.
    assert auth._xai_sole_owner_marker_is_valid(generation=7) is False

    # Next consume re-audits and strips the restored RT.
    result2 = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert auth._xai_shared_state_has_usable_tokens(result2)
    store = json.loads(other.read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(store)
    # Marker refreshed — digest must change vs the pre-restore clean fleet
    # (restored path is now present and RT-free, so inventory differs).
    marker_after = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_after.get("fleet_digest")
    assert marker_after["fleet_digest"] != digest_before
    assert auth._xai_sole_owner_marker_is_valid(generation=7) is True


def test_h3_concurrent_restore_cannot_certify_dirty_fleet(
    shared_env, tmp_path, monkeypatch
):
    """H3/R7-4: force the EXACT strip→inventory window with explicit sync.

    Thread A runs the production fleet sole-owner path. A window-forcing hook
    on the real strip (under held fleet locks) signals after strip returns and
    blocks before inventory/marker. Thread B uses the REAL auth-store lock path
    to restore a durable RT in that window — it must BLOCK until marker commit
    releases the locks. No scheduler-luck start-only barrier; no lock-bypass.

    Invariant: a valid marker must NEVER certify a fleet that still holds a
    durable RT; restore_acquired must not precede marker_done.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=5)
    dirty = profiles_root / "dirty" / "auth.json"
    _seed_legacy_pool_auth(dirty, access="dirty-at", refresh="dirty-rt")
    target = profiles_root / "race" / "auth.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    in_strip_inventory_window = threading.Event()
    release_window = threading.Event()
    order: list = []
    order_lock = threading.Lock()
    errors = []

    real_strip = auth._strip_legacy_xai_oauth_secrets_under_held_locks
    real_persist = auth._persist_xai_sole_owner_marker_payload

    def strip_window(*args, **kwargs):
        out = real_strip(*args, **kwargs)
        with order_lock:
            order.append("strip_done")
        # Exact strip→inventory gap: still under fleet locks held by caller.
        in_strip_inventory_window.set()
        assert release_window.wait(timeout=10.0), "timed out in strip→inventory window"
        return out

    def persist_recording(payload):
        result = real_persist(payload)
        with order_lock:
            order.append("marker_done")
        return result

    monkeypatch.setattr(
        auth, "_strip_legacy_xai_oauth_secrets_under_held_locks", strip_window
    )
    monkeypatch.setattr(
        auth, "_persist_xai_sole_owner_marker_payload", persist_recording
    )

    def do_ensure():
        try:
            auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
        except Exception as exc:
            errors.append(("ensure", exc))

    def do_restore():
        assert in_strip_inventory_window.wait(timeout=10.0)
        try:
            with auth._auth_store_lock(target_path=target):
                with order_lock:
                    order.append("restore_acquired")
                _seed_legacy_pool_auth(
                    target, access="race-at", refresh="race-restore-rt"
                )
            with order_lock:
                order.append("restore_done")
        except Exception as exc:
            errors.append(("restore", exc))

    t_a = threading.Thread(target=do_ensure, name="h3-ensure")
    t_b = threading.Thread(target=do_restore, name="h3-restore")
    t_a.start()
    t_b.start()
    # Wait until production path is parked in the strip→inventory window.
    assert in_strip_inventory_window.wait(timeout=10.0)
    # Restore thread must be blocked on the held store lock (not yet acquired).
    time.sleep(0.15)
    with order_lock:
        assert "restore_acquired" not in order, (
            "restore acquired lock during strip→inventory window: " + str(order)
        )
    release_window.set()
    t_a.join(timeout=30.0)
    t_b.join(timeout=30.0)
    assert not t_a.is_alive() and not t_b.is_alive()

    for label, exc in errors:
        if label == "ensure":
            raise AssertionError(f"ensure failed unexpectedly: {exc}") from exc

    with order_lock:
        snapshot = list(order)
    assert "strip_done" in snapshot, snapshot
    assert "marker_done" in snapshot, snapshot
    assert "restore_acquired" in snapshot, snapshot
    # Deterministic ordering: marker commit under fleet locks before restore.
    assert snapshot.index("marker_done") < snapshot.index("restore_acquired"), snapshot

    paths = auth._iter_xai_auth_json_paths(include_global_root=True, fail_loud=True)
    residual = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            store = auth._load_auth_store(path, fail_on_corrupt=True)
        except AuthError:
            residual.append(str(path))
            continue
        if auth._auth_store_holds_durable_xai_refresh_token(store):
            residual.append(str(path))

    if auth._xai_sole_owner_marker_is_valid(generation=5):
        assert residual == [], (
            f"valid sole-owner marker certified dirty fleet: {residual}"
        )
    else:
        assert residual, "expected residual RT when marker is invalid"
        auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
        for path in paths:
            if not path.is_file():
                continue
            store = auth._load_auth_store(path, fail_on_corrupt=True)
            assert not auth._auth_store_holds_durable_xai_refresh_token(store)
        assert auth._xai_sole_owner_marker_is_valid(generation=5) is True


def test_h3_inventory_read_error_rejects_marker(shared_env, tmp_path, monkeypatch):
    """H3 fail-closed: inventory read/stat error must reject marker creation.

    Uses a real unreadable file (chmod 0) — not a tautology monkeypatch — so
    the pre-fix path (hash error entries into the digest) fails this test and
    the fail-closed path raises.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="c-at", refresh="c-rt", generation=2)
    profile = shared_env["profile_auth"]
    profile.write_text(
        json.dumps({"version": 1, "providers": {}}),
        encoding="utf-8",
    )
    marker = auth._xai_sole_owner_marker_path()
    if marker.is_file():
        marker.unlink()

    profile.chmod(0o000)
    try:
        with pytest.raises(AuthError) as exc:
            auth._write_xai_sole_owner_marker(generation=2)
        assert exc.value.code in {
            "xai_shared_fleet_inventory_failed",
            "xai_shared_strip_incomplete",
            "auth_store_unreadable",
            "xai_auth_store_unreadable",
        }
        # Marker must not certify an unreadable fleet.
        assert not marker.is_file() or not auth._xai_sole_owner_marker_is_valid(
            generation=2
        )
    finally:
        profile.chmod(0o600)


def test_h3_rt_after_strip_before_marker_not_certified(
    shared_env, tmp_path, monkeypatch
):
    """H3 window: durable RT reappears after strip returns, before marker commit.

    Wraps the production strip so that immediately after it returns a durable
    RT is restored (the exact pre-fix strip→inventory gap). A valid marker
    must never certify that dirty fleet — either marker creation aborts
    (fail-closed re-audit under held locks) or the marker is invalid while
    residual RT remains.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=9)
    target = profiles_root / "window" / "auth.json"
    _seed_legacy_pool_auth(target, access="pre-at", refresh="pre-rt")
    marker = auth._xai_sole_owner_marker_path()
    if marker.is_file():
        marker.unlink()

    def _restore_after(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            _seed_legacy_pool_auth(
                target, access="window-at", refresh="window-rt"
            )
            return out

        return wrapped

    monkeypatch.setattr(
        auth,
        "_strip_legacy_xai_oauth_secrets",
        _restore_after(auth._strip_legacy_xai_oauth_secrets),
    )
    if hasattr(auth, "_strip_legacy_xai_oauth_secrets_under_held_locks"):
        monkeypatch.setattr(
            auth,
            "_strip_legacy_xai_oauth_secrets_under_held_locks",
            _restore_after(auth._strip_legacy_xai_oauth_secrets_under_held_locks),
        )

    raised = None
    try:
        auth._ensure_xai_fleet_sole_owner_verified(force=True, generation=9)
    except AuthError as exc:
        raised = exc

    residual = auth._auth_store_holds_durable_xai_refresh_token(
        json.loads(target.read_text(encoding="utf-8"))
    )
    valid = auth._xai_sole_owner_marker_is_valid(generation=9)
    # HARD invariant (must fail on pre-fix: marker written over restored RT).
    assert not (valid and residual), (
        "sole-owner marker certified a dirty fleet after strip→marker window restore"
    )
    if residual:
        # Dirty residual is only acceptable when marker creation aborted or
        # the marker does not validate.
        assert raised is not None or not valid


def test_h3_existence_only_marker_never_skips_audit(shared_env, tmp_path, monkeypatch):
    """H3: pre-H3 existence-only markers (no fleet_digest) must not skip audit."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="c-at", refresh="c-rt", generation=3)
    marker = auth._xai_sole_owner_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Existence-only legacy marker (schema/generation only — no fleet_digest).
    marker.write_text(
        json.dumps(
            {
                "verified_at": "2026-07-01T00:00:00Z",
                "generation": 3,
                "schema": 1,
            }
        ),
        encoding="utf-8",
    )
    assert marker.is_file()
    assert auth._xai_sole_owner_marker_is_valid(generation=3) is False

    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="dirty-at", refresh="dirty-rt"
    )
    auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert not auth._auth_store_holds_durable_xai_refresh_token(profile)
    # Marker rewritten with verifiable digest.
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload.get("fleet_digest")


def test_h4_tombstone_write_failure_leaves_non_promotable(shared_env, monkeypatch):
    """H4 exact window: tombstone-persist failure must NOT unlink → resurrect.

    Simulates durable tombstone write failure during global logout. Store must
    remain non-promotable (not bare-absent), failure surfaced, and a surviving
    local RT must not auto-promote afterward.
    """
    _write_shared(shared_env, access="live-at", refresh="live-rt", generation=4)
    prior = shared_env["store"].read_text(encoding="utf-8")

    real_fsync = auth._fsync_parent_dir

    def fsync_boom(path, *, context, code="xai_shared_parent_fsync_failed"):
        if "logout tombstone" in context or "clear" in context:
            raise OSError("simulated tombstone parent fsync failure")
        return real_fsync(path, context=context, code=code)

    monkeypatch.setattr(auth, "_fsync_parent_dir", fsync_boom)

    with pytest.raises(AuthError) as exc:
        auth._clear_shared_xai_state("global_logout")
    assert exc.value.code == "xai_shared_logout_tombstone_failed"
    assert "tombstone" in str(exc.value).lower() or "fsync" in str(exc.value).lower()

    # Store must still exist (no unlink-to-absent fallback).
    assert shared_env["store"].is_file()
    after_bytes = shared_env["store"].read_text(encoding="utf-8")
    assert after_bytes.strip()
    # Surviving local RT must NOT auto-promote into the failed-logout store.
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="surv-at", refresh="surv-rt"
    )
    result = auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    if result is not None:
        # Prior grant still usable (tombstone never committed) — must not be
        # replaced by the surviving local RT.
        assert result.get("refresh_token") != "surv-rt"
        assert result.get("refresh_token") == "live-rt"
    else:
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert shared.get("refresh_token") not in {"surv-rt"}
    assert shared_env["store"].is_file()
    # No bare-absent: either prior grant or a tombstone — never unlinked.
    # (fsync fails after os.replace in our helper, so file may already be
    # tombstoned on disk; either prior or tombstone is non-promotable for
    # surv-rt resurrection. What must not happen is store absence.)
    assert shared_env["store"].is_file()


def test_h5_parent_dir_fsync_failure_on_canonical_write_is_loud(
    shared_env, monkeypatch
):
    """H5: parent-dir open/fsync failure on canonical write is surfaced."""
    calls = {"n": 0}
    real_fsync = auth._fsync_parent_dir

    def fsync_fail(path, *, context, code="xai_shared_parent_fsync_failed"):
        calls["n"] += 1
        if "canonical write" in context:
            raise auth.AuthError(
                f"Failed to open parent directory for fsync ({context})",
                provider="xai-oauth",
                code=code,
                relogin_required=False,
            )
        return real_fsync(path, context=context, code=code)

    monkeypatch.setattr(auth, "_fsync_parent_dir", fsync_fail)
    with pytest.raises(AuthError) as exc:
        auth._write_shared_xai_state(
            {
                "access_token": "a",
                "refresh_token": "r",
                "token_type": "Bearer",
            },
            bump_generation=True,
        )
    assert exc.value.code == "xai_shared_persist_failed"
    assert calls["n"] >= 1
    # Must not swallow — error message is observable.
    assert "fsync" in str(exc.value).lower() or "parent" in str(exc.value).lower()


def test_h6_gate_off_dead_first_row_is_legacy_first_wins(monkeypatch, tmp_path):
    """H6/G7: gate-off store with dead/suppressed FIRST token-bearing row matches 60891a4ef.

    Round-4 strict reject was applied globally and changed gate-off resolution.
    Gate-off must be legacy first-wins again (byte-identical to 60891a4ef).
    """
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    # Gate OFF — no shared mode.
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    assert auth._xai_shared_auth_enabled() is False

    auth_path = hermes_home / "auth.json"
    # Dead/suppressed FIRST token-bearing provider row — legacy first-wins
    # returns it; strict shared-mode would reject it.
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "dead-at",
                            "refresh_token": "dead-rt",
                        },
                        "last_status": "dead",
                        "last_auth_error": {
                            "code": "invalid_grant",
                            "relogin_required": True,
                        },
                        "source": "device_code",
                    }
                },
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "live-later",
                            "source": "device_code",
                            "auth_type": "oauth",
                            "access_token": "live-at",
                            "refresh_token": "live-rt",
                            "priority": 1,
                        }
                    ]
                },
                "suppressed_sources": {"xai-oauth": ["device_code"]},
            }
        ),
        encoding="utf-8",
    )

    store = auth._load_auth_store()
    # sole_live=False (gate-off default): legacy first-wins → dead-rt.
    state = auth._xai_oauth_state_from_store(store)
    assert state is not None
    tokens = state.get("tokens") if isinstance(state.get("tokens"), dict) else {}
    assert tokens.get("refresh_token") == "dead-rt"
    assert tokens.get("access_token") == "dead-at"

    # End-to-end gate-off read path.
    read = auth._read_xai_oauth_tokens()
    assert read["tokens"]["refresh_token"] == "dead-rt"

    # Shared-mode sole_live path still rejects dead/suppressed.
    assert auth._xai_oauth_state_from_store(store, sole_live=True) is None


# ---------------------------------------------------------------------------
# Round-7: invalid-shape fail-closed / locked marker validation /
# refresh-stable digest / deterministic election→commit race
# ---------------------------------------------------------------------------


def _hidden_rt_list_payload():
    """Hermes R7-1 repro shape: JSON list wrapping a providers grant."""
    return [
        {
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "hidden-at",
                        "refresh_token": "hidden-rt",
                    }
                }
            }
        }
    ]


def _assert_raises_auth_store_unreadable(fn):
    with pytest.raises(AuthError) as exc:
        fn()
    assert exc.value.code in {
        "auth_store_unreadable",
        "xai_auth_store_unreadable",
        "xai_shared_strip_incomplete",
        "xai_shared_fleet_inventory_failed",
    }
    return exc.value


def test_r7_1_list_shaped_root_store_fails_closed_election_strip_audit(
    shared_env, tmp_path, monkeypatch
):
    """R7-1 Hermes repro: list-shaped root holding hidden RT must fail closed.

    Pre-fix (bf58c54e6): parseable list normalizes to empty → profile promote
    succeeds, hidden RT survives, marker certifies clean after consume.
    Post-fix: election / strip / pre-commit RT audit raise auth_store_unreadable
    (or wrap it); hidden RT is NOT promoted-over and NOT certified clean.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    root_auth = hermes_root / "auth.json"
    root_auth.write_text(json.dumps(_hidden_rt_list_payload()), encoding="utf-8")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)

    # Profile has a promotable grant — pre-fix would elect it and ignore root.
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt"
    )

    # Strict load itself rejects the shape.
    _assert_raises_auth_store_unreadable(
        lambda: auth._load_auth_store(root_auth, fail_on_corrupt=True)
    )

    # Election / migrate must not promote-over the hidden RT.
    with pytest.raises(AuthError) as exc:
        auth.migrate_xai_oauth_to_shared_store(source="profile", strip_legacy=True)
    assert exc.value.code in {
        "auth_store_unreadable",
        "xai_auth_store_unreadable",
        "xai_shared_strip_incomplete",
        "xai_shared_fleet_inventory_failed",
        "xai_promote_ambiguous_local",
    }
    # Hidden RT must still be on disk (not stripped as "empty"); profile grant
    # must not have been installed into the canonical store over it.
    assert "hidden-rt" in root_auth.read_text(encoding="utf-8")
    if shared_env["store"].is_file():
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        # If anything was written, it must not be a successful promote-over.
        if auth._xai_shared_state_has_usable_tokens(shared):
            pytest.fail("promoted over invalid-shaped root store holding hidden RT")

    # Fleet strip fail-closed on the invalid-shaped store.
    _assert_raises_auth_store_unreadable(
        lambda: auth._strip_legacy_xai_oauth_secrets(
            include_global_root=True, fail_loud=True
        )
    )

    # Pre-commit RT-free audit fail-closed.
    _assert_raises_auth_store_unreadable(
        lambda: auth._audit_fleet_refresh_token_free([root_auth], fail_loud=True)
    )

    # Marker must not certify clean over the hidden RT.
    assert auth._xai_sole_owner_marker_is_valid(generation=1) is False


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-store",
        42,
        3.14,
        True,
    ],
)
def test_r7_1_string_number_shaped_store_fails_closed(shared_env, raw, tmp_path, monkeypatch):
    """R7-1: string/number/bool-shaped stores fail closed under strict load."""
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    bad = hermes_root / "auth.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: bad)
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt"
    )

    _assert_raises_auth_store_unreadable(
        lambda: auth._load_auth_store(bad, fail_on_corrupt=True)
    )
    with pytest.raises(AuthError):
        auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    if shared_env["store"].is_file():
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert not (
            auth._xai_shared_state_has_usable_tokens(shared)
            and shared.get("refresh_token") == "prof-rt"
        )


def test_r7_1_empty_dict_and_absent_remain_empty(shared_env):
    """R7-1 preserve: absent file and ``{}`` still load as empty (not errors)."""
    missing = shared_env["hermes_home"] / "no-such-auth.json"
    assert not missing.exists()
    empty_absent = auth._load_auth_store(missing, fail_on_corrupt=True)
    assert empty_absent == {"version": auth.AUTH_STORE_VERSION, "providers": {}}

    empty_path = shared_env["hermes_home"] / "empty-auth.json"
    empty_path.write_text("{}", encoding="utf-8")
    empty_dict = auth._load_auth_store(empty_path, fail_on_corrupt=True)
    assert empty_dict == {"version": auth.AUTH_STORE_VERSION, "providers": {}}

    providers_empty = shared_env["hermes_home"] / "providers-empty.json"
    providers_empty.write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    ok = auth._load_auth_store(providers_empty, fail_on_corrupt=True)
    assert isinstance(ok.get("providers"), dict)


def test_r7_2_marker_validation_holds_locks_during_inventory(
    shared_env, tmp_path, monkeypatch
):
    """R7-2: validation inventory is lock-disciplined (no unlocked skip).

    Force a store change attempt DURING validation's inventory read via an
    explicit barrier. Concurrent writer must block on the store lock (or
    validation must re-check under lock) — a skip must not be granted from an
    unlocked/stale inventory snapshot while a dirty change is in flight.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=3)
    # Establish a valid clean marker.
    auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert auth._xai_sole_owner_marker_is_valid(generation=3) is True

    target = profiles_root / "during-validation" / "auth.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Path is enumerated (parent profile dir exists) but file absent → clean.

    in_inventory = threading.Event()
    release_inventory = threading.Event()
    order: list = []
    order_lock = threading.Lock()
    inventory_calls = {"n": 0}

    real_inventory = auth._xai_fleet_auth_inventory

    def inventory_window(*, fail_loud=True, paths=None):
        inventory_calls["n"] += 1
        # Only the first validation inventory is the forced TOCTOU window.
        if inventory_calls["n"] == 1:
            with order_lock:
                order.append("inventory_start")
            in_inventory.set()
            assert release_inventory.wait(timeout=10.0), "inventory window timeout"
            result = real_inventory(fail_loud=fail_loud, paths=paths)
            with order_lock:
                order.append("inventory_done")
            return result
        return real_inventory(fail_loud=fail_loud, paths=paths)

    monkeypatch.setattr(auth, "_xai_fleet_auth_inventory", inventory_window)

    validation_result = {"value": None, "exc": None}

    def do_validate():
        try:
            validation_result["value"] = auth._xai_sole_owner_marker_is_valid(
                generation=3
            )
        except Exception as exc:
            validation_result["exc"] = exc

    def do_dirty():
        assert in_inventory.wait(timeout=10.0)
        # Real lock path — must block while validation holds fleet store locks.
        with auth._auth_store_lock(target_path=target):
            with order_lock:
                order.append("dirty_acquired")
            _seed_legacy_pool_auth(
                target, access="during-at", refresh="during-rt"
            )
        with order_lock:
            order.append("dirty_done")

    t_v = threading.Thread(target=do_validate, name="r7-2-validate")
    t_d = threading.Thread(target=do_dirty, name="r7-2-dirty")
    t_v.start()
    t_d.start()
    assert in_inventory.wait(timeout=10.0)
    time.sleep(0.15)
    with order_lock:
        assert "dirty_acquired" not in order, (
            "dirty writer acquired store lock during locked validation inventory: "
            + str(order)
        )
    release_inventory.set()
    t_v.join(timeout=30.0)
    t_d.join(timeout=30.0)
    assert not t_v.is_alive() and not t_d.is_alive()
    assert validation_result["exc"] is None

    with order_lock:
        snapshot = list(order)
    assert "inventory_done" in snapshot, snapshot
    assert "dirty_acquired" in snapshot, snapshot
    # Dirty write cannot interleave under the inventory locks.
    assert snapshot.index("inventory_done") < snapshot.index("dirty_acquired"), snapshot

    # After concurrent dirt, marker must not validate (digest change) and audit
    # must run on next ensure (strip the restored RT).
    assert auth._xai_sole_owner_marker_is_valid(generation=3) is False
    auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    store = auth._load_auth_store(target, fail_on_corrupt=True)
    assert not auth._auth_store_holds_durable_xai_refresh_token(store)
    assert auth._xai_sole_owner_marker_is_valid(generation=3) is True


def test_r7_3_refresh_metadata_does_not_churn_fleet_digest(
    shared_env, tmp_path, monkeypatch
):
    """R7-3: normal refresh metadata must NOT invalidate the fleet-clean marker.

    Hermes repro: marker_valid_before_refresh True → after profile
    shared_generation/metadata write → still True. Reintroducing an RT must
    invalidate and force re-audit strip.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )

    _write_shared(shared_env, access="canon-at", refresh="canon-rt", generation=4)
    # Profile already holds non-secret shared reference (post-login settle state)
    # so the refresh path only mutates volatile metadata — not path presence.
    auth._write_profile_xai_shared_reference(
        enabled=True,
        last_refresh="2026-07-01T00:00:00Z",
        generation=4,
        set_active=True,
    )
    auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    marker = auth._xai_sole_owner_marker_path()
    assert marker.is_file()
    digest_before = json.loads(marker.read_text(encoding="utf-8"))["fleet_digest"]
    assert auth._xai_sole_owner_marker_is_valid(generation=4) is True
    marker_valid_before_refresh = True

    # Simulate post-refresh profile metadata settle (shared_generation bump).
    auth._write_profile_xai_shared_reference(
        enabled=True,
        last_refresh="2026-07-18T12:00:00Z",
        generation=5,
        set_active=True,
    )
    profile = json.loads(shared_env["profile_auth"].read_text(encoding="utf-8"))
    assert profile["providers"]["xai-oauth"].get("shared_generation") == 5
    assert profile["providers"]["xai-oauth"].get("last_refresh") == (
        "2026-07-18T12:00:00Z"
    )
    # Tokens must remain absent after metadata-only rewrite.
    assert "tokens" not in profile["providers"]["xai-oauth"]
    assert not profile["providers"]["xai-oauth"].get("refresh_token")

    marker_valid_after_refresh = auth._xai_sole_owner_marker_is_valid(generation=5)
    assert marker_valid_before_refresh is True
    assert marker_valid_after_refresh is True, (
        "metadata-only refresh churned fleet_digest / invalidated sole-owner marker"
    )
    digest_after = json.loads(marker.read_text(encoding="utf-8"))["fleet_digest"]
    assert digest_after == digest_before

    # Real RT reintroduction MUST still invalidate.
    other = profiles_root / "reintro" / "auth.json"
    _seed_legacy_pool_auth(other, access="re-at", refresh="re-rt")
    assert auth._xai_sole_owner_marker_is_valid(generation=5) is False
    auth.ensure_shared_xai_grant_from_local(strip_legacy=True)
    assert not auth._auth_store_holds_durable_xai_refresh_token(
        auth._load_auth_store(other, fail_on_corrupt=True)
    )
    assert auth._xai_sole_owner_marker_is_valid(generation=5) is True


def test_r7_4_h2_election_commit_blocks_unselected_store_inject(
    shared_env, tmp_path, monkeypatch
):
    """R7-4/H2: inject distinct RT into UNSELECTED store during election→commit.

    Promote holds BOTH store locks through elect + recheck + canonical write.
    A concurrent writer injecting a distinct live RT into the unselected root
    must BLOCK until commit (or promote raises ambiguous). Explicit events —
    no scheduler-luck, no lock-bypass of the production path.
    Fail-pre on 74eff54fa (root unlocked mid-promote) / pass-post.
    """
    hermes_root = tmp_path / "home" / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "profiles").mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    root_auth = hermes_root / "auth.json"
    # Root starts empty (unselected). Profile holds the sole live grant.
    root_auth.write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)
    _seed_legacy_pool_auth(
        shared_env["profile_auth"], access="prof-at", refresh="prof-rt-A"
    )
    assert not shared_env["store"].exists()

    in_commit_window = threading.Event()
    release_commit = threading.Event()
    order: list = []
    order_lock = threading.Lock()
    outcomes = {"promote": None, "promote_exc": None, "inject_exc": None}

    real_write = auth._write_shared_xai_state

    def write_window(*args, **kwargs):
        with order_lock:
            order.append("commit_start")
        # Still under dual store locks held by migrate.
        in_commit_window.set()
        assert release_commit.wait(timeout=10.0), "commit window timeout"
        result = real_write(*args, **kwargs)
        with order_lock:
            order.append("commit_done")
        return result

    monkeypatch.setattr(auth, "_write_shared_xai_state", write_window)

    def do_promote():
        try:
            outcomes["promote"] = auth.migrate_xai_oauth_to_shared_store(
                source="profile", strip_legacy=True
            )
        except Exception as exc:
            outcomes["promote_exc"] = exc

    def do_inject():
        assert in_commit_window.wait(timeout=10.0)
        try:
            with auth._auth_store_lock(target_path=root_auth):
                with order_lock:
                    order.append("inject_acquired")
                _seed_legacy_pool_auth(
                    root_auth, access="root-at", refresh="root-rt-B"
                )
            with order_lock:
                order.append("inject_done")
        except Exception as exc:
            outcomes["inject_exc"] = exc

    t_p = threading.Thread(target=do_promote, name="r7-h2-promote")
    t_i = threading.Thread(target=do_inject, name="r7-h2-inject")
    t_p.start()
    t_i.start()
    assert in_commit_window.wait(timeout=10.0)
    time.sleep(0.15)
    with order_lock:
        assert "inject_acquired" not in order, (
            "inject acquired unselected-store lock during election→commit: "
            + str(order)
        )
    release_commit.set()
    t_p.join(timeout=30.0)
    t_i.join(timeout=30.0)
    assert not t_p.is_alive() and not t_i.is_alive()

    with order_lock:
        snapshot = list(order)
    assert "commit_start" in snapshot, snapshot
    # Concurrent inject must not complete before commit under held locks.
    if "inject_acquired" in snapshot and "commit_done" in snapshot:
        assert snapshot.index("commit_done") < snapshot.index("inject_acquired"), (
            snapshot
        )

    if outcomes["promote_exc"] is not None:
        # Acceptable: ambiguous if inject somehow became visible, or lock fail.
        assert isinstance(outcomes["promote_exc"], AuthError)
        assert outcomes["promote_exc"].code in {
            "xai_promote_ambiguous_local",
            "xai_promote_local_race",
            "xai_auth_store_lock_failed",
            "auth_store_unreadable",
        }
    else:
        assert outcomes["promote"] is not None
        assert outcomes["promote"].get("refresh_token") == "prof-rt-A"
        shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
        assert shared.get("refresh_token") == "prof-rt-A"
        # Inject ran after commit — must not have been silently merged mid-window.
        assert "inject_acquired" in snapshot, snapshot
