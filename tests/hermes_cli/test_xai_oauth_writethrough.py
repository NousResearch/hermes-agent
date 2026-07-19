"""Regression tests for xAI OAuth refresh write-through to the global root.

Companion to ``test_xai_oauth_profile_auth.py``. That file covers the READ
fallback (profile -> credential pool -> global root). These cover the WRITE
side: when a profile that has no own ``providers.xai-oauth`` block refreshes
the (rotating) grant it resolved from the root fallback, the rotated tokens
must be written back to the global root too. Otherwise root keeps a revoked
refresh token and every other profile reading root's stale grant dies with
``invalid_grant`` once its access token expires (issue #43589).

The tests drive the real ``_save_xai_oauth_tokens`` against real on-disk auth
stores (profile + root under ``tmp_path``) rather than mocking the save
boundary, so they exercise the actual atomic write path.
"""

import json

import pytest

from hermes_cli import auth


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    Returns (profile_path, root_path). The pytest seat belt in
    ``_write_through_xai_oauth_to_global_root`` only refuses the *real* user's
    ``$HOME/.hermes/auth.json``; a tmp_path root is allowed, so we point HOME
    away from the tmp root to keep the guard from tripping on these fixtures.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    # Keep the pytest write seat belt from matching our tmp root.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def test_refresh_writes_through_to_root_when_profile_has_no_own_state(profile_and_root):
    """Profile reading root's grant must push rotated tokens back to root."""
    profile_path, root_path = profile_and_root
    # Profile has NO own xai-oauth block (reads root via fallback).
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    rotated = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "token_type": "Bearer",
    }
    auth._save_xai_oauth_tokens(rotated)

    # Profile got the rotated chain (existing behavior).
    profile = _read_store(profile_path)
    assert profile["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-refresh"

    # AND the global root no longer holds the revoked refresh token (#43589).
    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["access_token"] == "new-access"
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-refresh"


def test_refresh_does_not_touch_root_when_profile_has_own_state(profile_and_root):
    """A profile that genuinely shadows root must NOT clobber the root grant."""
    profile_path, root_path = profile_and_root
    # Profile has its OWN xai-oauth block: it shadows root legitimately.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "profile-old",
                        "refresh_token": "profile-old-refresh",
                    }
                }
            },
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "root-untouched",
                        "refresh_token": "root-untouched-refresh",
                    }
                }
            },
        },
    )

    auth._save_xai_oauth_tokens(
        {"access_token": "profile-new", "refresh_token": "profile-new-refresh"}
    )

    profile = _read_store(profile_path)
    assert profile["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "profile-new-refresh"

    # Root is a separate grant chain — must be left exactly as-is.
    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["access_token"] == "root-untouched"
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "root-untouched-refresh"


def test_write_through_is_noop_in_classic_mode(tmp_path, monkeypatch):
    """Classic mode (profile == root) already saves to root; no double write."""
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    # Classic mode: _global_auth_file_path returns None.
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: None)
    _write_store(profile_path, {"version": 1, "providers": {}})

    # Should not raise and should persist to the single store.
    auth._save_xai_oauth_tokens(
        {"access_token": "a", "refresh_token": "r"}
    )
    store = _read_store(profile_path)
    assert store["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "r"


def test_write_through_failure_does_not_break_profile_save(profile_and_root, monkeypatch):
    """A failed root write-through must not break the profile's own save."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {}})

    # Make the root write blow up; the profile save must still succeed.
    real_save = auth._save_auth_store

    def _exploding_save(store, target_path=None):
        if target_path is not None and target_path == root_path:
            raise OSError("simulated root write failure")
        return real_save(store, target_path)

    monkeypatch.setattr(auth, "_save_auth_store", _exploding_save)

    auth._save_xai_oauth_tokens({"access_token": "a", "refresh_token": "r"})

    profile = _read_store(profile_path)
    assert profile["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "r"


def test_singleton_refresh_holds_root_source_lock_and_preserves_active_providers(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {"version": 1, "active_provider": "openrouter", "providers": {}},
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    },
                    "discovery": {
                        "client_id": "xai-client",
                        "token_endpoint": "https://auth.x.ai/oauth/token",
                        "redirect_uri": "http://localhost:1455/auth/callback",
                    },
                }
            },
        },
    )

    lock_observed = []

    def _refresh(*_args, **_kwargs):
        holder = auth._auth_lock_holder_for(root_path)
        lock_observed.append(getattr(holder, "depth", 0) > 0)
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "token_type": "Bearer",
            "last_refresh": "2026-07-19T04:00:00+00:00",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh)

    resolved = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "new-access"
    assert lock_observed == [True]
    assert _read_store(profile_path)["active_provider"] == "openrouter"
    root = _read_store(root_path)
    assert root["active_provider"] == "openai-codex"
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-refresh"


def test_two_root_owned_rotations_do_not_create_profile_shadow_and_reach_sibling(
    tmp_path, monkeypatch
):
    profile_one = tmp_path / "profiles" / "one" / "auth.json"
    profile_two = tmp_path / "profiles" / "two" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    active_profile = [profile_one]
    monkeypatch.setattr(auth, "_auth_file_path", lambda: active_profile[0])
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    for profile_path in (profile_one, profile_two):
        _write_store(
            profile_path,
            {"version": 1, "active_provider": "openrouter", "providers": {}},
        )
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "access-0",
                        "refresh_token": "refresh-0",
                    },
                    "discovery": {
                        "token_endpoint": "https://auth.x.ai/oauth/token"
                    },
                }
            },
        },
    )
    refresh_inputs = []

    def _rotate(_access_token, refresh_token, **_kwargs):
        refresh_inputs.append(refresh_token)
        generation = len(refresh_inputs)
        return {
            "access_token": f"access-{generation}",
            "refresh_token": f"refresh-{generation}",
            "token_type": "Bearer",
            "last_refresh": f"2026-07-19T0{generation}:00:00+00:00",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _rotate)

    first = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    second = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert first["api_key"] == "access-1"
    assert second["api_key"] == "access-2"
    assert refresh_inputs == ["refresh-0", "refresh-1"]
    assert "xai-oauth" not in _read_store(profile_one)["providers"]
    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "refresh-2"

    active_profile[0] = profile_two
    sibling = auth.resolve_xai_oauth_runtime_credentials(
        force_refresh=False,
        refresh_if_expiring=False,
    )
    assert sibling["api_key"] == "access-2"
    assert "xai-oauth" not in _read_store(profile_two)["providers"]


@pytest.mark.parametrize(
    "profile_state",
    [
        {
            "auth_mode": "oauth_pkce",
            "discovery": {"client_id": "profile-metadata-only"},
        },
        {
            "tokens": {"access_token": "partial-access"},
            "discovery": {"client_id": "profile-partial"},
        },
        {
            "tokens": {},
            "last_auth_error": {
                "code": "xai_refresh_failed",
                "relogin_required": True,
            },
        },
    ],
    ids=["metadata-only", "partial-token", "quarantined"],
)
def test_unusable_profile_state_does_not_own_root_refresh(
    profile_and_root, monkeypatch, profile_state
):
    profile_path, root_path = profile_and_root
    profile_store = {
        "version": 1,
        "active_provider": "openrouter",
        "providers": {"xai-oauth": profile_state},
    }
    _write_store(profile_path, profile_store)
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "xai-oauth": {
                    "auth_mode": "oauth_device_code",
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    },
                    "discovery": {
                        "token_endpoint": "https://auth.x.ai/oauth/token"
                    },
                }
            },
        },
    )
    refresh_inputs = []

    def _rotate(_access_token, refresh_token, **_kwargs):
        refresh_inputs.append(refresh_token)
        return {
            "access_token": "root-access-new",
            "refresh_token": "root-refresh-new",
            "token_type": "Bearer",
            "last_refresh": "2026-07-19T08:00:00+00:00",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _rotate)

    resolved = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "root-access-new"
    assert refresh_inputs == ["root-refresh"]
    assert _read_store(profile_path) == profile_store
    root = _read_store(root_path)
    assert root["active_provider"] == "openai-codex"
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "root-refresh-new"


def test_terminal_root_refresh_does_not_quarantine_unusable_profile_state(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    profile_store = {
        "version": 1,
        "active_provider": "openrouter",
        "providers": {
            "xai-oauth": {
                "tokens": {},
                "last_auth_error": {
                    "code": "prior_profile_failure",
                    "relogin_required": True,
                },
            }
        },
    }
    _write_store(profile_path, profile_store)
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "dead-root-access",
                        "refresh_token": "dead-root-refresh",
                    },
                    "discovery": {
                        "token_endpoint": "https://auth.x.ai/oauth/token"
                    },
                }
            },
        },
    )

    def _terminal_refresh(*_args, **_kwargs):
        raise auth.AuthError(
            "Root refresh grant was revoked",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _terminal_refresh)

    with pytest.raises(auth.AuthError, match="revoked"):
        auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert _read_store(profile_path) == profile_store
    root = _read_store(root_path)
    assert root["active_provider"] == "openai-codex"
    root_state = root["providers"]["xai-oauth"]
    assert not root_state["tokens"].get("access_token")
    assert not root_state["tokens"].get("refresh_token")
    assert root_state["last_auth_error"]["reason"] == "runtime_refresh_failure"


def test_terminal_pool_refresh_quarantines_root_without_activation(
    profile_and_root, monkeypatch
):
    from agent.credential_pool import load_pool

    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {"version": 1, "active_provider": "openrouter", "providers": {}},
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "dead-access",
                        "refresh_token": "dead-refresh",
                    },
                    "discovery": {
                        "client_id": "xai-client",
                        "token_endpoint": "https://auth.x.ai/oauth/token",
                        "redirect_uri": "http://localhost:1455/auth/callback",
                    },
                }
            },
        },
    )

    pool = load_pool("xai-oauth")
    assert pool.select() is not None
    lock_observed = []

    def _terminal_refresh(*_args, **_kwargs):
        holder = auth._auth_lock_holder_for(root_path)
        lock_observed.append(getattr(holder, "depth", 0) > 0)
        raise auth.AuthError(
            "Refresh session has been revoked",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _terminal_refresh)

    assert pool.try_refresh_current() is None

    assert lock_observed == [True]
    profile = _read_store(profile_path)
    root = _read_store(root_path)
    assert profile["active_provider"] == "openrouter"
    assert "xai-oauth" not in profile["providers"]
    assert root["active_provider"] == "openai-codex"
    root_state = root["providers"]["xai-oauth"]
    assert not root_state["tokens"].get("access_token")
    assert not root_state["tokens"].get("refresh_token")
    assert root_state["last_auth_error"]["reason"] == "credential_pool_refresh_failure"
