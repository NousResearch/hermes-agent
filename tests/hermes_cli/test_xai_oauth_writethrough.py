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


def test_successful_save_clears_stale_last_auth_error(profile_and_root):
    """A successful token save resolves any prior failure — the stale
    ``last_auth_error`` record must be cleared so it can't mislead future
    triage (a revoked-token error left behind after a successful re-login
    looks like a live outage but isn't).
    """
    profile_path, _root_path = profile_and_root
    # Profile carries a leftover error record from an earlier failed refresh.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "dead-access",
                        "refresh_token": "dead-refresh",
                    },
                    "last_auth_error": {
                        "provider": "xai-oauth",
                        "code": "xai_refresh_failed",
                        "message": "Refresh token has been revoked",
                        "relogin_required": True,
                        "at": "2026-07-13T10:33:29Z",
                    },
                }
            },
        },
    )

    # A fresh, successful login/refresh persists good tokens.
    auth._save_xai_oauth_tokens({"access_token": "new-access", "refresh_token": "new-refresh"})

    state = _read_store(profile_path)["providers"]["xai-oauth"]
    assert state["tokens"]["refresh_token"] == "new-refresh"
    # The stale error record is gone.
    assert "last_auth_error" not in state
