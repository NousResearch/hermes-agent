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
from pathlib import Path

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


def test_disabled_global_fallback_rejects_root_xai_and_prevents_write_through(
    tmp_path, monkeypatch
):
    """Profile isolation covers both xAI root reads and refresh write-through."""
    from hermes_cli import managed_scope
    import hermes_cli.config as config_module

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root_dir = tmp_path / ".hermes"
    profile_dir = root_dir / "profiles" / "isolated"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    (profile_dir / "config.yaml").write_text(
        "auth:\n  global_fallback: false\n", encoding="utf-8"
    )
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    profile_path = profile_dir / "auth.json"
    root_path = root_dir / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {
        "version": 1,
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": "root-access",
                    "refresh_token": "root-refresh",
                }
            }
        },
    })

    with pytest.raises(auth.AuthError, match="No xAI OAuth credentials stored"):
        auth._read_xai_oauth_tokens()

    auth._save_xai_oauth_tokens({
        "access_token": "profile-access",
        "refresh_token": "profile-refresh",
    })
    assert _read_store(profile_path)["providers"]["xai-oauth"]["tokens"] == {
        "access_token": "profile-access",
        "refresh_token": "profile-refresh",
    }
    assert _read_store(root_path)["providers"]["xai-oauth"]["tokens"] == {
        "access_token": "root-access",
        "refresh_token": "root-refresh",
    }
