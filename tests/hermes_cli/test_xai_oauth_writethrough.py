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

from hermes_cli import auth, auth_xai


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
    root = tmp_path / "hermes"
    profile_path = root / "profiles" / "work" / "auth.json"
    profile_path.parent.mkdir(parents=True)
    root_path = root / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(profile_path.parent))
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


@pytest.mark.parametrize("profile_state", [None, {}, {"tokens": {"token_type": "Bearer"}}],
                         ids=["owned", "borrowed", "quarantined-profile"])
def test_runtime_refresh_locks_the_owning_store(profile_and_root, monkeypatch, profile_state):
    profile, root = profile_and_root
    borrowed = profile_state is not None
    store = root if borrowed else profile
    if profile_state:
        _write_store(profile, {"version": 1, "providers": {"xai-oauth": profile_state}})
    other_store = profile if borrowed else root
    other_before = other_store.read_bytes() if other_store.exists() else None
    _write_store(store, {"version": 1, "active_provider": "another-provider", "providers": {
        "xai-oauth": {
            "tokens": {"access_token": "access-original", "refresh_token": "refresh-original"},
            "discovery": {"token_endpoint": "https://example.invalid/token"},
        },
    }})

    def refresh(_access, _refresh, **_kwargs):
        assert auth._auth_lock_holder_for(profile).depth > 0
        assert bool(getattr(auth._auth_lock_holder_for(root), "depth", 0)) == borrowed
        return {
            "access_token": "access-renewed", "refresh_token": "refresh-renewed",
            "last_refresh": "2026-01-01T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)
    result = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert result["api_key"] == "access-renewed"
    persisted = _read_store(store)
    assert persisted["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "refresh-renewed"
    assert persisted["active_provider"] == "another-provider"
    assert (other_store.read_bytes() if other_store.exists() else None) == other_before


@pytest.mark.parametrize("borrowed", [True, False], ids=["root", "profile"])
def test_proactive_refresh_adopts_a_peer_rotation_inside_the_skew(profile_and_root, monkeypatch, borrowed):
    profile, root = profile_and_root
    store = root if borrowed else profile
    old_pair = {"access_token": "access-original", "refresh_token": "refresh-original"}
    new_pair = {"access_token": "access-renewed", "refresh_token": "refresh-renewed"}
    state = {"tokens": old_pair, "discovery": {"token_endpoint": "https://example.invalid/token"}}
    _write_store(store, {"version": 1, "providers": {"xai-oauth": state}})
    read_initial = auth_xai._read_xai_oauth_tokens
    expiry_checks = []

    def observe_then_rotate():
        observed = read_initial()
        assert observed["tokens"] == old_pair
        with auth._auth_store_lock(target_path=store):
            _write_store(store, {"version": 1, "providers": {"xai-oauth": {**state, "tokens": new_pair}}})
        return observed

    def inside_skew(access, skew):
        assert skew == 120
        expiry_checks.append(access)
        return True  # Even the peer's replacement would qualify for a new refresh.

    def unexpected_refresh(*_args, **_kwargs):
        pytest.fail("the proactive caller should use the credential already in the store")

    monkeypatch.setattr(auth_xai, "_read_xai_oauth_tokens", observe_then_rotate)
    monkeypatch.setattr(auth_xai, "_xai_access_token_is_expiring", inside_skew)
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", unexpected_refresh)
    result = auth.resolve_xai_oauth_runtime_credentials(refresh_skew_seconds=120)

    assert expiry_checks[0] == old_pair["access_token"]
    assert result["api_key"] == new_pair["access_token"]
    assert _read_store(store)["providers"]["xai-oauth"]["tokens"] == new_pair
    assert not (profile if borrowed else root).exists()


def test_terminal_runtime_refresh_quarantines_the_borrowed_root(profile_and_root, monkeypatch):
    profile, root = profile_and_root
    _write_store(profile, {"version": 1, "providers": {"xai-oauth": {"tokens": {}}}})
    profile_before = profile.read_bytes()
    _write_store(root, {"version": 1, "providers": {"xai-oauth": {
        "tokens": {"access_token": "stored-access", "refresh_token": "stored-refresh", "token_type": "Bearer"},
        "discovery": {"token_endpoint": "https://example.invalid/token"},
    }}})

    def refresh(*_args, **_kwargs):
        raise auth.AuthError(
            "Refresh declined", provider="xai-oauth", code="xai_refresh_failed", relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)

    with pytest.raises(auth.AuthError, match="Refresh declined"):
        auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    persisted = _read_store(root)["providers"]["xai-oauth"]
    assert persisted["tokens"] == {"token_type": "Bearer"}
    assert persisted["last_auth_error"]["code"] == "xai_refresh_failed"
    assert profile.read_bytes() == profile_before


@pytest.mark.parametrize("borrowed", [True, False], ids=["root", "profile"])
@pytest.mark.parametrize("terminal", [False, True], ids=["refreshed", "quarantined"])
def test_runtime_refresh_updates_only_its_pool_row(profile_and_root, monkeypatch, borrowed, terminal):
    profile, root = profile_and_root
    store = root if borrowed else profile
    provider = {"tokens": {}, "discovery": {"token_endpoint": "https://example.invalid/token"}}
    first = {"source": "manual:device_code", "access_token": "selected-access", "refresh_token": "selected-refresh"}
    second = {"id": "other", "access_token": "other-access", "refresh_token": "other-refresh"}
    original = {"version": 1, "active_provider": "another-provider", "providers": {"xai-oauth": provider},
                "credential_pool": {"xai-oauth": [first, second]}}
    _write_store(store, original)

    def refresh(access, refresh_token, **_kwargs):
        assert (access, refresh_token) == ("selected-access", "selected-refresh")
        if terminal:
            raise auth.AuthError("Refresh declined", provider="xai-oauth", code="xai_refresh_failed",
                                 relogin_required=True)
        return {"access_token": "renewed-access", "refresh_token": "renewed-refresh",
                "last_refresh": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)
    if terminal:
        with pytest.raises(auth.AuthError, match="Refresh declined"):
            auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)
        assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "other-access"
    else:
        result = auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)
        assert result["api_key"] == "renewed-access"
        assert auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "renewed-access"

    persisted = _read_store(store)
    changed, untouched = persisted["credential_pool"]["xai-oauth"]
    assert untouched == second
    assert persisted["providers"] == original["providers"]
    assert persisted["active_provider"] == original["active_provider"]
    assert not (profile if borrowed else root).exists()
    assert changed["id"].startswith("legacy-")
    if terminal:
        assert changed["last_status"] == "dead"
        assert "access_token" not in changed and "refresh_token" not in changed
    else:
        assert changed["refresh_token"] == "renewed-refresh"
        assert changed["last_refresh"] == "2026-01-01T00:00:00Z"
