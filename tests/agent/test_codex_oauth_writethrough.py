"""Regression tests for openai-codex OAuth refresh write-through to global root.

Companion to ``tests/hermes_cli/test_xai_oauth_writethrough.py``. That file
covers the xAI WRITE side; these cover the equivalent for openai-codex.

The hazard (mirrors xAI #43589): openai-codex rotates the refresh_token on
every refresh. When a profile that has no own ``providers.openai-codex`` block
resolves the grant from the global-root fallback and the credential pool
rotates it, the rotated chain is written only to the PROFILE auth store —
leaving root holding a now-revoked refresh token. Every other profile reading
root's stale grant then dies with ``refresh_token_reused`` once its access
token expires.

These tests drive the real
``CredentialPool._sync_device_code_entry_to_auth_store`` against real on-disk
auth stores (profile + root under ``tmp_path``) rather than mocking the save
boundary, so they exercise the actual atomic write + write-through path.
"""

import json

import pytest

from hermes_cli import auth
from agent import credential_pool as cp
from agent.credential_pool import CredentialPool, PooledCredential


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _codex_entry() -> PooledCredential:
    """A pooled openai-codex entry carrying freshly-rotated tokens."""
    return PooledCredential(
        provider="openai-codex",
        id="openai-codex",
        label="codex",
        auth_type="oauth",
        priority=0,
        source="device_code",
        access_token="new-access",
        refresh_token="new-refresh",
        last_refresh="2026-06-18T00:00:00Z",
    )


def _sync(entry: PooledCredential) -> None:
    pool = CredentialPool("openai-codex", [entry])
    pool._sync_device_code_entry_to_auth_store(entry)


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    Returns (profile_path, root_path). The pytest seat belt in
    ``_write_through_codex_oauth_to_global_root`` only refuses the *real*
    user's ``$HOME/.hermes/auth.json``; a tmp_path root is allowed, so we
    point HOME away from the tmp root to keep the guard from tripping.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    # Patch both namespaces: auth.py owns the write path + the write-through
    # helper's own resolution, while credential_pool.py binds its own imported
    # references for the gate check (`from hermes_cli.auth import ...`).
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setattr(cp, "_global_auth_file_path", lambda: root_path)
    # Keep the pytest write seat belt from matching our tmp root.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def test_refresh_writes_through_to_root_when_profile_has_no_own_state(profile_and_root):
    """Profile reading root's grant must push rotated tokens back to root."""
    profile_path, root_path = profile_and_root
    # Profile has NO own openai-codex block (reads root via fallback). The
    # codex sync path requires an existing tokens dict to mutate, which
    # _load_provider_state resolves from the root fallback.
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    _sync(_codex_entry())

    # Profile got the rotated chain (existing behavior).
    profile = _read_store(profile_path)
    assert profile["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"

    # AND the global root no longer holds the revoked refresh token (#43589).
    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"]["access_token"] == "new-access"
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"


def test_refresh_does_not_touch_root_when_profile_has_own_state(profile_and_root):
    """A profile that genuinely shadows root must NOT clobber the root grant."""
    profile_path, root_path = profile_and_root
    # Profile has its OWN openai-codex block: it shadows root legitimately.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
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
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-untouched",
                        "refresh_token": "root-untouched-refresh",
                    }
                }
            },
        },
    )

    _sync(_codex_entry())

    profile = _read_store(profile_path)
    assert profile["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"

    # Root is a separate grant chain — must be left exactly as-is.
    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"]["access_token"] == "root-untouched"
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "root-untouched-refresh"


def test_write_through_is_noop_in_classic_mode(tmp_path, monkeypatch):
    """Classic mode (profile == root) already saves to root; no double write."""
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    # Classic mode: _global_auth_file_path returns None in both namespaces.
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(cp, "_global_auth_file_path", lambda: None)
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    # Should not raise and should persist to the single store.
    _sync(_codex_entry())
    store = _read_store(profile_path)
    assert store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"


def test_write_through_failure_does_not_break_profile_save(profile_and_root, monkeypatch):
    """A failed root write-through must not break the profile's own save."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    # Make the root write blow up; the profile save must still succeed.
    real_save = auth._save_auth_store

    def _exploding_save(store, target_path=None):
        if target_path is not None and target_path == root_path:
            raise OSError("simulated root write failure")
        return real_save(store, target_path)

    monkeypatch.setattr(auth, "_save_auth_store", _exploding_save)

    _sync(_codex_entry())

    profile = _read_store(profile_path)
    assert profile["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"
