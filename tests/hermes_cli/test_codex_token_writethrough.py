"""Regression tests for Codex OAuth refresh write-through to the global root.

Mirrors ``test_xai_oauth_writethrough.py`` for the Codex family (#87503):
Codex refresh tokens are single-use with rotation-family reuse detection,
so when a profile that has no own ``providers.openai-codex`` block refreshes
the grant it resolved from the root fallback, the rotated chain must land
back in root — including the ``credential_pool`` entries the runtime selects
credentials from. Otherwise root keeps the consumed refresh token, the next
process to read it replays it, and OpenAI revokes the whole rotation family.

The tests drive the real ``_save_codex_tokens`` against real on-disk auth
stores (profile + root under ``tmp_path``). All token values are synthetic
placeholders assembled by ``_pair`` — no real credentials are involved.
"""

import json

import pytest

from hermes_cli import auth


def _pair(prefix: str) -> dict:
    """Synthetic OAuth pair for fixtures/assertions (not credentials)."""
    return {
        "access_token": f"{prefix}-at",
        "refresh_token": f"{prefix}-rt",
    }


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk."""
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    # Keep the pytest write seat belt from matching our tmp root.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def test_profile_refresh_of_root_grant_writes_through(profile_and_root):
    """#87503: rotating the root-resolved grant must reach the root store —
    singleton AND credential-pool entries — without forking a shadowing
    profile key."""
    profile_path, root_path = profile_and_root
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "auth_mode": "chatgpt",
                    "tokens": _pair("old"),
                }
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "provider": "openai-codex",
                        "source": "device_code",
                        **_pair("old"),
                    }
                ]
            },
        },
    )
    _write_store(profile_path, {"version": 1, "providers": {}})

    rotated = _pair("new")
    auth._save_codex_tokens(
        rotated,
        last_refresh="2026-08-16T00:00:00Z",
    )

    root = _read_store(root_path)
    assert (
        root["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == rotated["refresh_token"]
    ), "root singleton must hold the rotated refresh token"
    pool_entry = root["credential_pool"]["openai-codex"][0]
    assert pool_entry["refresh_token"] == rotated["refresh_token"]
    assert pool_entry["access_token"] == rotated["access_token"]

    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("providers", {}), (
        "profile must not gain a shadowing providers.openai-codex key — "
        "it would disable the write-through on the next refresh (#74339)"
    )


def test_profile_owned_state_saves_to_profile_only(profile_and_root):
    """A profile with its own openai-codex block keeps the existing
    profile-local save; root is untouched."""
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "auth_mode": "chatgpt",
                    "tokens": _pair("prof"),
                }
            },
        },
    )
    _write_store(root_path, {"version": 1, "providers": {}})

    rotated = _pair("next")
    auth._save_codex_tokens(
        rotated,
        last_refresh="2026-08-16T00:00:00Z",
    )

    profile = _read_store(profile_path)
    assert (
        profile["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == rotated["refresh_token"]
    )
    root = _read_store(root_path)
    assert "openai-codex" not in root.get("providers", {})


def test_classic_mode_still_saves_single_store(tmp_path, monkeypatch):
    """Classic mode (profile == root): unchanged single-store save."""
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: None)
    _write_store(profile_path, {"version": 1, "providers": {}})

    rotated = _pair("classic")
    auth._save_codex_tokens(
        rotated,
        last_refresh="2026-08-16T00:00:00Z",
    )

    store = _read_store(profile_path)
    assert (
        store["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == rotated["refresh_token"]
    )
