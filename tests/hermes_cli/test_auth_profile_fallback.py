"""Tests for cross-profile auth fallback.

When ``HERMES_HOME`` points to a named profile, ``read_credential_pool()``
and ``get_provider_auth_state()`` fall back to the global-root
``auth.json`` per-provider when the profile has no entries for that
provider.  Writes still target the profile only.

See the #18594 follow-up report: profile workers couldn't see providers
authenticated only at the global root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_auth_store(pool: dict | None = None, providers: dict | None = None) -> dict:
    store: dict = {"version": 1}
    if pool is not None:
        store["credential_pool"] = pool
    if providers is not None:
        store["providers"] = providers
    return store


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Set up a global root + an active profile under Path.home()/.hermes/profiles/coder.

    * Path.home() -> tmp_path
    * Global root -> tmp_path/.hermes            (has its own auth.json fixture)
    * Profile     -> tmp_path/.hermes/profiles/coder   (active, HERMES_HOME points here)

    This mirrors the real "named profile mounted under the default root"
    layout that profile users actually have on disk.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_dir = global_root / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    return {"global": global_root, "profile": profile_dir}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# read_credential_pool — provider-slice reads
# ---------------------------------------------------------------------------


def test_profile_with_zero_entries_falls_back_to_global(profile_env):
    """Empty profile pool inherits the global-root entries for that provider."""
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "glob-1",
            "label": "global-key",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-global",
        }],
    }))
    # Profile auth.json: exists but has no openrouter entries.
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={}))

    entries = read_credential_pool("openrouter")
    assert len(entries) == 1
    assert entries[0]["id"] == "glob-1"
    assert entries[0]["access_token"] == "sk-or-global"


def test_profile_with_entries_fully_shadows_global(profile_env):
    """Once the profile has any entries for a provider, global is ignored."""
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "glob-1",
            "label": "global-key",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-global",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "prof-1",
            "label": "profile-key",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-profile",
        }],
    }))

    entries = read_credential_pool("openrouter")
    assert len(entries) == 1
    assert entries[0]["id"] == "prof-1"
    assert entries[0]["access_token"] == "sk-or-profile"


def test_per_provider_shadowing_is_independent(profile_env):
    """Profile can override one provider while inheriting another from global."""
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "glob-or",
            "label": "global-or",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-global",
        }],
        "anthropic": [{
            "id": "glob-ant",
            "label": "global-ant",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-ant-global",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        # Profile has openrouter only — anthropic should still fall back.
        "openrouter": [{
            "id": "prof-or",
            "label": "profile-or",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-profile",
        }],
    }))

    or_entries = read_credential_pool("openrouter")
    ant_entries = read_credential_pool("anthropic")
    assert [e["id"] for e in or_entries] == ["prof-or"]
    assert [e["id"] for e in ant_entries] == ["glob-ant"]


def test_missing_global_auth_file_is_safe(profile_env):
    """Profile processes that never had a global auth.json still work."""
    from hermes_cli.auth import read_credential_pool

    # No global auth.json written at all.
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "prof-1",
            "label": "profile",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-profile",
        }],
    }))

    assert read_credential_pool("openrouter")[0]["id"] == "prof-1"
    assert read_credential_pool("anthropic") == []


def test_malformed_global_auth_file_does_not_break_profile_read(profile_env):
    (profile_env["global"] / "auth.json").write_text("{not valid json")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "prof-1",
            "label": "profile",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-profile",
        }],
    }))

    from hermes_cli.auth import read_credential_pool

    # Profile reads still work; malformed global is silently ignored.
    assert read_credential_pool("openrouter")[0]["id"] == "prof-1"
    # And no fallback for anthropic since global is unreadable.
    assert read_credential_pool("anthropic") == []


# ---------------------------------------------------------------------------
# read_credential_pool — whole-pool reads (provider_id=None)
# ---------------------------------------------------------------------------


def test_whole_pool_merges_global_providers_when_missing_locally(profile_env):
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "glob-or",
            "label": "global-or",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-global",
        }],
        "anthropic": [{
            "id": "glob-ant",
            "label": "global-ant",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-ant-global",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "prof-or",
            "label": "profile-or",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-or-profile",
        }],
    }))

    pool = read_credential_pool(None)
    # Profile wins for openrouter, global fills in anthropic.
    assert [e["id"] for e in pool["openrouter"]] == ["prof-or"]
    assert [e["id"] for e in pool["anthropic"]] == ["glob-ant"]


# ---------------------------------------------------------------------------
# get_provider_auth_state — singleton fallback
# ---------------------------------------------------------------------------


def test_provider_auth_state_falls_back_to_global_when_profile_has_none(profile_env):
    from hermes_cli.auth import get_provider_auth_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "nous-global", "refresh_token": "rt-global"},
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))

    state = get_provider_auth_state("nous")
    assert state is not None
    assert state["access_token"] == "nous-global"


def test_provider_auth_state_profile_wins_when_present(profile_env):
    from hermes_cli.auth import get_provider_auth_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "nous-global"},
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "nous-profile"},
    }))

    state = get_provider_auth_state("nous")
    assert state is not None
    assert state["access_token"] == "nous-profile"


def test_provider_auth_state_returns_none_when_neither_has_it(profile_env):
    from hermes_cli.auth import get_provider_auth_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={}))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))

    assert get_provider_auth_state("nous") is None


# ---------------------------------------------------------------------------
# _load_provider_state — internal global fallback (issue #18594 follow-up)
#
# Several runtime helpers (notably ``resolve_nous_runtime_credentials`` and
# ``resolve_nous_access_token``) call ``_load_provider_state`` directly with
# a profile-loaded auth store rather than going through
# ``get_provider_auth_state``. Without the fallback wired into
# ``_load_provider_state`` itself, those helpers raise ``"Hermes is not
# logged into Nous Portal"`` even though the user has a valid global Nous
# login. These tests pin the per-provider shadowing into the helper.
# ---------------------------------------------------------------------------


def test_load_provider_state_falls_back_to_global(profile_env):
    """When the loaded profile store has no provider entry, fall back to global."""
    from hermes_cli.auth import _load_auth_store, _load_provider_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "global-nous-token", "refresh_token": "rt"},
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))

    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "nous")
    assert state is not None
    assert state["access_token"] == "global-nous-token"


def test_load_provider_state_profile_wins_over_global(profile_env):
    from hermes_cli.auth import _load_auth_store, _load_provider_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "global-token"},
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "profile-token"},
    }))

    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "nous")
    assert state is not None
    assert state["access_token"] == "profile-token"


def test_load_provider_state_returns_none_when_neither_has_it(profile_env):
    from hermes_cli.auth import _load_auth_store, _load_provider_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={}))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))

    auth_store = _load_auth_store()
    assert _load_provider_state(auth_store, "nous") is None


def test_load_provider_state_classic_mode_no_fallback(tmp_path, monkeypatch):
    """In classic mode there is no global to fall back to; behavior is unchanged."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    hermes_home = tmp_path / "classic"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write(hermes_home / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "classic-token"},
    }))

    from hermes_cli.auth import _load_auth_store, _load_provider_state

    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "nous")
    assert state is not None
    assert state["access_token"] == "classic-token"
    # Absent providers still return None.
    assert _load_provider_state(auth_store, "anthropic") is None


def test_profile_codex_refresh_updates_global_source_without_local_shadow(
    profile_env, monkeypatch,
):
    """Refreshing inherited Codex OAuth must rotate the canonical root store.

    Codex refresh tokens can be single-use.  Persisting the rotated pair into
    the active profile would leave the root token stale and create a local
    entry that shadows future root re-authentication.
    """
    from hermes_cli import auth

    _write(profile_env["global"] / "auth.json", _make_auth_store(
        providers={
            "openai-codex": {
                "tokens": {
                    "access_token": "root-access-old",
                    "refresh_token": "root-refresh-old",
                },
                "last_refresh": "2026-01-01T00:00:00Z",
            },
        },
        pool={
            "openai-codex": [{
                "id": "default",
                "label": "default",
                "auth_type": "oauth",
                "source": "device_code",
                "access_token": "root-access-old",
                "refresh_token": "root-refresh-old",
            }],
        },
    ))
    _write(
        profile_env["profile"] / "auth.json",
        _make_auth_store(providers={}, pool={}),
    )

    monkeypatch.setattr(
        auth,
        "refresh_codex_oauth_pure",
        lambda access_token, refresh_token, timeout_seconds=20.0: {
            "access_token": "root-access-new",
            "refresh_token": "root-refresh-new",
            "last_refresh": "2026-01-02T00:00:00Z",
        },
    )

    resolved = auth.resolve_codex_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "root-access-new"
    root_store = json.loads((profile_env["global"] / "auth.json").read_text())
    assert (
        root_store["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "root-refresh-new"
    )
    assert (
        root_store["credential_pool"]["openai-codex"][0]["refresh_token"]
        == "root-refresh-new"
    )
    profile_store = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert "openai-codex" not in profile_store.get("providers", {})
    assert "openai-codex" not in profile_store.get("credential_pool", {})


def test_profile_codex_cli_recovery_repairs_invalid_global_source(
    profile_env, monkeypatch,
):
    """CLI recovery of malformed inherited state must not create a shadow."""
    from hermes_cli import auth

    _write(profile_env["global"] / "auth.json", _make_auth_store(
        providers={
            "openai-codex": {
                "tokens": {"refresh_token": "invalid-root-refresh"},
            },
        },
        pool={},
    ))
    _write(
        profile_env["profile"] / "auth.json",
        _make_auth_store(providers={}, pool={}),
    )
    monkeypatch.setattr(
        auth,
        "_import_codex_cli_tokens",
        lambda: {
            "access_token": "recovered-access",
            "refresh_token": "recovered-refresh",
        },
    )

    resolved = auth.resolve_codex_runtime_credentials(
        refresh_if_expiring=False,
    )

    assert resolved["api_key"] == "recovered-access"
    root_store = json.loads((profile_env["global"] / "auth.json").read_text())
    assert (
        root_store["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "recovered-refresh"
    )
    profile_store = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert "openai-codex" not in profile_store.get("providers", {})


def test_file_lock_reentrancy_is_scoped_to_lock_path(tmp_path, monkeypatch):
    """Nested profile/root locks must each acquire a kernel-level lock."""
    import threading

    from hermes_cli import auth

    if auth.fcntl is None:
        pytest.skip("fcntl unavailable")
    real_flock = auth.fcntl.flock
    exclusive_acquires = []

    def tracking_flock(fd, operation):
        if operation & auth.fcntl.LOCK_EX:
            exclusive_acquires.append(fd)
        return real_flock(fd, operation)

    monkeypatch.setattr(auth.fcntl, "flock", tracking_flock)
    holder = threading.local()
    with auth._file_lock(
        tmp_path / "profile.lock", holder, 1.0, "profile lock timeout"
    ):
        with auth._file_lock(
            tmp_path / "root.lock", holder, 1.0, "root lock timeout"
        ):
            pass

    assert len(exclusive_acquires) == 2


def test_profile_codex_resolver_sees_global_pool_only_credential(profile_env):
    """Direct resolver fallback must honor the same global pool ownership."""
    from hermes_cli import auth

    _write(
        profile_env["global"] / "auth.json",
        _make_auth_store(
            providers={},
            pool={
                "openai-codex": [{
                    "id": "manual-1",
                    "label": "manual",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:device_code",
                    "access_token": "root-pool-access",
                    "refresh_token": "root-pool-refresh",
                }]
            },
        ),
    )
    _write(
        profile_env["profile"] / "auth.json",
        _make_auth_store(providers={}, pool={}),
    )

    resolved = auth.resolve_codex_runtime_credentials(
        refresh_if_expiring=False,
    )

    assert resolved["api_key"] == "root-pool-access"
    assert resolved["source"] == "credential_pool"


def test_load_provider_state_malformed_global_does_not_break_profile(profile_env):
    """A corrupt global auth.json must not break profile reads."""
    (profile_env["global"] / "auth.json").write_text("{not valid json")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "profile-token"},
    }))

    from hermes_cli.auth import _load_auth_store, _load_provider_state

    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "nous")
    assert state is not None
    assert state["access_token"] == "profile-token"


# ---------------------------------------------------------------------------
# Classic mode — no fallback path should ever trigger
# ---------------------------------------------------------------------------


def test_classic_mode_does_not_double_read_same_file(tmp_path, monkeypatch):
    """In classic mode (HERMES_HOME == global root), no fallback path runs.

    This guards against the merge accidentally duplicating entries when the
    profile and global resolve to the same directory.
    """
    # Put Path.home() under a subdir so the seat belt in _auth_file_path()
    # sees tmp_path/home/.hermes as the "real home" — which is NOT equal
    # to the HERMES_HOME we set (tmp_path/classic), so the guard passes.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    hermes_home = tmp_path / "classic"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write(hermes_home / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "only",
            "label": "classic",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-classic",
        }],
    }))

    from hermes_cli.auth import read_credential_pool, _global_auth_file_path

    # Classic mode: HERMES_HOME is set to a custom path that is NOT under
    # ~/.hermes/profiles/ — get_default_hermes_root() returns HERMES_HOME
    # itself, so the profile root and global root are the same directory,
    # and the helper correctly returns None (no fallback).
    assert _global_auth_file_path() is None
    # And the read should return exactly one entry (not two).
    entries = read_credential_pool("openrouter")
    assert len(entries) == 1
    assert entries[0]["id"] == "only"


# ---------------------------------------------------------------------------
# Writes stay scoped to the profile
# ---------------------------------------------------------------------------


def test_write_credential_pool_targets_profile_not_global(profile_env):
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "glob-1",
            "label": "global",
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": "sk-global",
        }],
    }))

    write_credential_pool("openrouter", [{
        "id": "prof-new",
        "label": "profile-new",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-profile-new",
    }])

    # Global auth.json unchanged.
    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert global_data["credential_pool"]["openrouter"][0]["id"] == "glob-1"

    # Profile auth.json holds the new entry.
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert profile_data["credential_pool"]["openrouter"][0]["id"] == "prof-new"

    # Subsequent read returns profile (shadows global).
    assert [e["id"] for e in read_credential_pool("openrouter")] == ["prof-new"]
