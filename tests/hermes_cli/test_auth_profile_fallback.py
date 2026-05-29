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


def test_codex_pool_shares_only_global_device_code_entry(profile_env):
    """Only the canonical Codex token family is shared across named profiles."""
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "glob-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "global-at",
            "refresh_token": "global-rt",
        }, {
            "id": "root-manual-codex",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "root-manual-at",
            "refresh_token": "root-manual-rt",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "stale-profile-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "stale-profile-at",
            "refresh_token": "stale-profile-rt",
        }, {
            "id": "profile-manual-codex",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "profile-manual-at",
            "refresh_token": "profile-manual-rt",
        }],
    }))

    assert [entry["id"] for entry in read_credential_pool("openai-codex")] == [
        "glob-codex",
        "profile-manual-codex",
    ]
    assert [entry["id"] for entry in read_credential_pool(None)["openai-codex"]] == [
        "glob-codex",
        "profile-manual-codex",
    ]


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


def test_write_codex_credential_pool_splits_shared_and_profile_entries(profile_env):
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "root-manual",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "root-manual-at",
            "refresh_token": "root-manual-rt",
        }],
    }))

    write_credential_pool("openai-codex", [
        {
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "shared-at",
            "refresh_token": "shared-rt",
        },
        {
            "id": "profile-dashboard",
            "source": "manual:dashboard_device_code",
            "auth_type": "oauth",
            "access_token": "dashboard-at",
            "refresh_token": "dashboard-rt",
        },
        {
            "id": "profile-api-key",
            "source": "manual:api_key",
            "auth_type": "api_key",
            "access_token": "sk-profile",
        },
    ])

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert [e["id"] for e in global_data["credential_pool"]["openai-codex"]] == [
        "root-manual",
        "shared-codex",
    ]
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert [e["id"] for e in profile_data["credential_pool"]["openai-codex"]] == [
        "profile-dashboard",
        "profile-api-key",
    ]
    assert [e["id"] for e in read_credential_pool("openai-codex")] == [
        "shared-codex",
        "profile-dashboard",
        "profile-api-key",
    ]


def test_codex_profile_pool_flush_preserves_newer_shared_root_entry(profile_env):
    from agent.credential_pool import PooledCredential, load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "old-at",
            "refresh_token": "old-rt",
        }],
    }))
    pool = load_pool("openai-codex")

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "new-at",
            "refresh_token": "new-rt",
        }],
    }))
    pool.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "profile-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-profile",
    }))

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["access_token"] == "new-at"
    assert shared["refresh_token"] == "new-rt"
    assert shared.get("last_status") is None
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert [e["id"] for e in profile_data["credential_pool"]["openai-codex"]] == [
        "profile-api-key",
    ]


def test_codex_profile_pool_flush_does_not_restore_removed_shared_root_entry(profile_env):
    from agent.credential_pool import PooledCredential, load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "old-at",
            "refresh_token": "old-rt",
        }],
    }))
    pool = load_pool("openai-codex")

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [],
    }))
    pool.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "profile-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-profile",
    }))

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert global_data["credential_pool"]["openai-codex"] == []
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert [e["id"] for e in profile_data["credential_pool"]["openai-codex"]] == [
        "profile-api-key",
    ]


def test_codex_profile_pool_flush_does_not_apply_status_to_newer_access_token(profile_env):
    from dataclasses import replace

    from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED, load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "old-at",
            "refresh_token": "shared-rt",
        }],
    }))
    pool = load_pool("openai-codex")
    shared = pool.entries()[0]
    pool._replace_entry(shared, replace(
        shared,
        last_status=STATUS_EXHAUSTED,
        last_error_code=429,
    ))

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "new-at",
            "refresh_token": "shared-rt",
        }],
    }))
    pool.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "profile-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-profile",
    }))

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["access_token"] == "new-at"
    assert shared["refresh_token"] == "shared-rt"
    assert shared.get("last_status") is None
    assert shared.get("last_error_code") is None


def test_codex_profile_pool_flush_persists_matching_shared_cooldown(profile_env):
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "shared-at",
            "refresh_token": "shared-rt",
        }],
    }))
    pool = load_pool("openai-codex")
    assert pool.select() is not None

    assert pool.mark_exhausted_and_rotate(status_code=429) is None

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["refresh_token"] == "shared-rt"
    assert shared["last_status"] == STATUS_EXHAUSTED
    assert shared["last_error_code"] == 429


def test_codex_profile_pool_reset_persists_matching_shared_status_clear(profile_env):
    from agent.credential_pool import load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "shared-at",
            "refresh_token": "shared-rt",
            "last_status": "exhausted",
            "last_status_at": 1234,
            "last_error_code": 429,
        }],
    }))
    pool = load_pool("openai-codex")

    assert pool.reset_statuses() == 1

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["refresh_token"] == "shared-rt"
    assert shared["last_status"] is None
    assert shared["last_status_at"] is None
    assert shared["last_error_code"] is None


def test_codex_profile_load_persists_newly_seeded_shared_root_entry(profile_env):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import CODEX_REFRESH_OWNER

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "openai-codex": {
            "tokens": {
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            },
            "refresh_owner": CODEX_REFRESH_OWNER,
        },
    }))

    pool = load_pool("openai-codex")

    assert pool.entries()[0].source == "device_code"
    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["source"] == "device_code"
    assert shared["access_token"] == "shared-at"
    assert shared["refresh_token"] == "shared-rt"


def test_codex_profile_pool_flush_deduplicates_shared_source_with_stale_id(profile_env):
    from hermes_cli.auth import write_credential_pool

    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "canonical-row",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            }],
        },
    })

    write_credential_pool("openai-codex", [{
        "id": "stale-process-row",
        "source": "device_code",
        "auth_type": "oauth",
        "access_token": "shared-at",
        "refresh_token": "shared-rt",
    }], preserve_shared_entries=True)

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    entries = global_data["credential_pool"]["openai-codex"]
    assert len(entries) == 1
    assert entries[0]["id"] == "canonical-row"
    assert entries[0]["refresh_token"] == "shared-rt"


def test_codex_profile_local_flush_does_not_clear_newer_shared_cooldown(profile_env):
    from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED, load_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "shared-codex",
            "source": "device_code",
            "auth_type": "oauth",
            "access_token": "shared-at",
            "refresh_token": "shared-rt",
        }],
    }))
    first = load_pool("openai-codex")
    second = load_pool("openai-codex")
    assert first.select() is not None

    assert first.mark_exhausted_and_rotate(status_code=429) is None
    second.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "profile-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-profile",
    }))

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    shared = global_data["credential_pool"]["openai-codex"][0]
    assert shared["refresh_token"] == "shared-rt"
    assert shared["last_status"] == STATUS_EXHAUSTED
    assert shared["last_error_code"] == 429


def test_codex_profile_stale_add_does_not_restore_removed_manual_entry(profile_env):
    from agent.credential_pool import PooledCredential, load_pool

    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "removed-manual",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "spent-at",
            "refresh_token": "spent-rt",
        }],
    }))
    stale = load_pool("openai-codex")

    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [],
    }))
    stale.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "new-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-new",
    }))

    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert [entry["id"] for entry in profile_data["credential_pool"]["openai-codex"]] == [
        "new-api-key",
    ]


def test_codex_profile_shared_remove_clears_root_state_atomically(profile_env):
    from agent.credential_pool import PooledCredential, load_pool

    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "shared-codex",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            }],
        },
    })
    remover = load_pool("openai-codex")
    stale = load_pool("openai-codex")

    assert remover.remove_index(1) is not None
    stale.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "profile-api-key",
        "source": "manual:api_key",
        "auth_type": "api_key",
        "access_token": "sk-profile",
    }))

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert "openai-codex" not in global_data.get("providers", {})
    assert global_data["credential_pool"]["openai-codex"] == []
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert [entry["id"] for entry in profile_data["credential_pool"]["openai-codex"]] == [
        "profile-api-key",
    ]


def test_codex_profile_shared_remove_preserves_newer_manual_entries(profile_env):
    from agent.credential_pool import load_pool

    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "shared-codex",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            }],
        },
    })
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "manual-codex",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "manual-old-at",
            "refresh_token": "manual-old-rt",
        }],
    }))
    stale = load_pool("openai-codex")

    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openai-codex": [{
            "id": "manual-codex",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": "manual-new-at",
            "refresh_token": "manual-new-rt",
        }, {
            "id": "concurrent-api-key",
            "source": "manual:api_key",
            "auth_type": "api_key",
            "access_token": "sk-concurrent",
        }],
    }))
    shared_index = next(
        index
        for index, entry in enumerate(stale.entries(), start=1)
        if entry.source == "device_code"
    )
    assert stale.remove_index(shared_index) is not None

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert "openai-codex" not in global_data.get("providers", {})
    assert global_data["credential_pool"]["openai-codex"] == []
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    entries = {
        entry["id"]: entry
        for entry in profile_data["credential_pool"]["openai-codex"]
    }
    assert entries["manual-codex"]["refresh_token"] == "manual-new-rt"
    assert entries["concurrent-api-key"]["access_token"] == "sk-concurrent"


def test_codex_profile_round_robin_shared_order_is_profile_local(profile_env, monkeypatch):
    from agent.credential_pool import STRATEGY_ROUND_ROBIN, load_pool
    from hermes_cli.auth import CODEX_REFRESH_OWNER

    sibling = profile_env["global"] / "profiles" / "sibling"
    sibling.mkdir()
    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
                "refresh_owner": CODEX_REFRESH_OWNER,
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "shared-codex",
                "source": "device_code",
                "auth_type": "oauth",
                "priority": 0,
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            }],
        },
    })
    for profile, manual_id in (
        (profile_env["profile"], "coder-manual"),
        (sibling, "sibling-manual"),
    ):
        _write(profile / "auth.json", _make_auth_store(pool={
            "openai-codex": [{
                "id": manual_id,
                "source": "manual:api_key",
                "auth_type": "api_key",
                "priority": 1,
                "access_token": f"sk-{manual_id}",
            }],
        }))

    coder = load_pool("openai-codex")
    coder._strategy = STRATEGY_ROUND_ROBIN
    assert coder.select().id == "shared-codex"

    root_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert root_data["credential_pool"]["openai-codex"][0]["priority"] == 0
    coder_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert coder_data["credential_pool_shared_order"]["openai-codex"]["device_code"] == 1

    monkeypatch.setenv("HERMES_HOME", str(sibling))
    assert [entry.id for entry in load_pool("openai-codex").entries()] == [
        "shared-codex",
        "sibling-manual",
    ]

    monkeypatch.setenv("HERMES_HOME", str(profile_env["profile"]))
    assert [entry.id for entry in load_pool("openai-codex").entries()] == [
        "coder-manual",
        "shared-codex",
    ]


def test_clear_codex_auth_clears_matching_root_active_provider(profile_env):
    from hermes_cli.auth import clear_provider_auth

    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
            },
        },
    })
    _write(profile_env["profile"] / "auth.json", {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {},
    })

    assert clear_provider_auth() is True

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert global_data["active_provider"] is None
    assert "openai-codex" not in global_data["providers"]


def test_clear_codex_auth_clears_profile_entries_and_shared_root_state(profile_env):
    from hermes_cli.auth import clear_provider_auth, read_credential_pool

    _write(profile_env["global"] / "auth.json", {
        "version": 1,
        "active_provider": "nous",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "root-manual",
                "source": "manual:device_code",
                "auth_type": "oauth",
                "access_token": "root-manual-at",
                "refresh_token": "root-manual-rt",
            }, {
                "id": "shared-codex",
                "source": "device_code",
                "auth_type": "oauth",
                "access_token": "shared-at",
                "refresh_token": "shared-rt",
            }],
        },
    })
    _write(profile_env["profile"] / "auth.json", {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "stale-profile-at",
                    "refresh_token": "stale-profile-rt",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "profile-manual",
                "source": "manual:dashboard_device_code",
                "auth_type": "oauth",
                "access_token": "profile-manual-at",
                "refresh_token": "profile-manual-rt",
            }],
        },
    })

    assert clear_provider_auth() is True

    global_data = json.loads((profile_env["global"] / "auth.json").read_text())
    assert global_data["active_provider"] == "nous"
    assert "openai-codex" not in global_data["providers"]
    assert [e["id"] for e in global_data["credential_pool"]["openai-codex"]] == [
        "root-manual",
    ]
    profile_data = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert profile_data["active_provider"] is None
    assert "openai-codex" not in profile_data["providers"]
    assert "openai-codex" not in profile_data["credential_pool"]
    assert read_credential_pool("openai-codex") == []
