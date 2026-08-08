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
import time
from contextlib import contextmanager
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


def test_profile_can_disable_global_credential_pool_fallback(profile_env):
    """Security-sensitive profiles can refuse every root pool credential."""
    from hermes_cli.auth import read_credential_pool

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "xai-oauth": [{
            "id": "global-xai",
            "label": "global-xai",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": "global-access",
            "refresh_token": "global-refresh",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={}))
    (profile_env["profile"] / "config.yaml").write_text(
        "auth:\n  global_fallback: false\n"
    )

    assert read_credential_pool("xai-oauth") == []
    assert read_credential_pool(None) == {}


def test_managed_config_can_disable_global_credential_pool_fallback(
    profile_env, monkeypatch
):
    """An administrator-pinned false value overrides the profile config."""
    from hermes_cli import managed_scope
    from hermes_cli.auth import read_credential_pool
    import hermes_cli.config as config_module

    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "xai-oauth": [{
            "id": "global-xai",
            "label": "global-xai",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual",
            "access_token": "global-access",
            "refresh_token": "global-refresh",
        }],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={}))
    (profile_env["profile"] / "config.yaml").write_text(
        "auth:\n  global_fallback: true\n"
    )
    managed_dir = profile_env["global"] / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "auth:\n  global_fallback: false\n"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    config_module._LOAD_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    assert read_credential_pool("xai-oauth") == []


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


def test_profile_can_disable_global_provider_state_fallback(profile_env):
    """The same opt-out covers singleton provider state, not only pools."""
    from hermes_cli.auth import get_provider_auth_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={
        "nous": {"access_token": "nous-global", "refresh_token": "rt-global"},
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))
    (profile_env["profile"] / "config.yaml").write_text(
        "auth:\n  global_fallback: false\n"
    )

    assert get_provider_auth_state("nous") is None


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






# ---------------------------------------------------------------------------
# Classic mode — no fallback path should ever trigger
# ---------------------------------------------------------------------------




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




def test_auth_lock_reentrancy_is_scoped_after_profile_context_switch(profile_env):
    """Changing profile context cannot inherit another store's lock depth."""
    import hermes_cli.auth as auth
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_b = profile_env["global"] / "profiles" / "reviewer"
    profile_b.mkdir(parents=True)
    profile_b_lock = profile_b / "auth.lock"

    with auth._auth_store_lock():
        holder_a = auth._auth_lock_holder_for(profile_env["profile"] / "auth.json")
        assert getattr(holder_a, "depth", 0) == 1

        token = set_hermes_home_override(profile_b)
        try:
            holder_b = auth._auth_lock_holder_for(profile_b / "auth.json")
            assert holder_b is not holder_a
            assert getattr(holder_b, "depth", 0) == 0
            assert not profile_b_lock.exists()

            with auth._auth_store_lock():
                assert profile_b_lock.exists()
                assert getattr(holder_b, "depth", 0) == 1
        finally:
            reset_hermes_home_override(token)

    assert getattr(holder_a, "depth", 0) == 0


# ---------------------------------------------------------------------------
# write_credential_pool — stale-snapshot cooldown merge
# ---------------------------------------------------------------------------


@pytest.fixture()
def classic_env(tmp_path, monkeypatch):
    """Classic single-root layout (HERMES_HOME != ~/.hermes, no profiles)."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    hermes_home = tmp_path / "classic"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _pool_entry(**overrides) -> dict:
    entry = {
        "id": "cred-x",
        "label": "key-x",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-x",
    }
    entry.update(overrides)
    return entry




def test_write_pool_never_merges_cooldown_onto_reauthed_entry(classic_env):
    """A token change means re-auth: the old cooldown must never carry over.

    A fresh login intentionally clears the entry's status; resurrecting the
    stale cooldown onto the new credentials would bench a just-authorized key.
    """
    from hermes_cli.auth import write_credential_pool

    _write(classic_env / "auth.json", _make_auth_store(pool={
        "openrouter": [_pool_entry(
            access_token="sk-old",
            last_status="exhausted",
            last_status_at=time.time() - 60,  # newer AND unexpired
            last_error_code=429,
        )],
    }))

    # Same entry id, freshly re-authed with a new token and cleared status.
    write_credential_pool("openrouter", [_pool_entry(access_token="sk-new")])

    data = json.loads((classic_env / "auth.json").read_text())
    persisted = data["credential_pool"]["openrouter"][0]
    assert persisted["access_token"] == "sk-new"
    assert persisted.get("last_status") != "exhausted"
    assert persisted.get("last_error_code") is None
