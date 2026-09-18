"""Every profile owns its credentials.

A named profile (``HERMES_HOME`` under ``profiles/<name>``) resolves provider state and the
credential pool from ITS OWN ``auth.json`` only. The root ``~/.hermes/auth.json`` is never a
read fallback and never a write-through target (#111724): an isolated service profile must fail
closed instead of acting — and rotating tokens — as the owner. Writes stay scoped to the profile.
"""

from __future__ import annotations

import json
import time
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
    """Global root + an active named profile under Path.home()/.hermes/profiles/coder.

    * Path.home() -> tmp_path
    * Global root -> tmp_path/.hermes            (has its own auth.json fixture)
    * Profile     -> tmp_path/.hermes/profiles/coder   (active, HERMES_HOME points here)
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


_ROOT_STORE = _make_auth_store(
    providers={
        "nous": {"access_token": "nous-root", "refresh_token": "rt-root"},
        "xai-oauth": {"auth_mode": "oauth_pkce",
                      "tokens": {"access_token": "xai-root-access", "refresh_token": "xai-root-refresh"}},
    },
    pool={
        "openrouter": [{"id": "glob-1", "label": "root", "auth_type": "api_key", "priority": 0,
                        "source": "manual", "access_token": "sk-root"}],
        "openai-codex": [{"id": "glob-codex", "auth_type": "oauth", "priority": 0,
                          "source": "manual:device_code", "access_token": "root-codex-access",
                          "refresh_token": "root-codex-refresh"}],
    },
)


def test_named_profile_never_reads_the_root_store(profile_env):
    """Provider state, pool slices and the whole-pool read all stop at the profile's own auth.json;
    the profile's own rows are what it sees, and a root-only provider resolves to nothing."""
    from hermes_cli.auth import get_provider_auth_state, read_credential_pool, resolve_codex_runtime_credentials
    from hermes_cli.auth_xai import _read_xai_oauth_tokens
    from hermes_cli.auth import AuthError

    _write(profile_env["global"] / "auth.json", _ROOT_STORE)
    _write(profile_env["profile"] / "auth.json", _make_auth_store(
        providers={},
        pool={"openrouter": [{"id": "prof-1", "auth_type": "api_key", "priority": 0,
                              "source": "manual", "access_token": "sk-profile"}]},
    ))

    assert get_provider_auth_state("nous") is None
    assert read_credential_pool("openai-codex") == []
    assert [e["id"] for e in read_credential_pool("openrouter")] == ["prof-1"]
    assert set(read_credential_pool()) == {"openrouter"}
    with pytest.raises(AuthError):
        _read_xai_oauth_tokens()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(refresh_if_expiring=False)

    # Control: a home that owns the store (single-profile layout) resolves its credentials.
    own = profile_env["profile"].parent / "owner"
    own.mkdir()
    _write(own / "auth.json", _ROOT_STORE)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(own)
    try:
        assert get_provider_auth_state("nous")["access_token"] == "nous-root"
        assert read_credential_pool("openrouter")[0]["id"] == "glob-1"
    finally:
        reset_hermes_home_override(token)


def test_profile_refresh_never_writes_through_to_the_root_store(profile_env):
    """A token save inside a profile lands in the profile's auth.json; the root store is
    byte-identical afterwards (xAI OAuth save, Codex save, pool write, cooldown clear)."""
    from hermes_cli.auth import write_credential_pool
    from hermes_cli.auth_codex import _save_codex_tokens, clear_codex_pool_quota_cooldowns
    from hermes_cli.auth_xai import _save_xai_oauth_tokens

    root_file = profile_env["global"] / "auth.json"
    root_store = json.loads(json.dumps(_ROOT_STORE))
    root_store["credential_pool"]["openai-codex"][0].update(
        last_status="exhausted", last_error_reason="rate_limit", last_error_reset_at=4_102_444_800)
    _write(root_file, root_store)
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}, pool={}))
    before = root_file.read_bytes()

    _save_xai_oauth_tokens({"access_token": "xai-new", "refresh_token": "xai-new-rt"}, set_active=False)
    _save_codex_tokens({"access_token": "codex-new", "refresh_token": "codex-new-rt"})
    write_credential_pool("openrouter", [{"id": "prof-new", "auth_type": "api_key", "priority": 0,
                                          "source": "manual", "access_token": "sk-profile-new"}])
    assert clear_codex_pool_quota_cooldowns() == 0  # the root's exhausted row is not ours to clear

    assert root_file.read_bytes() == before
    profile_store = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert profile_store["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "xai-new-rt"
    assert profile_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "codex-new-rt"
    assert [e["id"] for e in profile_store["credential_pool"]["openrouter"]] == ["prof-new"]


def test_malformed_or_missing_root_store_never_breaks_a_profile_read(profile_env):
    from hermes_cli.auth import read_credential_pool

    (profile_env["global"] / "auth.json").write_text("{not valid json")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{"id": "prof-1", "auth_type": "api_key", "priority": 0,
                        "source": "manual", "access_token": "sk-profile"}]}))
    assert read_credential_pool("openrouter")[0]["id"] == "prof-1"


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
