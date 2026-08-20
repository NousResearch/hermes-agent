"""Tests for cross-profile auth fallback.

When ``HERMES_HOME`` points to a named profile, ``read_credential_pool()``
and ``get_provider_auth_state()`` fall back to the global-root
``auth.json`` per-provider when the profile has no entries for that
provider. Ordinary writes still target the profile; refreshes persist rotating
credentials back to the store that supplied them.

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


def _codex_auth_store(
    access_token: str,
    refresh_token: str,
    *,
    marker: str,
    active_provider: str = "openai-codex",
) -> dict:
    return {
        "version": 1,
        "active_provider": active_provider,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-08-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": f"{marker}-codex",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
            ],
        },
        "marker": marker,
    }


# ---------------------------------------------------------------------------
# read_credential_pool — provider-slice reads
# ---------------------------------------------------------------------------








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


def test_provider_auth_state_returns_none_when_neither_has_it(profile_env):
    from hermes_cli.auth import get_provider_auth_state

    _write(profile_env["global"] / "auth.json", _make_auth_store(providers={}))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}))

    assert get_provider_auth_state("nous") is None


def test_codex_refresh_persists_to_root_fallback_without_profile_shadow(
    profile_env, monkeypatch
):
    import hermes_cli.auth as auth

    root_path = profile_env["global"] / "auth.json"
    profile_path = profile_env["profile"] / "auth.json"
    _write(
        root_path,
        _codex_auth_store(
            "root-old-at",
            "root-old-rt",
            marker="root",
            active_provider="openrouter",
        ),
    )
    _write(
        profile_path,
        {
            **_make_auth_store(
                pool={"anthropic": [{"id": "profile-unrelated"}]},
                providers={"anthropic": {"api_key": "profile-unrelated"}},
            ),
            "active_provider": "anthropic",
        },
    )
    profile_before = profile_path.read_bytes()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        assert access_token == "root-old-at"
        assert refresh_token == "root-old-rt"
        return {
            "access_token": "root-new-at",
            "refresh_token": "root-new-rt",
        }

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", fake_refresh)

    resolved = auth.resolve_codex_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "root-new-at"
    root = json.loads(root_path.read_text())
    assert root["providers"]["openai-codex"]["tokens"] == {
        "access_token": "root-new-at",
        "refresh_token": "root-new-rt",
    }
    assert root["credential_pool"]["openai-codex"][0]["access_token"] == "root-new-at"
    assert root["credential_pool"]["openai-codex"][0]["refresh_token"] == "root-new-rt"
    assert root["active_provider"] == "openrouter"
    assert root["marker"] == "root"
    assert profile_path.read_bytes() == profile_before


def test_codex_refresh_keeps_profile_owned_auth_local(profile_env, monkeypatch):
    import hermes_cli.auth as auth

    root_path = profile_env["global"] / "auth.json"
    profile_path = profile_env["profile"] / "auth.json"
    _write(root_path, _codex_auth_store("root-at", "root-rt", marker="root"))
    _write(
        profile_path,
        _codex_auth_store(
            "profile-old-at",
            "profile-old-rt",
            marker="profile",
            active_provider="openrouter",
        ),
    )
    root_before = root_path.read_bytes()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        assert access_token == "profile-old-at"
        assert refresh_token == "profile-old-rt"
        return {
            "access_token": "profile-new-at",
            "refresh_token": "profile-new-rt",
        }

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", fake_refresh)

    resolved = auth.resolve_codex_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "profile-new-at"
    profile = json.loads(profile_path.read_text())
    assert profile["providers"]["openai-codex"]["tokens"] == {
        "access_token": "profile-new-at",
        "refresh_token": "profile-new-rt",
    }
    assert profile["credential_pool"]["openai-codex"][0]["access_token"] == "profile-new-at"
    assert profile["credential_pool"]["openai-codex"][0]["refresh_token"] == "profile-new-rt"
    assert profile["active_provider"] == "openrouter"
    assert profile["marker"] == "profile"
    assert root_path.read_bytes() == root_before


def test_codex_cli_recovery_persists_to_root_fallback_without_profile_shadow(
    profile_env, monkeypatch
):
    import hermes_cli.auth as auth

    root_path = profile_env["global"] / "auth.json"
    profile_path = profile_env["profile"] / "auth.json"
    _write(
        root_path,
        _codex_auth_store(
            "root-old-at",
            "root-rejected-rt",
            marker="root",
            active_provider="openrouter",
        ),
    )
    _write(
        profile_path,
        {
            **_make_auth_store(pool={}, providers={}),
            "active_provider": "anthropic",
            "profile_marker": "preserve",
        },
    )
    profile_before = profile_path.read_bytes()

    def reject_refresh(*_args, **_kwargs):
        raise auth.AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", reject_refresh)
    monkeypatch.setattr(
        auth,
        "_import_codex_cli_tokens",
        lambda: {
            "access_token": "recovered-at",
            "refresh_token": "recovered-rt",
        },
    )

    resolved = auth.resolve_codex_runtime_credentials(force_refresh=True)

    assert resolved["api_key"] == "recovered-at"
    root = json.loads(root_path.read_text())
    assert root["providers"]["openai-codex"]["tokens"] == {
        "access_token": "recovered-at",
        "refresh_token": "recovered-rt",
    }
    assert root["credential_pool"]["openai-codex"][0]["access_token"] == "recovered-at"
    assert root["credential_pool"]["openai-codex"][0]["refresh_token"] == "recovered-rt"
    assert root["active_provider"] == "openrouter"
    assert root["marker"] == "root"
    assert profile_path.read_bytes() == profile_before


def test_codex_malformed_root_recovery_rechecks_concurrent_in_lock_repair(
    profile_env, monkeypatch
):
    """A fresh root repair must win over a stale pre-transaction read error."""
    import hermes_cli.auth as auth

    root_path = profile_env["global"] / "auth.json"
    profile_path = profile_env["profile"] / "auth.json"
    malformed = _codex_auth_store(
        "root-malformed-access",
        "root-malformed-refresh",
        marker="root",
        active_provider="openrouter",
    )
    malformed["providers"]["openai-codex"]["tokens"].pop("access_token")
    _write(root_path, malformed)
    _write(
        profile_path,
        {
            **_make_auth_store(
                pool={"anthropic": [{"id": "profile-unrelated"}]},
                providers={"anthropic": {"api_key": "profile-unrelated"}},
            ),
            "active_provider": "anthropic",
            "profile_marker": "preserve",
        },
    )
    profile_before = profile_path.read_bytes()

    repaired = _codex_auth_store(
        "root-repaired-access",
        "root-repaired-refresh",
        marker="root",
        active_provider="openrouter",
    )
    repaired["providers"]["anthropic"] = {
        "api_key": "root-unrelated-provider"
    }
    repaired["root_marker"] = {"preserve": True}
    real_transaction = auth._provider_state_transaction
    repair_bytes = []

    @contextmanager
    def repair_before_source_transaction(*args, **kwargs):
        # This wrapper is reached only after the first malformed read raised.
        # Install the competing writer's valid chain before this reader gets
        # the real active-profile -> source-store transaction.
        _write(root_path, repaired)
        repair_bytes.append(root_path.read_bytes())
        with real_transaction(*args, **kwargs) as transaction:
            yield transaction

    def fail_if_cli_imported():
        pytest.fail("CLI import must not overwrite a valid concurrent root repair")

    monkeypatch.setattr(
        auth, "_provider_state_transaction", repair_before_source_transaction
    )
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", fail_if_cli_imported)

    resolved = auth.resolve_codex_runtime_credentials()

    assert resolved["api_key"] == "root-repaired-access"
    assert repair_bytes
    assert root_path.read_bytes() == repair_bytes[-1]
    assert profile_path.read_bytes() == profile_before


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
