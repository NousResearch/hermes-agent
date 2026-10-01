"""A read-through profile must never materialize its own copy of a single-use refresh chain.

When a named profile has no local ``providers.<id>`` it resolves credentials through the
global-root fallback (``_load_provider_state_with_source``). If a rotation or a fresh login then
persists into the PROFILE store, two physical copies of one single-use OAuth grant exist for the
same account: the profile's copy goes stale the moment any peer rotates, and its next refresh
redeems an already-spent token. The Portal answers that with ``refresh_token_reused`` and revokes
the whole session chain — which the user sees as "logged out again", repeatedly, all day.

The guard: ``_persist_provider_state_to_store`` / ``_save_active_provider_state`` route a
single-use-refresh provider's write back to the root store it was read from, so the profile stays a
pure reader. A profile that already owns a local grant keeps rotating in place (unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SINGLE_USE = "nous"


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Global root + an active named profile, mirroring the real on-disk layout."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_dir = global_root / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    # Drop the module-level global-store memo so each test reads its own tmp root.
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None, raising=False)
    return {"global": global_root, "profile": profile_dir}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _grant(tag: str) -> dict:
    return {
        "access_token": f"access-{tag}",
        "refresh_token": f"refresh-{tag}",
        "client_id": "hermes-cli",
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
        "token_type": "Bearer",
        "scope": "inference:invoke",
        "expires_at": "2030-01-01T00:00:00+00:00",
    }


def _seed_root(env, tag: str = "root") -> None:
    _write(env["global"] / "auth.json", {"version": 1, "providers": {SINGLE_USE: _grant(tag)}})


def _seed_profile(env, providers: dict | None = None) -> None:
    _write(env["profile"] / "auth.json", {"version": 1, "providers": providers or {}})


def _profile_provider(env):
    payload = json.loads((env["profile"] / "auth.json").read_text())
    return (payload.get("providers") or {}).get(SINGLE_USE)


def _root_provider(env):
    payload = json.loads((env["global"] / "auth.json").read_text())
    return (payload.get("providers") or {}).get(SINGLE_USE)


def test_readthrough_profile_detected(profile_env):
    """The guard's predicate is True exactly for a profile with no local grant but a root grant."""
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    _seed_profile(profile_env)
    assert auth_mod._inherits_global_provider_state(
        SINGLE_USE, auth_mod._auth_file_path()
    ) is True


def test_profile_that_owns_a_local_grant_is_not_readthrough(profile_env):
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    _seed_profile(profile_env, {SINGLE_USE: _grant("local")})
    assert auth_mod._inherits_global_provider_state(
        SINGLE_USE, auth_mod._auth_file_path()
    ) is False


def test_classic_mode_root_store_is_not_readthrough(profile_env, monkeypatch):
    """A store that IS the global root is never 'read-through': it owns its own state.

    Exercised at the predicate level — pointing ``HERMES_HOME`` at the root itself trips the
    pytest seat belt in ``_auth_file_path()`` (which refuses to touch a store that looks like the
    real user's), so the path is supplied explicitly instead.
    """
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    root_path = profile_env["global"] / "auth.json"
    assert auth_mod._inherits_global_provider_state(SINGLE_USE, root_path) is False


def test_rotation_write_does_not_create_a_profile_copy(profile_env):
    """The core regression: a rotation for a read-through profile lands in the root store."""
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    _seed_profile(profile_env)

    auth_mod._persist_provider_state_to_store(
        SINGLE_USE, _grant("rotated"), auth_mod._auth_file_path(), set_active=False
    )

    assert _profile_provider(profile_env) is None, "profile must stay a pure reader"
    assert _root_provider(profile_env)["refresh_token"] == "refresh-rotated"


def test_login_in_readthrough_profile_lands_in_root(profile_env):
    """A fresh login (device code / billing step-up) must not split the grant."""
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    _seed_profile(profile_env)

    wrote = auth_mod._save_active_provider_state(SINGLE_USE, _grant("fresh"))

    assert Path(wrote).resolve() == (profile_env["global"] / "auth.json").resolve()
    assert _profile_provider(profile_env) is None, "login must not materialize a profile copy"
    assert _root_provider(profile_env)["refresh_token"] == "refresh-fresh"


def test_profile_owning_a_grant_still_rotates_in_place(profile_env):
    """A profile with its own grant keeps its existing behaviour (no silent relocation)."""
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env, tag="root")
    _seed_profile(profile_env, {SINGLE_USE: _grant("local")})

    wrote = auth_mod._save_active_provider_state(SINGLE_USE, _grant("rotated"))

    assert Path(wrote).resolve() == (profile_env["profile"] / "auth.json").resolve()
    assert _profile_provider(profile_env)["refresh_token"] == "refresh-rotated"


def test_non_single_use_provider_is_untouched(profile_env):
    """Only single-use-refresh providers get the read-through reroute."""
    import hermes_cli.auth as auth_mod

    _seed_root(profile_env)
    _seed_profile(profile_env)

    wrote = auth_mod._save_active_provider_state("openrouter", {"api_key": "sk-x"})

    assert Path(wrote).resolve() == (profile_env["profile"] / "auth.json").resolve()
    payload = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert (payload.get("providers") or {}).get("openrouter") == {"api_key": "sk-x"}