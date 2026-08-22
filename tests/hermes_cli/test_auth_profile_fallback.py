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


# ---------------------------------------------------------------------------
# read_credential_pool — read-time kimi base_url normalization (#5908)
# ---------------------------------------------------------------------------

KIMI_STALE_DEFAULT_URL = "https://api.moonshot.ai/v1"
KIMI_CODE_URL = "https://api.kimi.com/coding"
KIMI_LEGACY_V1_URL = "https://api.kimi.com/coding/v1"


def _kimi_entry(**overrides) -> dict:
    entry = {
        "id": "kimi-1",
        "label": "old Kimi key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-kimi-test-key",
        "base_url": KIMI_STALE_DEFAULT_URL,
    }
    entry.update(overrides)
    return entry


def test_stale_moonshot_default_url_is_repaired_on_load(profile_env):
    """#5908: entries persisted before prefix resolution keep the Moonshot
    default; loading a kimi entry must re-resolve from the key prefix."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_legacy_coding_v1_url_is_repaired_on_load(profile_env):
    """The old documented workaround ('/coding/v1') double-appends /v1 in the
    Anthropic SDK; loading must repair it to the canonical /coding URL."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url=KIMI_LEGACY_V1_URL)],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_legacy_coding_v1_url_preserved_for_non_kimi_key(profile_env):
    """A non-kimi key stored with /coding/v1 is treated as an explicit custom
    endpoint (possible OpenAI-compat use) and must not be rewritten."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(
            access_token="sk-legacy-moonshot-key",
            base_url=KIMI_LEGACY_V1_URL,
        )],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_LEGACY_V1_URL


def test_stale_default_with_trailing_slash_is_repaired(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url="https://api.moonshot.ai/v1/")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_empty_env_override_treated_as_unset(profile_env, monkeypatch):
    monkeypatch.setenv("KIMI_BASE_URL", "")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_missing_base_url_is_filled_on_load(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url=None)],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_empty_base_url_string_is_filled_on_load(profile_env):
    """An empty-string base_url is treated like a missing one and filled."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url="")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_whitespace_only_base_url_is_filled_on_load(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url="   ")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_custom_nondefault_url_is_preserved_on_load(profile_env):
    """A stored proxy/custom endpoint is the user's explicit choice — the
    load-time repair must never clobber it (the #18694 rejection reason)."""
    custom = "https://kimi-proxy.example.com/v1"
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url=custom)],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == custom


def test_env_override_wins_over_stale_url_on_load(profile_env, monkeypatch):
    monkeypatch.setenv("KIMI_BASE_URL", "https://env-override.example/v1")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == "https://env-override.example/v1"


def test_env_override_wins_over_custom_url_on_load(profile_env, monkeypatch):
    monkeypatch.setenv("KIMI_BASE_URL", "https://env-override.example/v1")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(base_url="https://kimi-proxy.example.com/v1")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == "https://env-override.example/v1"


def test_legacy_moonshot_key_keeps_moonshot_default_on_load(profile_env):
    """Keys without the sk-kimi- prefix belong on the Moonshot default; the
    repair must not move them to api.kimi.com."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(access_token="sk-legacy-moonshot-key")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_STALE_DEFAULT_URL


def test_runtime_api_key_field_drives_normalization(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(access_token="", runtime_api_key="sk-kimi-runtime-key")],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_fully_empty_credential_entry_left_untouched(profile_env):
    """An entry with no credential at all has nothing to infer a URL from;
    the load-time repair must not even fill a missing base_url for it."""
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry(
            access_token="",
            runtime_api_key=None,
            api_key=None,
            base_url=None,
        )],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0].get("base_url") is None


def test_non_kimi_provider_entries_untouched(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "zai": [{
            "id": "zai-1",
            "auth_type": "api_key",
            "access_token": "sk-zai-key",
            "base_url": "https://api.z.ai/api/paas/v4",
        }],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("zai")
    assert entries[0]["base_url"] == "https://api.z.ai/api/paas/v4"


def test_global_fallback_entries_normalized_on_load(profile_env):
    """The read-only global fallback slice must be repaired the same way."""
    _write(profile_env["global"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
    }))
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{
            "id": "prof-1",
            "auth_type": "api_key",
            "access_token": "sk-profile",
        }],
    }))

    from hermes_cli.auth import read_credential_pool

    entries = read_credential_pool("kimi-coding")
    assert entries[0]["base_url"] == KIMI_CODE_URL


def test_whole_pool_read_normalizes_kimi_entries(profile_env):
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
        "openrouter": [{
            "id": "prof-1",
            "auth_type": "api_key",
            "access_token": "sk-profile",
        }],
    }))

    from hermes_cli.auth import read_credential_pool

    pool = read_credential_pool()
    assert pool["kimi-coding"][0]["base_url"] == KIMI_CODE_URL
    assert pool["openrouter"][0]["id"] == "prof-1"


def test_pool_to_runtime_uses_normalized_url(profile_env, monkeypatch):
    """Regression: the runtime entry constructed from the persisted pool must
    carry the repaired URL (read_credential_pool → PooledCredential)."""
    # Hermetic: no ambient kimi env vars that load_pool's seeding could pick up.
    for var in ("KIMI_API_KEY", "KIMI_CODING_API_KEY", "KIMI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "kimi-coding": [_kimi_entry()],
    }))

    from agent.credential_pool import load_pool

    pool = load_pool("kimi-coding")
    entry = pool.select()
    assert entry is not None
    assert entry.base_url == KIMI_CODE_URL
