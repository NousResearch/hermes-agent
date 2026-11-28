"""Shared and profile-owned OAuth pool refresh transactions."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from agent import anthropic_credentials, credential_pool
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli import auth


@pytest.fixture
def stores(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile / "auth.json", root / "auth.json"


def _write_rows(path, entry, *, legacy=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = entry.to_dict()
    if legacy:
        row.pop("id")
    path.write_text(json.dumps({
        "version": 1, "credential_pool": {entry.provider: [row]},
    }), encoding="utf-8")


def _entry(provider):
    return PooledCredential.from_dict(provider, {
        "id": "account", "source": "manual:device_code", "auth_type": "oauth",
        "access_token": "access-original", "refresh_token": "refresh-original",
    })


def _write_singleton(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "providers": {entry.provider: {"tokens": {
        "access_token": entry.access_token, "refresh_token": entry.refresh_token,
    }}}}), encoding="utf-8")


@pytest.mark.parametrize("borrowed", [True, False], ids=["borrowed", "owned"])
@pytest.mark.parametrize("deferred", [False, True], ids=["explicit", "deferred"])
def test_pool_refresh_locks_the_owning_store(stores, monkeypatch, borrowed, deferred):
    profile, root = stores
    store = root if borrowed else profile
    entry = _entry("xai-oauth")
    _write_rows(store, entry)
    locks_during_refresh = []
    pool = load_pool(entry.provider)
    store_lock = auth._auth_store_lock

    @contextmanager
    def ordered_store_lock(*args, **kwargs):
        assert pool._lock._is_owned() is not deferred
        if kwargs.get("target_path") == root:
            assert auth._auth_lock_holder_for(profile).depth > 0
        with store_lock(*args, **kwargs):
            yield

    def refresh(_access, _refresh):
        file_locks = tuple(
            bool(getattr(auth._auth_lock_holder_for(path), "depth", 0)) for path in stores
        )
        locks_during_refresh.append((pool._lock._is_owned(), *file_locks))
        return {"access_token": "access-renewed", "refresh_token": "refresh-renewed"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)
    monkeypatch.setattr(credential_pool, "_auth_store_lock", ordered_store_lock)
    if deferred:
        pool._refresh_pending_entries(pool.entries())
        refreshed, = pool.entries()
    else:
        refreshed = pool.try_refresh_matching(credential_id=entry.id)

    assert locks_during_refresh == [(not deferred, True, borrowed)]
    assert refreshed.refresh_token == "refresh-renewed"
    saved = json.loads(store.read_text(encoding="utf-8"))["credential_pool"][entry.provider][0]
    assert saved["refresh_token"] == refreshed.refresh_token
    assert not (profile if borrowed else root).exists()


def test_codex_pool_adopts_the_current_persisted_row(stores, monkeypatch):
    profile, root = stores
    entry = _entry("openai-codex")
    _write_rows(root, entry)
    pool = load_pool(entry.provider)
    current = replace(entry, access_token="access-current", refresh_token="refresh-current")
    _write_rows(root, current)

    def refresh(*_args):
        pytest.fail("a usable updated pool credential needs no refresh")

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    refreshed = pool.try_refresh_matching(api_key_hint=entry.access_token)

    assert refreshed.access_token == current.access_token
    assert refreshed.refresh_token == current.refresh_token
    assert not profile.exists()


def test_codex_pool_keeps_a_newer_in_memory_generation(stores, monkeypatch):
    _, root = stores
    stored = replace(_entry("openai-codex"), last_refresh="2026-01-01T00:00:00Z")
    current = replace(
        stored, access_token="access-current", refresh_token="refresh-current",
        last_refresh="2026-01-02T00:00:00Z",
    )
    def refresh(access, refresh_token):
        assert (access, refresh_token) == (current.access_token, current.refresh_token)
        return {"access_token": "access-renewed", "refresh_token": "refresh-renewed"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    _write_rows(root, current)
    pool = load_pool(current.provider)
    _write_rows(root, stored)

    assert pool.try_refresh_matching(credential_id=current.id).refresh_token == "refresh-renewed"


@pytest.mark.parametrize("seed", [_write_rows, _write_singleton], ids=["pool-row", "singleton"])
def test_profile_login_discards_a_cached_root_selection(stores, monkeypatch, seed):
    profile, root = stores
    entry = _entry("openai-codex")
    seed(root, entry)
    pool = load_pool(entry.provider)
    cached, = pool.entries()
    local = replace(entry, id="profile-account", access_token="local-access", refresh_token="local-refresh")
    seed(profile, local)
    original = [path.read_bytes() for path in stores]

    def refresh(*_args):
        pytest.fail("the profile login replaced this selection")

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)

    assert pool.try_refresh_matching(credential_id=cached.id) is None
    assert pool.entries() == []
    assert [path.read_bytes() for path in stores] == original


def test_reloaded_pool_keeps_the_profile_login_separate(stores, monkeypatch):
    profile, root = stores
    entry = replace(_entry("openai-codex"), source="device_code")
    _write_rows(root, entry)
    local = replace(entry, access_token="local-access", refresh_token="local-refresh")
    _write_singleton(profile, local)
    root_before = root.read_bytes()

    def refresh(access, refresh_token):
        assert (access, refresh_token) == (local.access_token, local.refresh_token)
        return {"access_token": "local-renewed", "refresh_token": "local-next"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    pool = load_pool(entry.provider)
    refreshed = pool.try_refresh_matching(api_key_hint=local.access_token)

    assert refreshed.refresh_token == "local-next"
    assert root.read_bytes() == root_before
    assert auth._load_auth_store()["providers"][entry.provider]["tokens"]["refresh_token"] == "local-next"


@pytest.mark.parametrize("seed", [_write_rows, _write_singleton], ids=["pool-row", "singleton"])
def test_cached_borrowed_selection_is_dropped_after_revocation(stores, monkeypatch, seed):
    profile, root = stores
    entry = _entry("openai-codex")
    seed(root, entry)
    pool = load_pool(entry.provider)
    cached, = pool.entries()
    root.write_text('{"version": 1, "providers": {}}', encoding="utf-8")

    def refresh(*_args):
        pytest.fail("removed grants cannot be refreshed")

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)

    assert pool.try_refresh_matching(credential_id=cached.id) is None
    assert pool.entries() == []
    assert not profile.exists()


def test_stale_borrowed_selection_leaves_a_replacement_intact(stores, monkeypatch):
    profile, root = stores
    entry = _entry("openai-codex")
    _write_rows(root, entry)
    pool = load_pool(entry.provider)
    cached, = pool.entries()
    replacement = replace(cached, access_token="replacement-access", refresh_token="replacement-refresh")
    _write_rows(profile, replacement)
    pool._replace_entry(cached, replacement)
    original_lock = pool._lock

    @contextmanager
    def discard_lock():
        assert all(not getattr(auth._auth_lock_holder_for(path), "depth", 0) for path in stores)
        with original_lock:
            yield

    with monkeypatch.context() as patch:
        patch.setattr(pool, "_lock", discard_lock())
        assert pool._refresh_entry(cached, force=True) is None
    assert pool.entries() == [replacement]


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
@pytest.mark.parametrize("already_rotated", [True, False], ids=["adopt", "write-back"])
def test_borrowed_singleton_stays_on_its_root(stores, monkeypatch, provider, already_rotated):
    profile, root = stores
    original = _entry(provider)
    _write_singleton(root, original)
    pool = load_pool(provider)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps({"version": 1, "providers": {provider: {"tokens": {}}}}), encoding="utf-8")
    profile_before = profile.read_bytes()
    renewed = replace(original, access_token="root-current", refresh_token="root-next")
    if already_rotated:
        _write_singleton(root, renewed)

    def refresh(access, refresh_token):
        if already_rotated:
            pytest.fail("the current root grant needs no refresh")
        assert (access, refresh_token) == (original.access_token, original.refresh_token)
        return {"access_token": renewed.access_token, "refresh_token": renewed.refresh_token}

    name = "refresh_codex_oauth_pure" if provider == "openai-codex" else "refresh_xai_oauth_pure"
    monkeypatch.setattr(auth, name, refresh)

    result = pool.try_refresh_matching(api_key_hint=original.access_token)

    assert result.refresh_token == renewed.refresh_token
    assert profile.read_bytes() == profile_before
    assert auth._load_auth_store(root)["providers"][provider]["tokens"]["refresh_token"] == renewed.refresh_token


def test_legacy_borrowed_pool_row_is_matched_and_persisted(stores, monkeypatch):
    profile, root = stores
    original = _entry("xai-oauth")
    _write_rows(root, original, legacy=True)
    pool = load_pool(original.provider)
    cached, = pool.entries()

    assert load_pool(original.provider).entries()[0].id == cached.id

    def refresh(access, refresh_token):
        assert (access, refresh_token) == (original.access_token, original.refresh_token)
        return {"access_token": "access-renewed", "refresh_token": "refresh-renewed"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)

    result = pool.try_refresh_matching(credential_id=cached.id)
    saved, = auth._load_auth_store(root)["credential_pool"][original.provider]

    assert saved["id"] == cached.id == result.id
    assert saved["refresh_token"] == "refresh-renewed"
    assert not profile.exists()


@pytest.mark.parametrize("first", ["named", "default"])
def test_default_and_named_profile_adopt_the_same_legacy_grant(stores, monkeypatch, first):
    profile, root = stores
    original = _entry("xai-oauth")
    _write_rows(root, original, legacy=True)
    homes = {"default": root.parent, "named": profile.parent}
    pools = {}
    for name, home in homes.items():
        monkeypatch.setenv("HERMES_HOME", str(home))
        pools[name] = load_pool(original.provider)
    cached = {name: pool.entries()[0] for name, pool in pools.items()}
    assert "id" not in auth._load_auth_store(root)["credential_pool"][original.provider][0]

    def refresh(access, refresh_token):
        assert (access, refresh_token) == (original.access_token, original.refresh_token)
        return {"access_token": "access-renewed", "refresh_token": "refresh-renewed"}

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)
    monkeypatch.setenv("HERMES_HOME", str(homes[first]))
    renewed = pools[first].try_refresh_matching(credential_id=cached[first].id)

    def unexpected_refresh(*_args):
        pytest.fail("the second reader should use the credential already in the store")

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", unexpected_refresh)
    second, = homes.keys() - {first}
    monkeypatch.setenv("HERMES_HOME", str(homes[second]))
    adopted = pools[second].try_refresh_matching(credential_id=cached[second].id)
    saved, = auth._load_auth_store(root)["credential_pool"][original.provider]
    assert adopted.id == renewed.id == saved["id"] == cached["default"].id == cached["named"].id
    assert adopted.access_token == renewed.access_token == saved["access_token"] == "access-renewed"
    assert adopted.refresh_token == renewed.refresh_token == saved["refresh_token"] == "refresh-renewed"
    assert not profile.exists()


def test_owned_legacy_pool_does_not_refresh_a_removed_grant(stores, monkeypatch):
    _, root = stores
    original = _entry("xai-oauth")
    _write_rows(root, original, legacy=True)
    monkeypatch.setenv("HERMES_HOME", str(root.parent))
    pool = load_pool(original.provider)
    cached, = pool.entries()
    root.write_text('{"version": 1, "providers": {}}', encoding="utf-8")

    def refresh(*_args):
        pytest.fail("a cached grant missing from the owning store must not be refreshed")

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", refresh)
    assert pool.try_refresh_matching(credential_id=cached.id) is None
    assert pool.entries() == []


def test_cached_legacy_row_survives_runtime_refresh(stores, monkeypatch):
    profile, root = stores
    original = _entry("xai-oauth")
    row = original.to_dict()
    row.pop("id")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(json.dumps({
        "version": 1,
        "providers": {original.provider: {"discovery": {"token_endpoint": "https://example.invalid/token"}}},
        "credential_pool": {original.provider: [row]},
    }), encoding="utf-8")
    pool = load_pool(original.provider)
    cached, = pool.entries()
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", lambda *_args, **_kwargs: {
        "access_token": "access-renewed", "refresh_token": "refresh-renewed", "last_refresh": "2026-01-01T00:00:00Z",
    })

    auth.resolve_xai_oauth_runtime_credentials(force_refresh=True)

    def unexpected_refresh(*_args, **_kwargs):
        pytest.fail("the cached pool should adopt the saved credential")

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", unexpected_refresh)
    adopted = pool.try_refresh_matching(credential_id=cached.id)
    saved, = auth._load_auth_store(root)["credential_pool"][original.provider]

    assert adopted.id == saved["id"] == cached.id
    assert adopted.refresh_token == "refresh-renewed"
    assert not profile.exists()


def test_claude_code_refresh_uses_its_own_lock(stores, tmp_path, monkeypatch):
    profile, root = stores
    entry = replace(_entry("anthropic"), source="claude_code")
    _write_rows(root, _entry("anthropic"))

    claude_file = tmp_path / "claude" / ".credentials.json"
    monkeypatch.setattr(anthropic_credentials, "claude_code_credentials_path", lambda: claude_file)
    monkeypatch.setattr(anthropic_credentials, "read_claude_code_credentials", lambda: None)
    observed = []

    def refresh(_pool, current):
        observed.append(tuple(
            bool(getattr(auth._auth_lock_holder_for(path), "depth", 0))
            for path in (profile, claude_file, root)
        ))
        return replace(current, access_token="access-renewed", refresh_token="refresh-renewed")

    monkeypatch.setattr(CredentialPool, "_refresh_anthropic", refresh)
    pool = CredentialPool(entry.provider, [entry])

    assert pool.try_refresh_matching(credential_id=entry.id).refresh_token == "refresh-renewed"
    assert observed == [(True, True, False)]
