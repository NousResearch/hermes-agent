"""Legacy mutating preparation remains transactional without weakening typed pairs."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import providers
from agent.credential_pool import CredentialPool, PooledCredential
from agent.credential_pool_plugin import RuntimeOverride, durable_runtime_key
from providers.base import PoolRuntimeCredential, ProviderProfile

SOURCE = "https://api.individual.example"
FOREIGN = "https://api.enterprise.ghe.example"
GRANT = "durable-grant-0001"


class HookProfile(ProviderProfile):
    def prepare_pool_entry(self, entry, api_key):
        return self.prepare(entry, api_key)

    def refresh_pool_runtime_credentials(self, entry, *, reason):
        return self.renew(entry)


@pytest.fixture
def setup(monkeypatch):
    profile = HookProfile(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE,))
    row = PooledCredential(provider="pairprov", id="row-a", label="row-a", auth_type="api_key",
                           priority=0, source="manual", access_token=GRANT, base_url=SOURCE)
    pool = CredentialPool("pairprov", [row])
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile)
    return pool, row, profile


def _legacy(entry, api_key):
    entry.access_token = "tid=exchanged;exp=1"
    entry.base_url = FOREIGN
    return entry.access_token


def test_legacy_mutating_hook_adopts_key_and_endpoint_without_typed_allowlist(setup):
    pool, row, profile = setup
    object.__setattr__(profile, "prepare", _legacy)
    assert pool.prepare_runtime_entry(row, source_base_url=SOURCE) is row
    assert (row.runtime_api_key, row.runtime_base_url) == ("tid=exchanged;exp=1", FOREIGN)
    assert (row.access_token, row.base_url) == ("tid=exchanged;exp=1", FOREIGN)


@pytest.mark.parametrize("concurrent", ["grant", "remove", "renew"])
def test_legacy_mutations_cannot_mask_concurrent_updates(setup, concurrent):
    pool, row, profile = setup
    entered, release = Event(), Event()

    def prepare(copy, api_key):
        _legacy(copy, api_key)
        entered.set()
        assert release.wait(10)
        return copy.access_token

    object.__setattr__(profile, "prepare", prepare)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pool.prepare_runtime_entry, row)
        try:
            assert entered.wait(10)
            # Uncommitted legacy mutations must not leak before the hook returns.
            assert (row.access_token, row.base_url) == (GRANT, SOURCE)
            with pool._lock:
                if concurrent == "grant":
                    row.access_token = "new-grant"
                elif concurrent == "remove":
                    pool._entries = []
                else:
                    object.__setattr__(row, "_runtime_override", RuntimeOverride(GRANT, GRANT, "winner", SOURCE))
        finally:
            release.set()
        result = future.result(10)
    assert result is (row if concurrent == "renew" else None)
    assert row.base_url == SOURCE
    assert row.runtime_api_key == {"grant": "new-grant", "remove": GRANT, "renew": "winner"}[concurrent]


@pytest.mark.parametrize("kind", ["prepare", "refresh"])
def test_typed_hook_row_mutations_are_not_committed(setup, kind):
    pool, row, profile = setup

    def mutate(copy, *args):
        copy.access_token = "must-not-persist"
        copy.base_url = FOREIGN
        copy.extra["nested"] = "must-not-persist"
        return PoolRuntimeCredential("typed-bearer", SOURCE)

    object.__setattr__(profile, "prepare", mutate)
    object.__setattr__(profile, "renew", mutate)
    if kind == "prepare":
        result = pool.prepare_runtime_entry(row)
    else:
        result = pool.try_refresh_runtime_matching(credential_id=row.id, api_key_hint=GRANT)
    assert result is row
    assert (row.access_token, row.base_url, row.extra) == (GRANT, SOURCE, {})
    assert row.runtime_api_key == "typed-bearer"


@pytest.mark.parametrize("same_bearer", [False, True])
def test_late_runtime_renewal_preserves_newer_generation(setup, same_bearer):
    pool, row, profile = setup
    entered, release = Event(), Event()
    initial = RuntimeOverride(GRANT, durable_runtime_key(row), "initial-bearer", SOURCE)
    object.__setattr__(row, "_runtime_override", initial)

    calls = []

    def renew(copy):
        calls.append(copy.id)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
            return PoolRuntimeCredential("late-stale-bearer", SOURCE)
        return PoolRuntimeCredential("initial-bearer" if same_bearer else "winner", SOURCE)

    object.__setattr__(profile, "renew", renew)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pool.try_refresh_runtime_matching, credential_id=row.id,
                                 api_key_hint="initial-bearer")
        try:
            assert entered.wait(10)
            assert pool.try_refresh_runtime_matching(credential_id=row.id, api_key_hint="initial-bearer") is row
            winner = row.__dict__["_runtime_override"]
        finally:
            release.set()
        assert future.result(10) is None
    assert row.__dict__["_runtime_override"] is winner
