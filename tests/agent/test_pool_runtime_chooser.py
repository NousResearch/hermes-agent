"""The shared pool chooser (``choose_pool_key`` / ``PoolChoice``) prepares typed runtime pairs through
the pool's own transaction, keeps the derived key with its derived endpoint, and leaves the
persisted pool holding only the durable grant."""

import json
from types import SimpleNamespace

import pytest

import providers
from agent import auxiliary_client as aux
from agent import auxiliary_pool_recovery as recovery
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli import runtime_provider as rp
from providers.base import PoolRuntimeCredential, ProviderProfile

PROVIDER = "pairprov"
SOURCE = "https://api.individual.example"
TARGET = "https://api.business.example"
GRANT = "durable-grant-0001"
DERIVED = "derived-bearer-0001"


class PairProfile(ProviderProfile):
    def __init__(self, result=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "result", result)

    def prepare_pool_entry(self, entry, api_key):
        self.calls.append(api_key)
        return self.result if self.result is not None else PoolRuntimeCredential(DERIVED, TARGET)


@pytest.fixture
def env(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    profile = PairProfile(name=PROVIDER, base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile if name == PROVIDER else None)
    recovery.clear_runtime_overrides()
    # A real persisted pool: written through the pool's own persistence, read back by load_pool.
    CredentialPool(PROVIDER, [PooledCredential(
        provider=PROVIDER, id="row-a", label="row-a", auth_type="oauth", priority=0,
        source="manual", access_token=GRANT, base_url=SOURCE)])._persist()
    assert load_pool(PROVIDER).entries()[0].access_token == GRANT
    yield SimpleNamespace(profile=profile, auth=home / "auth.json")
    recovery.clear_runtime_overrides()


def _stored(env):
    text = env.auth.read_text()
    assert DERIVED not in text and TARGET not in text  # runtime-only: never persisted
    return json.loads(text)


@pytest.mark.parametrize("mode", ["select", "choose", "peek"])
def test_chooser_prepares_typed_pair_and_keeps_its_endpoint(env, mode):
    picked = rp.choose_pool_key(load_pool(PROVIDER), PROVIDER, mode=mode)
    assert picked is not None
    assert (picked.api_key, picked.base_url) == (DERIVED, TARGET)
    assert picked.entry.access_token == GRANT
    assert env.profile.calls == [GRANT]
    _stored(env)


def test_reloaded_pool_reuses_derived_pair_without_second_exchange(env):
    rp.choose_pool_key(load_pool(PROVIDER), PROVIDER, mode="peek")
    again = rp.choose_pool_key(load_pool(PROVIDER), PROVIDER, mode="peek")
    assert (again.api_key, again.base_url) == (DERIVED, TARGET)
    assert env.profile.calls == [GRANT]


def test_main_resolution_exchanges_once(env):
    runtime = rp._resolve_from_pool(PROVIDER, PROVIDER, {"provider": PROVIDER}, None, None, None)
    assert runtime is not None
    assert runtime["api_key"] == DERIVED and runtime["base_url"].rstrip("/") == TARGET
    assert env.profile.calls == [GRANT]
    _stored(env)


def test_refused_source_is_not_sent_the_derived_key(env):
    assert rp.choose_pool_key(load_pool(PROVIDER), PROVIDER, mode="peek",
                              admit_for="https://relay.unowned.example") is None


def test_pool_choice_commits_once_and_aux_client_carries_pair_and_row(env, monkeypatch):
    sent = []
    monkeypatch.setattr(aux, "_api_key_profile_supplied_client",
                        lambda provider, *, api_key, base_url: sent.append((api_key, base_url)) or SimpleNamespace())
    monkeypatch.setattr(aux, "_route_client", lambda req, client, model: (client, model))
    monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda provider: "m1")
    choice = rp.PoolChoice()
    req = aux._ResolveRequest(PROVIDER, PROVIDER, None, False, False, None, None, None, None, False, None, choice)
    pconfig = SimpleNamespace(inference_base_url=SOURCE, api_key_env_vars=())
    client, _model = aux._resolve_api_key_branch(req, pconfig, lambda p: {"api_key": "", "base_url": SOURCE})
    assert sent and sent[0][0] == DERIVED and sent[0][1].startswith(TARGET)
    assert client._hermes_credential_pool_entry_id == "row-a"
    assert choice.pool.entries()[0].request_count == 0  # choose leaves the pool alone
    choice.commit()  # the moved row is still the one choose found at the source endpoint
    row = choice.pool.entries()[0]
    assert row.request_count == 1
    assert (row.runtime_api_key, row.runtime_base_url) == (DERIVED, TARGET)  # override survives commit
    assert env.profile.calls == [GRANT]
    _stored(env)


def test_cached_peek_passes_pair_and_stamps_row(env, monkeypatch):
    calls = []

    def resolve(provider, model=None, async_mode=False, **kw):
        calls.append((kw["explicit_api_key"], kw["explicit_base_url"]))
        return SimpleNamespace(), "m1"

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    monkeypatch.setattr(aux, "_aux_probe_active", lambda: True)  # keep the global cache untouched
    client, _ = aux._get_cached_client(PROVIDER)
    assert calls == [(DERIVED, TARGET)]
    assert client._hermes_credential_pool_entry_id == "row-a"


def test_legacy_string_hook_keeps_row_endpoint(env):
    object.__setattr__(env.profile, "result", "legacy-key-0002")
    picked = rp.choose_pool_key(load_pool(PROVIDER), PROVIDER, mode="select")
    assert (picked.api_key, picked.base_url) == ("legacy-key-0002", None)
    assert picked.entry.runtime_base_url == SOURCE


def test_legacy_pool_without_transaction_rejects_typed_pair(env):
    row = load_pool(PROVIDER).entries()[0]
    legacy_pool = SimpleNamespace(has_credentials=lambda: True, select=lambda **kw: row)
    assert rp.choose_pool_key(legacy_pool, PROVIDER) is None


def test_nous_read_only_modes_never_refresh(monkeypatch):
    row = SimpleNamespace(id="n1", runtime_api_key="nous-key", access_token="nous-key")
    refreshed = []
    pool = SimpleNamespace(has_credentials=lambda: True, peek=lambda **kw: row, choose=lambda **kw: row,
                           try_refresh_current=lambda: refreshed.append(1),
                           prepare_runtime_entry=lambda *a, **kw: pytest.fail("nous is never prepared"))
    for mode in ("choose", "peek"):
        assert rp.choose_pool_key(pool, "nous", mode=mode).api_key == "nous-key"
    assert refreshed == []


def test_commit_never_exposes_durable_grant_to_existing_readers(env, monkeypatch):
    choice = rp.PoolChoice()
    picked = choice.choose(PROVIDER, PROVIDER, None, SOURCE)
    assert picked is not None
    row = picked.entry
    commit = picked.pool.commit

    def observing_commit(*args, **kwargs):
        # Readers retain a row reference and do not take the pool lock.
        assert row.runtime_api_key == DERIVED
        result = commit(*args, **kwargs)
        assert picked.pool.entries()[0].runtime_api_key == DERIVED
        return result

    monkeypatch.setattr(picked.pool, "commit", observing_commit)
    choice.commit()
    assert picked.pool.entries()[0].runtime_api_key == DERIVED


def test_round_robin_preserves_overrides_and_durable_changes_invalidate_them(env):
    from agent.credential_pool import STRATEGY_ROUND_ROBIN
    pool = load_pool(PROVIDER)
    other = PooledCredential(provider=PROVIDER, id="row-b", label="row-b", source="manual",
                             access_token=GRANT, auth_type="oauth", base_url=SOURCE, priority=1)
    pool._entries.append(other)
    pool._strategy = STRATEGY_ROUND_ROBIN
    for row in pool.entries():
        assert pool.prepare_runtime_entry(row) is not None
    row = pool.select()
    assert row is not None
    assert row.runtime_api_key == DERIVED
    assert all(item.runtime_api_key == DERIVED for item in pool.entries())
    changed = pool._adopt(row, access_token="replacement-grant", persist=False)
    assert changed.runtime_api_key == "replacement-grant"
    _stored(env)
