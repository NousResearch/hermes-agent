"""Auxiliary pool reads prepare derived runtime credentials and keep renewals across disk reloads."""

from types import SimpleNamespace

import pytest

import providers
from agent import auxiliary_client as aux
from agent import auxiliary_error_policy as aux_errors
from agent import auxiliary_pool_recovery as recovery
from agent.auxiliary_plugin_oauth import resolve_plugin_oauth_client
from agent.credential_pool import CredentialPool, PooledCredential
from providers.base import PoolRuntimeCredential, ProviderProfile

SOURCE = "https://api.individual.example"
TARGET = "https://api.business.example"
GRANT = "durable-grant-0001"
DERIVED = "derived-bearer-0001"
RENEWED = "renewed-bearer-0002"


class PairProfile(ProviderProfile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "renewal", None)

    def prepare_pool_entry(self, entry, api_key):
        self.calls.append(("prepare", entry.id, api_key))
        return PoolRuntimeCredential(DERIVED, TARGET)

    def refresh_pool_runtime_credentials(self, entry, *, reason):
        self.calls.append(("refresh", entry.id, reason))
        return self.renewal


@pytest.fixture
def env(monkeypatch, tmp_path):
    home_a = tmp_path / "profile-a"
    home_a.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    prof = PairProfile(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: prof if name == "pairprov" else None)
    loads = []

    def load_pool(provider):  # a fresh disk read every time, as the real loader does
        pool = CredentialPool(provider, [PooledCredential(
            provider=provider, id="row-a", label="row-a", auth_type="oauth", priority=0,
            source="manual", access_token=GRANT, base_url=SOURCE)])
        loads.append(pool)
        return pool

    sent = []

    def client(provider, *, api_key, base_url):
        sent.append((api_key, base_url))
        return SimpleNamespace(api_key=api_key, base_url=base_url)

    monkeypatch.setattr(aux, "load_pool", load_pool)
    monkeypatch.setattr(aux, "_api_key_profile_supplied_client", client)
    monkeypatch.setattr(aux, "_route_client", lambda req, c, m: (c, m))
    monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda p: "m1")
    monkeypatch.setattr(aux, "_evict_cached_clients", lambda p: None)
    monkeypatch.setattr(aux_errors, "auth_refresh", lambda verdict: True)
    getattr(recovery, "clear_runtime_overrides", lambda: None)()
    yield SimpleNamespace(profile=prof, sent=sent, loads=loads, tmp=tmp_path, mp=monkeypatch)
    getattr(recovery, "clear_runtime_overrides", lambda: None)()


def _req(**kw):
    return SimpleNamespace(**{"provider": "pairprov", "explicit_api_key": None, "explicit_base_url": None,
                              "model": None, **kw})


def test_selected_row_is_prepared_and_grant_never_sent(env):
    client, _model = resolve_plugin_oauth_client(_req())
    assert env.sent == [(DERIVED, TARGET)]
    assert all(GRANT != key for key, _ in env.sent)
    assert client._hermes_credential_pool_entry_id == "row-a"


def test_unattributed_aux_401_does_not_bench_the_current_row(env):
    first, _ = resolve_plugin_oauth_client(_req())
    env.profile.renewal = PoolRuntimeCredential(RENEWED, TARGET)
    assert not aux._recover_provider_pool("pairprov", RuntimeError("401"), failed_api_key=first.api_key,
                                          failed_credential_id=None, verdict=SimpleNamespace(
                                              status_code=401, reason=aux.FailoverReason.auth, error_context={},
                                              billing_unverified=False, model=None))
    assert env.loads[-1].entries()[0].last_status is None
    assert not [call for call in env.profile.calls if call[0] == "refresh"]


def test_renewal_survives_reload_and_retry_uses_new_key(env):
    first, _ = resolve_plugin_oauth_client(_req())
    env.profile.renewal = PoolRuntimeCredential(RENEWED, TARGET)
    assert aux._recover_provider_pool("pairprov", RuntimeError("401"), failed_api_key=first.api_key,
                                      failed_credential_id="row-a", verdict=SimpleNamespace(status_code=401, reason=None))
    retry, _ = resolve_plugin_oauth_client(_req())
    assert retry.api_key == RENEWED and retry.base_url == TARGET
    assert GRANT not in [key for key, _ in env.sent]
    assert len({id(p) for p in env.loads}) == len(env.loads) >= 3  # really reloaded each time


def test_main_prepared_bearer_is_attributed_to_same_row_in_aux(env):
    pool = aux.load_pool("pairprov")
    main_row = pool.prepare_runtime_entry(pool.entries()[0], source_base_url=SOURCE)
    assert main_row.runtime_api_key == DERIVED
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=DERIVED, explicit_base_url=TARGET))
    assert client._hermes_credential_pool_entry_id == "row-a"
    env.profile.renewal = PoolRuntimeCredential(RENEWED, TARGET)
    assert aux._recover_provider_pool("pairprov", RuntimeError("401"), failed_api_key=DERIVED,
                                      failed_credential_id="row-a", verdict=SimpleNamespace(status_code=401, reason=None))
    retry, _ = resolve_plugin_oauth_client(_req())
    assert retry.api_key == RENEWED


def test_explicit_derived_bearer_is_attributed_to_its_row(env):
    resolve_plugin_oauth_client(_req())
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=DERIVED, explicit_base_url=TARGET))
    assert client.api_key == DERIVED
    assert client._hermes_credential_pool_entry_id == "row-a"


def test_explicit_durable_grant_is_never_sent(env):
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=GRANT))
    assert GRANT not in [key for key, _ in env.sent]
    assert client is None or (client.api_key == DERIVED and client._hermes_credential_pool_entry_id == "row-a")


class RefreshOnlyProfile(ProviderProfile):
    def refresh_pool_runtime_credentials(self, entry, *, reason):
        return PoolRuntimeCredential(RENEWED, TARGET)


class IdentityStringProfile(RefreshOnlyProfile):
    def prepare_pool_entry(self, entry, api_key):
        return api_key


@pytest.mark.parametrize("profile_type", [RefreshOnlyProfile, IdentityStringProfile])
def test_explicit_existing_runtime_key_is_dispatched_with_row_attribution(env, profile_type):
    profile = profile_type(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    env.mp.setattr(providers, "get_provider_profile", lambda name: profile if name == "pairprov" else None)

    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=GRANT, explicit_base_url=SOURCE))

    assert env.sent == [(GRANT, SOURCE)]
    assert client.api_key == GRANT
    assert client._hermes_credential_pool_entry_id == "row-a"


@pytest.mark.parametrize("runtime_base_url", [None, TARGET])
def test_explicit_durable_grant_is_refused_when_typed_preparation_replays_it(env, runtime_base_url):
    class RawPairProfile(PairProfile):
        def prepare_pool_entry(self, entry, api_key):
            return PoolRuntimeCredential(api_key, runtime_base_url)

    profile = RawPairProfile(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    env.mp.setattr(providers, "get_provider_profile", lambda name: profile if name == "pairprov" else None)

    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=GRANT, explicit_base_url=SOURCE))

    assert client is None
    assert env.sent == []


def test_explicit_key_is_not_dispatched_when_pool_lookup_fails(env):
    def fail(_provider):
        raise OSError("pool unavailable")
    env.mp.setattr(aux, "load_pool", fail)
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=GRANT))
    assert client is None
    assert env.sent == []


def test_foreign_explicit_endpoint_is_refused(env):
    client, _ = resolve_plugin_oauth_client(_req(explicit_base_url="https://api.business.example.evil"))
    assert client is None
    assert env.sent == []


def test_cached_override_does_not_leak_across_profiles(env):
    resolve_plugin_oauth_client(_req())
    env.profile.renewal = PoolRuntimeCredential(RENEWED, TARGET)
    assert aux._recover_provider_pool("pairprov", RuntimeError("401"), failed_api_key=DERIVED,
                                      failed_credential_id="row-a", verdict=SimpleNamespace(status_code=401, reason=None))
    home_b = env.tmp / "profile-b"
    home_b.mkdir()
    env.mp.setenv("HERMES_HOME", str(home_b))
    client, _ = resolve_plugin_oauth_client(_req())
    assert client.api_key == DERIVED  # profile B prepares its own row, never adopts A's renewal
    assert env.profile.calls.count(("prepare", "row-a", GRANT)) == 2


def test_rotated_grant_invalidates_cached_override(env):
    resolve_plugin_oauth_client(_req())
    original = aux.load_pool

    def rotated(provider):
        pool = original(provider)
        pool.entries()[0].access_token = "durable-grant-0002"
        return pool

    env.mp.setattr(aux, "load_pool", rotated)
    client, _ = resolve_plugin_oauth_client(_req())
    assert client.api_key == DERIVED
    assert ("prepare", "row-a", "durable-grant-0002") in env.profile.calls


@pytest.mark.parametrize("foreign", ["https://relay.example.evil", "https://api.business.example.evil",
                                     "http://api.business.example", "https://api.business.example/v1"])
def test_explicit_derived_bearer_is_not_sent_to_foreign_endpoint(env, foreign):
    resolve_plugin_oauth_client(_req())  # the derived bearer is active and cached
    env.sent.clear()
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key=DERIVED, explicit_base_url=foreign))
    assert client is None
    assert env.sent == []


def test_unattributed_explicit_key_is_not_sent_to_foreign_endpoint(env):
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key="unknown-key-0001",
                                                 explicit_base_url="https://relay.example.evil"))
    assert client is None
    assert env.sent == []


def test_unattributed_explicit_key_to_allowlisted_endpoint_still_sent(env):
    client, _ = resolve_plugin_oauth_client(_req(explicit_api_key="session-key-0001", explicit_base_url=TARGET))
    assert env.sent == [("session-key-0001", TARGET)]
    assert getattr(client, "_hermes_credential_pool_entry_id", None) is None


def test_legacy_provider_explicit_key_behavior_unchanged(env):
    pool = CredentialPool("legacy", [PooledCredential(
        provider="legacy", id="row-l", label="row-l", auth_type="api_key", priority=0,
        source="manual", access_token="legacy-key-0001", base_url="https://custom.example/v1")])
    entry, key = recovery.entry_for_explicit_key(pool, "legacy-key-0001", source_base_url="https://other.example/v1")
    assert entry is not None and entry.id == "row-l" and key == "legacy-key-0001"
    entry, key = recovery.entry_for_explicit_key(pool, "other-key", source_base_url="https://other.example/v1")
    assert entry is None and key == "other-key"


# --- Auto api-key chain (_resolve_api_key_provider) for a provider deriving runtime credentials ---

@pytest.fixture
def chain(env):
    from hermes_cli import auth
    from hermes_cli import auth_plugin_providers as app
    rows = [("pairprov", SimpleNamespace(id="pairprov", name="Pair", auth_type="api_key", inference_base_url=SOURCE)),
            ("legacyprov", SimpleNamespace(id="legacyprov", name="Legacy", auth_type="api_key",
                                           inference_base_url="https://legacy.example/v1"))]
    env.mp.setattr(app, "canonical_registry_rows", lambda: list(rows))
    ambient = []

    def resolve(provider_id):
        ambient.append(provider_id)
        key = GRANT if provider_id == "pairprov" else "legacy-key-0001"
        return {"api_key": key, "base_url": SOURCE if provider_id == "pairprov" else "https://legacy.example/v1"}

    env.mp.setattr(auth, "resolve_api_key_provider_credentials", resolve)
    env.mp.setattr(aux, "_is_provider_unhealthy", lambda *a, **k: False)
    env.mp.setattr(aux, "_normalize_main_runtime", lambda _rt: {})
    built = []

    def create(*, api_key, base_url, **_kw):
        built.append((api_key, base_url))
        return SimpleNamespace(api_key=api_key, base_url=base_url)

    env.mp.setattr(aux, "_create_openai_client", create)
    return SimpleNamespace(rows=rows, ambient=ambient, built=built)


def test_auto_chain_pooled_client_is_stamped_and_401_scoped_to_row(env, chain):
    client, _model = aux._resolve_api_key_provider()
    assert chain.built and chain.built[0][0] == DERIVED
    assert GRANT not in [key for key, _ in chain.built]
    assert client._hermes_credential_pool_entry_id == "row-a"
    env.profile.renewal = PoolRuntimeCredential(RENEWED, TARGET)
    route = SimpleNamespace(resolved_provider="pairprov", client=client, main_runtime=None)
    failure = SimpleNamespace(error=RuntimeError("401"), verdict=SimpleNamespace(status_code=401, reason=None),
                              client=client)
    env.mp.setattr(aux, "_failure_sender", lambda r, f: client)
    env.mp.setattr(aux, "_recoverable_pool_provider", lambda *a, **k: "pairprov")
    assert aux._recover_failure_pool(route, failure)
    assert ("refresh", "row-a", "unauthorized") in env.profile.calls
    retry, _ = aux._resolve_api_key_provider()
    assert retry.api_key == RENEWED


def test_auto_chain_without_pool_never_sends_durable_grant_of_deriving_provider(env, chain):
    env.mp.setattr(aux, "load_pool", lambda provider: CredentialPool(provider, []))
    client, _model = aux._resolve_api_key_provider()
    assert GRANT not in [key for key, _ in chain.built]
    assert "pairprov" not in chain.ambient
    assert client is not None and client.api_key == "legacy-key-0001"  # legacy provider still served


def test_auto_chain_legacy_provider_ambient_fallback_unchanged(env, chain):
    chain.rows.pop(0)
    env.mp.setattr(aux, "load_pool", lambda provider: CredentialPool(provider, []))
    client, _model = aux._resolve_api_key_provider()
    assert chain.ambient == ["legacyprov"]
    assert client.api_key == "legacy-key-0001" and chain.built == [("legacy-key-0001", "https://legacy.example/v1")]
    assert getattr(client, "_hermes_credential_pool_entry_id", None) is None
