"""Runtime consumers adopt only centrally prepared pool credentials."""

from types import MethodType, SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp


SOURCE = "https://api.individual.githubcopilot.com"
TARGET = "https://api.business.githubcopilot.com"


def _entry(**changes):
    fields = dict(id="second", provider="openai", access_token="stored-token",
                  runtime_api_key="stored-token", base_url=SOURCE,
                  runtime_base_url=SOURCE, source="manual", label="second", priority=1,
                  last_status="available")
    return SimpleNamespace(**(fields | changes))


class PreparedPool(SimpleNamespace):
    provider = "openai"

    def __init__(self, entry, prepared):
        self.entry = entry
        self.prepared = prepared
        self.preparations = []

    def has_credentials(self):
        return True

    def select(self, **kwargs):
        return self.entry

    def prepare_runtime_entry(self, entry, *, source_base_url=None):
        self.preparations.append((entry, source_base_url))
        return self.prepared


@pytest.mark.parametrize("key", ["", "   "])
def test_legacy_pool_adapter_does_not_resolve_an_empty_secret(monkeypatch, key):
    row = _entry(access_token=key, runtime_api_key=key)
    pool = SimpleNamespace(provider="openai", has_credentials=lambda: True, select=lambda **kwargs: row)
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
    assert rp._resolve_from_pool("openai", "openai", {"provider": "openai"},
                                 None, None, "test-model") is None


def test_initial_resolution_installs_prepared_key_and_endpoint(monkeypatch):
    entry = _entry()
    prepared = _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET)
    pool = PreparedPool(entry, prepared)
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
    result = rp._resolve_from_pool("openai", "openai", {"provider": "openai"}, None, None, "test-model")
    assert result is not None
    assert result["api_key"] == "derived-bearer"
    assert result["base_url"] == TARGET
    assert pool.preparations == [(entry, None)]


def test_initial_resolution_prepares_before_rejecting_empty_stored_key(monkeypatch):
    entry = _entry(access_token="", runtime_api_key="")
    prepared = _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET)
    pool = PreparedPool(entry, prepared)
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
    result = rp._resolve_from_pool("openai", "openai", {"provider": "openai"}, None, None, "test-model")
    assert result is not None
    assert result["api_key"] == "derived-bearer"


def _agent(pool):
    from agent.client_lifecycle import ClientLifecycleMixin
    agent = SimpleNamespace(provider="openai", model="test-model", api_mode="chat_completions",
                            api_key="old-bearer", base_url=SOURCE, _credential_pool=pool,
                            _credential_pool_entry_id="first", _client_kwargs={"api_key": "old-bearer", "base_url": SOURCE},
                            _reapply_route_client_config=lambda **kwargs: None,
                            _replace_primary_openai_client=lambda **kwargs: None)
    agent._swap_credential = MethodType(ClientLifecycleMixin._swap_credential, agent)
    return agent


def test_swap_installs_prepared_runtime_without_changing_stored_credential():
    entry = _entry()
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    agent = _agent(pool)
    assert agent._swap_credential(entry) is True
    assert (agent.api_key, agent.base_url) == ("derived-bearer", TARGET)
    assert agent._client_kwargs == {"api_key": "derived-bearer", "base_url": TARGET}
    assert entry.access_token == "stored-token"
    assert pool.preparations == [(entry, SOURCE)]


def test_rotation_admits_prepared_endpoint_move():
    from agent.agent_runtime_helpers import recover_with_credential_pool
    entry = _entry(base_url=TARGET, runtime_base_url=TARGET)
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    pool.mark_exhausted_and_rotate = lambda **kwargs: entry
    pool.entries = lambda: [entry]
    agent = _agent(pool)
    recovered, retried = recover_with_credential_pool(agent, status_code=402, has_retried_429=False)
    assert recovered is True
    assert retried is False
    assert (agent.api_key, agent.base_url) == ("derived-bearer", TARGET)


def test_primary_rebind_prepares_before_rejecting_empty_stored_key():
    from agent.agent_runtime_helpers import _rebind_primary_credential_pool
    entry = _entry(access_token="", runtime_api_key="")
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    pool.has_available = lambda **kwargs: True
    agent = _agent(pool)
    _rebind_primary_credential_pool(agent, "openai", "test-model", lambda candidate: True,
                                    lambda: pool, None, False)
    assert (agent.api_key, agent.base_url) == ("derived-bearer", TARGET)


def test_child_lease_admits_prepared_endpoint_move():
    from tools.delegate_tool_child_run import _lease_child_credential
    entry = _entry(base_url=TARGET, runtime_base_url=TARGET)
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    pool.entries = lambda: [entry]
    leases = []
    released = []
    pool.acquire_lease = lambda entry_id=None: leases.append(entry_id) or entry.id
    pool.release_lease = released.append
    child = _agent(pool)
    assert _lease_child_credential(child) == (pool, entry.id)
    assert (child.api_key, child.base_url) == ("derived-bearer", TARGET)
    assert released == []


def test_child_releases_lease_when_swap_refuses():
    from tools.delegate_tool_child_run import _lease_child_credential
    entry = _entry()
    pool = PreparedPool(entry, entry)
    pool.entries = lambda: [entry]
    pool.acquire_lease = lambda entry_id=None: entry.id
    released = []
    pool.release_lease = released.append
    child = _agent(pool)
    child._swap_credential = lambda entry: False
    assert _lease_child_credential(child) == (pool, None)
    assert released == [entry.id]


def test_reclaim_keeps_retry_armed_when_runtime_preparation_refuses():
    from agent.agent_runtime_helpers import _revert_credential_rotation
    entry = _entry()
    pool = PreparedPool(entry, None)
    pool.reclaim = lambda entry_id, **kwargs: entry
    agent = _agent(pool)
    agent._credential_pool_revert_id = entry.id
    _revert_credential_rotation(agent)
    assert agent._credential_pool_revert_id == entry.id
    assert (agent.api_key, agent.base_url) == ("old-bearer", SOURCE)


def test_initial_resolution_preserves_configured_route_without_runtime_move(monkeypatch):
    entry = _entry(provider="xai", base_url="https://api.x.ai/v1", runtime_base_url="https://api.x.ai/v1")
    pool = PreparedPool(entry, entry)
    pool.provider = "xai"
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
    relay = "https://relay.example/v1"
    result = rp._resolve_from_pool("xai", "xai", {"provider": "xai", "base_url": relay},
                                   None, None, "test-model")
    assert result is not None
    assert result["base_url"] == relay
    assert pool.preparations == [(entry, None)]


@pytest.mark.parametrize("provider, canonical", [
    ("xai", "https://api.x.ai/v1"),
    ("openai-codex", "https://chatgpt.com/backend-api/codex"),
])
def test_initial_real_pool_preserves_configured_proxy(monkeypatch, provider, canonical):
    from agent.credential_pool import CredentialPool, PooledCredential
    row = PooledCredential(provider=provider, id="proxy", label="proxy", auth_type="api_key",
                           priority=0, source="manual", access_token="stored-token", base_url=canonical)
    pool = CredentialPool(provider, [row])
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
    relay = "https://relay.example/v1"
    result = rp._resolve_from_pool(provider, provider, {"provider": provider, "base_url": relay},
                                   None, None, "test-model")
    assert result is not None
    assert (result["api_key"], result["base_url"]) == ("stored-token", relay)


@pytest.mark.parametrize("stage", ["prepare", "swap"])
def test_child_releases_lease_when_binding_raises(stage):
    from tools.delegate_tool_child_run import _lease_child_credential
    entry = _entry()
    pool = PreparedPool(entry, entry)
    pool.entries = lambda: [entry]
    pool.acquire_lease = lambda entry_id=None: entry.id
    released = []
    pool.release_lease = released.append
    child = _agent(pool)

    def fail(*args, **kwargs):
        raise ValueError("synthetic binding failure")

    if stage == "prepare":
        pool.prepare_runtime_entry = fail
    else:
        child._swap_credential = fail
    assert _lease_child_credential(child) == (pool, None)
    assert released == [entry.id]


def test_swap_rechecks_active_override_before_any_agent_mutation():
    entry = _entry()
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    agent = _agent(pool)
    prepared = pool.prepare_runtime_entry(entry, source_base_url=agent.base_url)
    pool.prepared = None  # The active route/override became inadmissible after selection.
    before = dict(vars(agent))
    before_kwargs = dict(agent._client_kwargs)
    assert agent._swap_credential(prepared) is False
    assert vars(agent) == before
    assert agent._client_kwargs == before_kwargs
    assert pool.preparations[-1] == (prepared, SOURCE)


def test_reclaim_adopts_prepared_runtime_and_disarms_retry():
    from agent.agent_runtime_helpers import _revert_credential_rotation
    entry = _entry()
    pool = PreparedPool(entry, _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET))
    pool.reclaim = lambda entry_id, **kwargs: entry
    agent = _agent(pool)
    agent._credential_pool_revert_id = entry.id
    _revert_credential_rotation(agent)
    assert (agent.api_key, agent.base_url) == ("derived-bearer", TARGET)
    assert agent._credential_pool_revert_id is None


def test_child_releases_rejected_entry_and_leases_prepared_alternative():
    from tools.delegate_tool_child_run import _lease_child_credential
    rejected = _entry(id="rejected")
    entry = _entry(base_url=TARGET, runtime_base_url=TARGET)
    prepared = _entry(runtime_api_key="derived-bearer", runtime_base_url=TARGET)
    pool = PreparedPool(entry, prepared)
    pool.entries = lambda: [rejected, entry]
    pool.prepare_runtime_entry = lambda candidate, **kwargs: None if candidate.id == rejected.id else prepared
    acquired, released = [], []
    pool.acquire_lease = lambda entry_id=None: acquired.append(entry_id) or entry_id or rejected.id
    pool.release_lease = released.append
    child = _agent(pool)
    assert _lease_child_credential(child) == (pool, entry.id)
    assert (child.api_key, child.base_url) == ("derived-bearer", TARGET)
    assert acquired == [None, entry.id]
    assert released == [rejected.id]


def test_primary_rebind_does_not_report_refused_swap_as_restored(caplog):
    from agent.agent_runtime_helpers import _rebind_primary_credential_pool
    entry = _entry()
    pool = PreparedPool(entry, entry)
    pool.has_available = lambda **kwargs: True
    agent = _agent(pool)
    agent._swap_credential = lambda entry, **kwargs: False
    with caplog.at_level("INFO", logger="agent.agent_runtime_helpers"):
        _rebind_primary_credential_pool(agent, "openai", "test-model", lambda candidate: True,
                                        lambda: pool, None, False)
    assert "Restore re-selected" not in caplog.text


@pytest.mark.parametrize("stage", ["initial", "restore", "swap"])
@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("bad", [
    "https://api.individual.githubcopilot.com.evil.test",
    "https://api.individual.githubcopilot.com/v1",
    "https://api.individual.githubcopilot.com?x=1",
    "https://user@api.individual.githubcopilot.com",
    "https://api.individual.githubcopilot.com:8443",
])
def test_adoption_preserves_exact_source_and_target_allowlist(monkeypatch, stage, side, bad):
    import providers
    from agent.credential_pool import CredentialPool, PooledCredential
    from agent.agent_runtime_helpers import _rebind_primary_credential_pool
    from providers.base import PoolRuntimeCredential, ProviderProfile

    class PairProfile(ProviderProfile):
        def prepare_pool_entry(self, entry, api_key):
            return PoolRuntimeCredential("derived-bearer", bad if side == "target" else TARGET)

    profile = PairProfile(name="openai", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile if name == "openai" else None)
    row = PooledCredential(provider="openai", id="guarded", label="guarded", auth_type="api_key",
                           priority=0, source="manual", access_token="stored-token",
                           base_url=bad if side == "source" else SOURCE)
    pool = CredentialPool("openai", [row])
    agent = _agent(pool)
    if stage == "initial":
        monkeypatch.setattr(rp, "load_pool", lambda provider: pool)
        assert rp._resolve_from_pool("openai", "openai", {"provider": "openai"},
                                     None, None, "test-model") is None
    elif stage == "restore":
        _rebind_primary_credential_pool(agent, "openai", "test-model", lambda candidate: True,
                                        lambda: pool, None, False)
        assert (agent.api_key, agent.base_url) == ("old-bearer", SOURCE)
    else:
        if side == "source":
            agent.base_url = bad
        before = (agent.api_key, agent.base_url, dict(agent._client_kwargs))
        assert agent._swap_credential(row) is False
        assert (agent.api_key, agent.base_url, agent._client_kwargs) == before
    assert (row.runtime_api_key, row.runtime_base_url) == ("stored-token", row.base_url)
