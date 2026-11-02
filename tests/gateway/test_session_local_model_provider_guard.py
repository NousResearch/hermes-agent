"""The native ``session.create`` path rejects a model its provider cannot serve, before any state exists.

Desktop and the current CLI create sessions here, not through the legacy ``tui_gateway`` handler, so a
model with no provider used to inherit the profile's and fail or fall back on its first turn.
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture
def plugin_provider():
    """A provider that only a plugin knows about, shaped like a Claude subscription bridge."""
    from providers import _REGISTRY, register_provider
    from providers.base import ProviderProfile

    name = "claude-bridge-fixture"
    register_provider(ProviderProfile(
        name=name, base_url=f"process://{name}", auth_type="external_process",
        fallback_models=("claude-sonnet-5-5", "claude-opus-5-5"), model_aliases={"sonnet": "claude-sonnet-5-5"},
    ))
    yield name
    _REGISTRY.pop(name, None)


@pytest.fixture
def namespaced_host():
    """A hosted catalog whose ids share an account path, shaped like Fireworks: one head, many vendors."""
    from providers import _REGISTRY, register_provider
    from providers.base import ProviderProfile

    name = "namespaced-host-fixture"
    register_provider(ProviderProfile(
        name=name, base_url="https://host.example/v1",
        fallback_models=("accounts/host/models/kimi-k2", "accounts/host/models/glm-5"),
    ))
    yield name
    _REGISTRY.pop(name, None)


@asynccontextmanager
async def _creator(tmp_path, monkeypatch, provider, *, base_url="", dotenv=""):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  default: claude-sonnet-5-5\n  provider: {provider}\n  base_url: '{base_url}'\n", encoding="utf-8")
    (home / ".env").write_text(dotenv, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run, "_hermes_home", home)  # the home a new session's `.env` scope reads
    model_config = {"default": "claude-sonnet-5-5", "provider": provider, "base_url": base_url}
    monkeypatch.setattr(run, "_load_gateway_config", lambda *a: {"platform_toolsets": {"cli": []}, "model": model_config})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id="fixture", instance_id="owner")
    actor = AuthorityConnection(authority, object(), {"user_id": "human", "profile_id": "fixture"}).actor
    counter = iter(range(100))

    def create(**params):
        return create_local_session(authority, actor, {
            "request_id": f"r{next(counter)}", "source": "gui", "cwd": str(tmp_path), "toolsets": [], **params})

    create.rows = lambda: authority.db.search_sessions(limit=100)
    try:
        yield create
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "gpt-6.1-terra",  # the reported pair: provider implied by the profile
    "gpt-5.5",
    "grok-5",
])
async def test_foreign_family_on_a_plugin_provider_is_refused_before_any_state(plugin_provider, monkeypatch, tmp_path, model):
    from hermes_state_runtime import RuntimeStoreError

    async with _creator(tmp_path, monkeypatch, plugin_provider) as create:
        with pytest.raises(RuntimeStoreError) as refused:
            create(model=model)

        assert refused.value.reason == "model_provider_conflict"
        assert refused.value.field == "model"
        assert model in refused.value.detail and plugin_provider in refused.value.detail
        assert create.rows() == []


@pytest.mark.asyncio
async def test_explicit_incoherent_pair_is_refused(plugin_provider, monkeypatch, tmp_path):
    from hermes_state_runtime import RuntimeStoreError

    async with _creator(tmp_path, monkeypatch, plugin_provider) as create:
        with pytest.raises(RuntimeStoreError, match="model_provider_conflict"):
            create(model="gpt-6.1-terra", provider="anthropic")


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {},  # no pin: the profile default
    {"model": "claude-sonnet-5-5"},  # listed
    {"model": "claude-opus-9"},  # same family, not listed yet
    {"model": "sonnet"},  # alias
    {"model": "my-private-slug"},  # no vendor claims it
    {"model": "gpt-6.1-terra", "provider": "openai-codex"},  # the provider's own family
    {"model": "gpt-6.1-terra", "provider": "custom"},  # Hermes cannot know a custom endpoint's models
    {"model": "gpt-6.1-terra", "base_url": "https://proxy.example/v1"},  # the client names the route
])
async def test_coherent_and_undecidable_pairs_still_create(plugin_provider, monkeypatch, tmp_path, params):
    async with _creator(tmp_path, monkeypatch, plugin_provider) as create:
        assert create(**params).session_id


@pytest.mark.asyncio
async def test_redirected_anthropic_endpoint_stays_undecidable(monkeypatch, tmp_path):
    async with _creator(tmp_path, monkeypatch, "anthropic", base_url="https://proxy.example/anthropic") as create:
        assert create(model="gpt-6.1-terra").session_id


@pytest.mark.asyncio
async def test_a_namespaced_catalog_names_no_vendor_so_stays_undecidable(namespaced_host, monkeypatch, tmp_path):
    async with _creator(tmp_path, monkeypatch, namespaced_host) as create:
        assert create(model="deepseek-v4").session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "claude"])  # `claude` is an alias of the same provider
async def test_a_base_url_in_the_profile_env_stays_undecidable(monkeypatch, tmp_path, provider):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    dotenv = "ANTHROPIC_BASE_URL=https://proxy.example/anthropic\n"
    async with _creator(tmp_path, monkeypatch, "anthropic", dotenv=dotenv) as create:
        assert create(model="gpt-6.1-terra", provider=provider).session_id


@pytest.mark.asyncio
async def test_an_exact_retry_keeps_the_session_it_already_has(monkeypatch, tmp_path):
    """The gate runs after the retry returns: a proxy removed since the create is no reason to lose a chat."""
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    dotenv = "ANTHROPIC_BASE_URL=https://proxy.example/anthropic\n"
    async with _creator(tmp_path, monkeypatch, "anthropic", dotenv=dotenv) as create:
        created = create(model="gpt-6.1-terra", request_id="kept")
        (tmp_path / "home" / ".env").write_text("", encoding="utf-8")

        assert create(model="gpt-6.1-terra", request_id="kept").session_id == created.session_id
        with pytest.raises(RuntimeStoreError, match="model_provider_conflict"):
            create(model="gpt-6.1-terra", request_id="new")


@pytest.mark.asyncio
async def test_a_client_reads_the_refusal_off_the_wire(monkeypatch, tmp_path):
    """Through the RPC dispatch Desktop and the CLI use: a named refusal, and no session behind it."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: claude-sonnet-5-5\n  provider: anthropic\n", encoding="utf-8")
    monkeypatch.setattr(run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="fixture")
    owner = AuthorityConnection(authority, object(), {"user_id": "owner"})
    try:
        response = await owner.dispatch({"id": "c", "method": "session.create", "params": {
            "request_id": "c", "source": "gui", "cwd": str(home), "toolsets": [], "model": "gpt-6.1-terra"}})

        assert response["error"]["message"] == "model_provider_conflict"
        assert response["error"]["data"]["field"] == "model"
        assert "gpt-6.1-terra" in response["error"]["data"]["detail"]
        assert authority.db.search_sessions(limit=10) == []
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_native_anthropic_refuses_the_foreign_family(monkeypatch, tmp_path):
    from hermes_state_runtime import RuntimeStoreError

    async with _creator(tmp_path, monkeypatch, "anthropic") as create:
        with pytest.raises(RuntimeStoreError, match="model_provider_conflict"):
            create(model="gpt-6.1-terra")
