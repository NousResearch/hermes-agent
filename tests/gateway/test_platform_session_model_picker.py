"""Read-only picker catalog for Inline, without exposing operational endpoint inputs."""

import asyncio

import pytest

from hermes_agent.gateway.config import ChannelOverride, Platform, PlatformConfig
from tests.gateway.test_platform_session_model_settings import _Adapter


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    import hermes_agent.gateway.run as gateway_run
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session import SessionSource

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    adapter = object.__new__(_Adapter)
    adapter.gateway_runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="picker-chat", chat_type="dm")
    return adapter, runner, source, tmp_path


@pytest.fixture
def catalog(monkeypatch):
    from hermes_cli import model_switch_providers

    calls = []
    rows = [{
        "slug": "picked-provider", "name": "Picked", "models": ["picked-model"],
        "total_models": 7, "is_current": True,
        "api_url": "https://user:secret@models.example/v1?token=secret",
        "api_key": "secret", "token": "secret", "credential_pool": {"key": "secret"},
        "extra_headers": {"X-Secret": "secret"}, "future_secret": "secret",
    }]

    def lookup(**kwargs):
        calls.append(kwargs)
        return rows

    monkeypatch.setattr(model_switch_providers, "list_picker_providers", lookup)
    return calls, rows


def test_raw_override_and_profile_catalog_config_stay_inside_core(settings_env, catalog):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: global-model\n  provider: global-provider\n"
        "providers:\n  mine:\n    api_key: config-secret\n"
        "custom_providers:\n  - name: custom\n    base_url: https://custom.example/v1\n"
        "model_catalog:\n  excluded_providers: [excluded]\n"
    )
    key = runner._session_key_for_source(source)
    raw = "https://user:secret@picked.example/v1?api_key=secret#secret"
    override = {"model": "picked-model", "provider": "picked-provider", "base_url": raw}
    runner._session_model_overrides[key] = override
    adapter.list_session_model_picker_providers(source, max_models=23, include_moa=False)
    calls, _ = catalog
    assert calls == [{
        "current_provider": "picked-provider", "current_base_url": raw, "current_model": "picked-model",
        "user_providers": {"mine": {"api_key": "config-secret"}},
        "custom_providers": [{"name": "custom", "base_url": "https://custom.example/v1"}],
        "excluded_providers": ["excluded"], "max_models": 23, "include_moa": False,
        "probe_custom_providers": False, "probe_current_custom_provider": False,
        "non_blocking_catalogs": True,
    }]
    assert runner._session_model_overrides[key] is override
    assert override["base_url"] == raw


def test_rows_are_detached_and_allowlisted(settings_env, catalog):
    adapter, _, source, _ = settings_env
    calls, rows = catalog
    result = adapter.list_session_model_picker_providers(source)
    assert result == [{"slug": "picked-provider", "name": "Picked", "models": ["picked-model"],
                       "total_models": 7, "is_current": True}]
    assert result is not rows and result[0] is not rows[0]
    result[0]["models"].append("plugin-model")
    result[0]["name"] = "plugin-name"
    assert rows[0]["models"] == ["picked-model"]
    rows[0]["models"].append("later-model")
    assert result[0]["models"] == ["picked-model", "plugin-model"]
    assert calls[0]["max_models"] == 100
    assert calls[0]["include_moa"] is True


def test_channel_provider_only_and_no_session_allocation(settings_env, catalog, monkeypatch):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: global-model\n  provider: global-provider\n  base_url: https://global.example/v1\n"
    )
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(channel_overrides={
        source.chat_id: ChannelOverride(provider="channel-provider")})}

    def forbidden(*args, **kwargs):
        raise AssertionError("Picker reads must not prepare a session")

    monkeypatch.setattr(runner, "_session_state", forbidden)
    monkeypatch.setattr(runner, "_rehydrate_session_model_override", forbidden)
    before = vars(source).copy()
    adapter.list_session_model_picker_providers(source)
    assert vars(source) == before
    assert "_sessions" not in vars(runner)
    calls, _ = catalog
    assert calls[0]["current_provider"] == "channel-provider"
    assert calls[0]["current_model"] == "global-model"
    assert calls[0]["current_base_url"] == "https://global.example/v1"


@pytest.mark.parametrize("unattached", [True, False])
def test_unavailable_route_returns_none_without_catalog(settings_env, catalog, unattached):
    adapter, runner, source, _ = settings_env
    if unattached:
        adapter.gateway_runner = None
    else:
        source.profile_route_rejected = True
    assert adapter.list_session_model_picker_providers(source) is None
    assert catalog[0] == []
    assert "_sessions" not in vars(runner)


@pytest.mark.asyncio
async def test_thread_hop_binds_owning_profile_for_catalog(settings_env, catalog, monkeypatch):
    from hermes_agent.gateway.session_identity import RoutingIdentity
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret
    from hermes_cli import model_switch_providers

    adapter, runner, source, home = settings_env
    secondary = home / "secondary"
    secondary.mkdir()
    (home / "config.yaml").write_text("model: primary-model\n")
    (secondary / ".env").write_text("PICKER_MODEL=secondary-model\nCATALOG_KEY=secondary-secret\n")
    (secondary / "config.yaml").write_text("model:\n  default: ${PICKER_MODEL}\n  provider: secondary-provider\n")
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="secondary", authorization_home=home, runtime_home=secondary,
    )

    def lookup(**kwargs):
        assert get_hermes_home() == secondary
        assert get_secret("CATALOG_KEY") == "secondary-secret"
        assert kwargs["current_model"] == "secondary-model"
        assert kwargs["current_provider"] == "secondary-provider"
        return []

    monkeypatch.setattr(model_switch_providers, "list_picker_providers", lookup)
    assert await asyncio.to_thread(adapter.list_session_model_picker_providers, source) == []
    assert get_hermes_home() == home
    assert "_sessions" not in vars(runner)


def test_catalog_fetches_stay_in_the_owning_profile_scope(settings_env, monkeypatch):
    """With more than three keyed providers, blocking catalogs prefetch in a thread pool that does not
    carry the profile scope; cache-only catalogs keep every fetch in the owning profile."""
    from agent.secret_scope import get_secret
    from hermes_agent.gateway.session_identity import RoutingIdentity
    from hermes_cli import models
    from hermes_constants import get_hermes_home

    adapter, _, source, home = settings_env
    secondary = home / "secondary"
    secondary.mkdir()
    keys = ["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY"]
    (secondary / ".env").write_text("".join(f"{k}=secondary-{k.lower()}\n" for k in keys))
    (secondary / "config.yaml").write_text("model:\n  default: some-model\n  provider: deepseek\n")
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="secondary", authorization_home=home, runtime_home=secondary,
    )
    seen = []

    def fetch(slug, *args, **kwargs):
        seen.append((get_hermes_home(), get_secret("DEEPSEEK_API_KEY")))
        return []

    monkeypatch.setattr(models, "cached_provider_model_ids", fetch)
    adapter.list_session_model_picker_providers(source, include_moa=False)
    assert seen
    assert set(seen) == {(secondary, "secondary-deepseek_api_key")}


def test_persisted_pick_reaches_catalog_without_rehydration(settings_env, catalog):
    from hermes_agent.gateway.session import SessionStore
    from hermes_agent.gateway.config import GatewayConfig

    adapter, runner, source, home = settings_env
    before_restart = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    key = before_restart.get_or_create_session(source).session_key
    raw = "https://user:secret@picked.example/v1?token=secret"
    before_restart.set_model_override(key, {"model": "saved-model", "provider": "saved-provider", "base_url": raw})
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    adapter.list_session_model_picker_providers(source)
    assert catalog[0][0]["current_base_url"] == raw
    assert catalog[0][0]["current_provider"] == "saved-provider"
    assert catalog[0][0]["current_model"] == "saved-model"
    assert runner._peek_session_state(key) is None


def test_v12_custom_endpoint_is_listed_once_like_the_model_command(settings_env):
    """``providers:`` entries reach the catalog through the same compat view ``/model`` uses, so the
    current endpoint is not listed a second time as a synthetic custom row."""
    adapter, _, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: local-model\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n"
        "providers:\n  mylocal:\n    name: My Local\n    base_url: http://127.0.0.1:9/v1\n"
        "    models: [local-model, other-model]\n"
    )
    rows = adapter.list_session_model_picker_providers(source, include_moa=False)
    assert [row["slug"] for row in rows if row["is_current"]] == ["mylocal"]
