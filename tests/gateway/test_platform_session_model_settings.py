"""Public display-only model settings for platform plugins (Inline settings sheet)."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("A settings read must not send messages")

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    adapter = object.__new__(_Adapter)
    adapter.gateway_runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="settings-chat", chat_type="dm")
    return adapter, runner, source, tmp_path


def test_defaults_without_session_allocation(settings_env):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: configured-model\n  provider: configured-provider\n  base_url: https://models.example/v1\n"
    )
    before = vars(source).copy()
    result = adapter.get_session_model_settings(source)
    assert vars(source) == before
    assert "_sessions" not in vars(runner)
    assert result == {
        "session_key": runner._session_key_for_source(source),
        "model": "configured-model", "provider": "configured-provider",
        "base_url": "https://models.example/v1",
    }
    assert runner._peek_session_state(result["session_key"]) is None


@pytest.mark.parametrize("config, expected", [
    ("{}", {"model": "", "provider": "openrouter", "base_url": ""}),
    ("model: flat-model", {"model": "flat-model", "provider": "openrouter", "base_url": ""}),
])
def test_missing_settings_and_flat_model(settings_env, config, expected):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(config)
    result = adapter.get_session_model_settings(source)
    assert {key: result[key] for key in expected} == expected
    assert runner._peek_session_state(result["session_key"]) is None


def test_override_snapshot_excludes_secrets_and_preserves_legacy_read(settings_env):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: default-provider\n  api_key: config-secret\n"
    )
    key = runner._session_key_for_source(source)
    override = {
        "model": "picked-model", "provider": "picked-provider", "base_url": "https://picked.example/v1",
        "api_key": "session-secret", "credential_pool": {"key": "pool-secret"},
        "extra_headers": {"X-Secret": "header-secret"},
    }
    runner._session_model_overrides[key] = override
    result = adapter.get_session_model_settings(source)
    assert result == {"session_key": key, "model": "picked-model", "provider": "picked-provider",
                      "base_url": "https://picked.example/v1"}
    result["model"] = "plugin-edit"
    assert runner._session_model_overrides[key] is override
    assert override["model"] == "picked-model"
    override["provider"] = "new-provider"
    assert result["provider"] == "picked-provider"
    assert adapter.get_session_model_settings(source)["provider"] == "new-provider"


def test_partial_override_falls_back_to_defaults(settings_env):
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: default-provider\n  base_url: https://default.example/v1\n"
    )
    key = runner._session_key_for_source(source)
    runner._session_model_overrides[key] = {"model": "picked-model", "provider": None, "base_url": ""}
    assert adapter.get_session_model_settings(source) == {
        "session_key": key, "model": "picked-model", "provider": "default-provider",
        "base_url": "https://default.example/v1",
    }


def test_persisted_override_is_shown_before_the_first_turn_after_restart(settings_env, monkeypatch):
    import hermes_state
    from gateway.session import SessionStore

    def _no_sqlite():
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _no_sqlite)
    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text("model:\n  default: default-model\n  provider: default-provider\n")
    before_restart = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    key = before_restart.get_or_create_session(source).session_key
    before_restart.set_model_override(key, {
        "model": "picked-model", "provider": "picked-provider", "base_url": "https://picked.example/v1"})
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    assert adapter.get_session_model_settings(source) == {
        "session_key": key, "model": "picked-model", "provider": "picked-provider",
        "base_url": "https://picked.example/v1",
    }
    assert runner._peek_session_state(key) is None


def test_unattached_adapter_returns_none(settings_env):
    adapter, _, source, _ = settings_env
    adapter.gateway_runner = None
    assert adapter.get_session_model_settings(source) is None


def test_existing_session_without_override_and_absent_config(settings_env):
    adapter, runner, source, _ = settings_env
    key = runner._session_key_for_source(source)
    state = runner._session_state(key)
    assert adapter.get_session_model_settings(source) == {
        "session_key": key, "model": "", "provider": "openrouter", "base_url": "",
    }
    assert runner._peek_session_state(key) is state
    assert state.conversation.model_override is None


@pytest.mark.parametrize("base_url, expected", [
    ("https://user:password@models.example/v1?api_key=hidden#secret", "https://models.example/v1"),
    ("https://[::1]:8000/v1", "https://[::1]:8000/v1"),
    ("https://user:password@[invalid", ""),
    ("https://models.example:not-a-port/v1", ""),
    ("opaque-secret", ""),
    ("https:opaque-secret", ""),
])
def test_endpoint_excludes_url_credentials(settings_env, base_url, expected):
    adapter, runner, source, _ = settings_env
    key = runner._session_key_for_source(source)
    runner._session_state(key).conversation.model_override = {"base_url": base_url}
    assert adapter.get_session_model_settings(source)["base_url"] == expected
    assert runner._session_model_overrides[key]["base_url"] == base_url


def test_channel_and_session_precedence(settings_env):
    from gateway.config import ChannelOverride

    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text("model:\n  default: global-model\n  provider: global-provider\n")
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(channel_overrides={
        source.chat_id: ChannelOverride(model="channel-model", provider="channel-provider")})}
    assert adapter.get_session_model_settings(source)["model"] == "channel-model"
    assert adapter.get_session_model_settings(source)["provider"] == "channel-provider"
    key = runner._session_key_for_source(source)
    runner._session_state(key).conversation.model_override = {"model": "session-model"}
    result = adapter.get_session_model_settings(source)
    assert result["model"] == "session-model"
    assert result["provider"] == "channel-provider"


@pytest.mark.parametrize("model, provider", [
    (None, "channel-provider"),
    ("channel-model", None),
])
def test_channel_model_and_provider_defaults_are_independent(settings_env, model, provider):
    from gateway.config import ChannelOverride

    adapter, runner, source, home = settings_env
    (home / "config.yaml").write_text(
        "model:\n  default: global-model\n  provider: global-provider\n  base_url: https://global.example/v1\n"
    )
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(channel_overrides={
        source.chat_id: ChannelOverride(model=model, provider=provider)})}
    key = runner._session_key_for_source(source)
    assert adapter.get_session_model_settings(source) == {
        "session_key": key, "model": model or "global-model",
        "provider": provider or "global-provider", "base_url": "https://global.example/v1",
    }
    runner._session_state(key).conversation.model_override = {"provider": "session-provider"}
    result = adapter.get_session_model_settings(source)
    assert result["model"] == (model or "global-model")
    assert result["provider"] == "session-provider"
    assert result["base_url"] == "https://global.example/v1"


def test_pinned_profile_config_and_source_are_isolated(settings_env):
    from gateway.session_identity import RoutingIdentity

    adapter, runner, source, home = settings_env
    secondary = home / "secondary"
    secondary.mkdir()
    (home / "config.yaml").write_text("model: primary-model\n")
    (secondary / ".env").write_text("UI_MODEL=secondary-model\n")
    (secondary / "config.yaml").write_text("model:\n  default: ${UI_MODEL}\n  provider: secondary-provider\n")
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="secondary",
        authorization_home=home, runtime_home=secondary,
    )
    before = vars(source).copy()
    result = adapter.get_session_model_settings(source)
    assert result["model"] == "secondary-model"
    assert result["provider"] == "secondary-provider"
    assert result["session_key"].startswith("agent:secondary:")
    assert vars(source) == before
    assert runner._peek_session_state(result["session_key"]) is None


def test_read_does_not_allocate_or_rehydrate_or_resolve_credentials(settings_env, monkeypatch):
    import gateway.run as gateway_run

    adapter, runner, source, _ = settings_env

    def forbidden(*args, **kwargs):
        raise AssertionError("Settings reads must not prepare runtime state")

    monkeypatch.setattr(runner, "_session_state", forbidden)
    monkeypatch.setattr(runner, "_rehydrate_session_model_override", forbidden)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", forbidden)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", forbidden)
    assert adapter.get_session_model_settings(source)["model"] == ""
    assert "_sessions" not in vars(runner)


def test_rejected_profile_route_has_no_settings(settings_env):
    adapter, runner, source, _ = settings_env
    source.profile_route_rejected = True
    assert adapter.get_session_model_settings(source) is None
    assert "_sessions" not in vars(runner)
