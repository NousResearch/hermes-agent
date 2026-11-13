"""Gateway default-session aliases resolve the whole route before the first platform turn."""

from types import SimpleNamespace

from gateway import run
from gateway.run_turn_prepare import GatewayTurnPrepareMixin


def test_platform_default_alias_uses_its_model_endpoint_and_key(monkeypatch):
    """A platform turn must not send the raw alias to model.provider's endpoint."""
    config = {
        "model": {"default": "myalias", "provider": "openrouter"},
        "model_aliases": {
            "myalias": {
                "model": "some-vendor/some-model-id",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9999/v1",
                "key_env": "DEMO_API_KEY",
            },
        },
    }
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr("hermes_cli.runtime_provider._get_model_config", lambda: config["model"])
    requests = []

    def resolve(config_snapshot, **kwargs):
        assert config_snapshot == config
        requests.append(kwargs)
        return {
            "provider": kwargs.get("requested") or "openrouter",
            "base_url": kwargs.get("explicit_base_url") or "https://openrouter.ai/api/v1",
            "api_key": kwargs.get("explicit_api_key") or "sk-default-provider",
            "api_mode": "chat_completions",
        }, None

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_with_fallback", resolve)
    state = SimpleNamespace(conversation=SimpleNamespace(last_resolved_model=""))
    runner = SimpleNamespace(
        _resolve_session_key_or_none=lambda source, key: None,
        _peek_session_state=lambda key: state,
        _session_state=lambda key: state,
    )

    model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(
        runner, user_config=config)
    assert (model, runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
        "some-vendor/some-model-id", "custom", "http://127.0.0.1:9999/v1", "sk-alias-host")
    assert requests == [{
        "requested": "custom",
        "target_model": "some-vendor/some-model-id",
        "explicit_base_url": "http://127.0.0.1:9999/v1",
        "explicit_api_key": "sk-alias-host",
    }]
