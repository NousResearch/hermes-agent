"""Regression tests for startup model/provider routing (#87189)."""

from hermes_cli import model_switch


def test_startup_route_uses_configured_nous_provider(monkeypatch):
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "nous/deepseek-v4-pro",
        user_providers={"nous": {"base_url": "https://inference.example/v1"}},
    )
    assert route == model_switch.StartupModelRoute("deepseek-v4-pro", "nous", "")


def test_startup_route_keeps_configured_custom_provider_name(monkeypatch):
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "ollama/qwen3.5:4b",
        user_providers={"ollama": {"base_url": "http://localhost:11434/v1"}},
    )
    assert route == model_switch.StartupModelRoute("qwen3.5:4b", "ollama", "")


def test_startup_route_does_not_consume_aggregator_namespace(monkeypatch):
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "openrouter/anthropic/claude-sonnet",
        user_providers={"openrouter": {"base_url": "https://openrouter.ai/api/v1"}},
    )
    assert route is None


def test_startup_route_resolves_dict_alias_and_preserves_endpoint(monkeypatch):
    monkeypatch.setattr(
        model_switch,
        "DIRECT_ALIASES",
        {
            "localqwen": model_switch.DirectAlias(
                "qwen3.5:4b", "custom", "http://localhost:11434/v1"
            )
        },
    )
    route = model_switch.resolve_startup_model_route("localqwen")
    assert route == model_switch.StartupModelRoute(
        "qwen3.5:4b", "custom", "http://localhost:11434/v1"
    )


def test_model_aliases_dict_entries_are_loaded(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "aliases": {
                    "localqwen": {
                        "model": "qwen3.5:4b",
                        "provider": "custom",
                        "base_url": "http://localhost:11434/v1",
                    }
                }
            }
        },
    )
    aliases = model_switch._load_direct_aliases()
    assert aliases["localqwen"] == model_switch.DirectAlias(
        "qwen3.5:4b", "custom", "http://localhost:11434/v1"
    )