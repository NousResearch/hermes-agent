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


def test_startup_route_aggregator_native_slug_stays_on_aggregator(monkeypatch):
    """On OpenRouter, ``anthropic/claude-...`` is an aggregator-native slug.

    A ``providers.anthropic`` block in the same config must NOT steal the
    route — bare vendor slugs resolve WITHIN the aggregator first
    (aggregator-aware resolution contract).
    """
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    monkeypatch.setattr(
        "hermes_cli.models._find_openrouter_slug",
        lambda name: "anthropic/claude-opus-4.6",
    )
    route = model_switch.resolve_startup_model_route(
        "anthropic/claude-opus-4.6",
        current_provider="openrouter",
        user_providers={"anthropic": {"apiKey": "sk-test"}},
    )
    assert route is None


def test_startup_route_non_aggregator_current_provider_still_routes(monkeypatch):
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "nous/deepseek-v4-pro",
        current_provider="anthropic",
        user_providers={"nous": {"base_url": "https://inference.example/v1"}},
    )
    assert route == model_switch.StartupModelRoute("deepseek-v4-pro", "nous", "")


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


def test_startup_route_url_alias_never_keeps_foreign_provider_label(monkeypatch):
    """A URL-bearing alias labelled ``anthropic`` must resolve as ``custom``.

    Keeping the label would let the alias reach the anthropic
    explicit-runtime branch with a foreign base_url and put the live vendor
    token on the alias host's wire (#28660 / #83612).
    """
    monkeypatch.setattr(
        model_switch,
        "DIRECT_ALIASES",
        {
            "urlalias": model_switch.DirectAlias(
                "qwen3.5:4b", "anthropic", "http://localhost:11434/v1"
            )
        },
    )
    route = model_switch.resolve_startup_model_route("urlalias")
    assert route is not None
    assert route.provider == "custom"
    assert route.base_url == "http://localhost:11434/v1"


def test_startup_route_alias_carries_own_api_key(monkeypatch):
    monkeypatch.setattr(
        model_switch,
        "DIRECT_ALIASES",
        {
            "keyed": model_switch.DirectAlias(
                "some-model",
                "custom",
                "https://proxy.example/v1",
                api_key="sk-alias-key",
            )
        },
    )
    route = model_switch.resolve_startup_model_route("keyed")
    assert route is not None
    assert route.api_key == "sk-alias-key"


def test_startup_route_explicit_provider_wins_over_alias_label(monkeypatch):
    monkeypatch.setattr(
        model_switch,
        "DIRECT_ALIASES",
        {"ds": model_switch.DirectAlias("deepseek-chat", "deepseek", "")},
    )
    route = model_switch.resolve_startup_model_route(
        "ds", explicit_provider="openrouter"
    )
    assert route is not None
    assert route.provider == "openrouter"
    assert route.model == "deepseek-chat"


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
