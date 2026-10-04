"""Regression tests for startup model/provider routing (#87189)."""

import pytest

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


def _write_named_provider(tmp_path, monkeypatch):
    """A ``providers:`` entry the user selects by the documented ``custom:<name>:<model>`` form."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: claude-sonnet-4-5\n  provider: anthropic\n"
        "providers:\n  jetson-vllm:\n    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key: EMPTY\n    api_mode: chat_completions\n    models: [nemotron-nano-30b]\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    from hermes_cli.config import load_config
    return load_config()


def test_startup_route_decodes_custom_colon_qualified_model(tmp_path, monkeypatch):
    """``custom:<name>:<model>`` selects the named provider instead of leaving the unsplit string
    to the configured default (#73943: the prompt went to api.anthropic.com before a 404)."""
    cfg = _write_named_provider(tmp_path, monkeypatch)
    route = model_switch.resolve_startup_model_route(
        "custom:jetson-vllm:nemotron-nano-30b", current_provider="anthropic",
        user_providers=cfg.get("providers"))
    assert route == model_switch.StartupModelRoute("nemotron-nano-30b", "custom:jetson-vllm", "")
    # The caller's providers are the only source: without the entry the prefix is bare ``custom``.
    assert model_switch.resolve_startup_model_route(
        "custom:jetson-vllm:nemotron-nano-30b", current_provider="anthropic", user_providers={}
    ).provider == "custom"
    # A colon inside a plain model id is not a provider delimiter.
    assert model_switch.resolve_startup_model_route(
        "anthropic/claude-3.5-sonnet:beta", current_provider="anthropic",
        user_providers=cfg.get("providers")) is None


def test_oneshot_and_tui_qualified_model_never_reaches_default_provider(tmp_path, monkeypatch):
    """``hermes -z -m custom:<name>:<model>`` and ``hermes --tui -m …`` route through the same
    startup owner, so provider auto-detection never hands the qualified string to the configured
    default (#73943)."""
    from hermes_cli.oneshot import _resolve_model_and_provider

    cfg = _write_named_provider(tmp_path, monkeypatch)
    from tui_gateway import server as tui_server  # binds the config path at import: after HERMES_HOME
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *_a, **_k: pytest.fail("auto-detection ran on a provider-qualified model"))
    monkeypatch.setattr(
        "hermes_cli.models.detect_static_provider_for_model",
        lambda *_a, **_k: pytest.fail("auto-detection ran on a provider-qualified model"))
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "custom:jetson-vllm:nemotron-nano-30b")
    choice = _resolve_model_and_provider(cfg, None, None)
    assert (choice.provider, choice.model) == ("custom:jetson-vllm", "nemotron-nano-30b")
    assert tui_server._resolve_startup_runtime() == ("nemotron-nano-30b", "custom:jetson-vllm")
