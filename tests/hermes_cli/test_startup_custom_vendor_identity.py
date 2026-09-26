"""A selected named endpoint owns its vendor-prefixed model ID (#123997)."""
import pytest

from hermes_cli.model_switch import StartupModelRoute, resolve_startup_model_route


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    from hermes_cli.config import load_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: custom:nvidia\n  default: nvidia/fixture-model\n"
        "providers:\n  nvidia:\n    api: https://custom.example.invalid/v1\n"
        "    api_key: fixture-only-key\n", encoding="utf-8"
    )
    return load_config()


def test_selected_custom_vendor_keeps_runtime_identity_and_full_id(custom_config):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = custom_config
    raw = cfg["model"]["default"]
    selected = cfg["model"]["provider"]
    route = resolve_startup_model_route(
        raw, current_provider=selected, user_providers=cfg["providers"]
    )
    assert route == StartupModelRoute(raw, selected)
    runtime = resolve_runtime_provider(requested=route.provider)
    # Runtime uses the custom wire adapter; endpoint/key retain the named entry.
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == cfg["providers"]["nvidia"]["api"]
    assert runtime["api_key"] == "fixture-only-key"


def test_oneshot_keeps_selected_custom_vendor(custom_config, monkeypatch):
    from hermes_cli.oneshot import _resolve_model_and_provider

    monkeypatch.setenv("HERMES_INFERENCE_MODEL", custom_config["model"]["default"])
    choice = _resolve_model_and_provider(custom_config, None, None)
    assert (choice.provider, choice.model) == ("custom:nvidia", "nvidia/fixture-model")


def test_cli_and_tui_keep_selected_custom_vendor(custom_config, monkeypatch):
    from types import SimpleNamespace
    import cli
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from tui_gateway import server

    monkeypatch.setattr(cli, "CLI_CONFIG", custom_config)
    instance = SimpleNamespace()
    CLIInitMixin._init_model_and_provider(instance, None, None, None, None)
    assert (instance.requested_provider, instance.model) == (
        "custom:nvidia", "nvidia/fixture-model"
    )
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", custom_config["model"]["default"])
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    assert server._resolve_startup_runtime() == ("nvidia/fixture-model", "custom:nvidia")


@pytest.mark.parametrize("raw", ["", "fixture-model", "nvidia/", "/fixture-model"])
def test_empty_or_unqualified_input_keeps_caller_fallback(raw, custom_config):
    assert resolve_startup_model_route(
        raw, current_provider="custom:nvidia", user_providers=custom_config["providers"]
    ) is None


def test_explicit_provider_and_colon_selection_still_win(custom_config):
    providers = custom_config["providers"]
    assert resolve_startup_model_route(
        "nvidia/fixture-model", explicit_provider="other", current_provider="custom:nvidia",
        user_providers=providers,
    ) is None
    assert resolve_startup_model_route(
        "custom:nvidia:other-model", current_provider="custom:nvidia", user_providers=providers,
    ) == StartupModelRoute("other-model", "custom:nvidia")


def test_aggregator_probe_failure_does_not_lose_selected_custom(custom_config, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr("hermes_cli.providers.is_routing_aggregator", unavailable)
    assert resolve_startup_model_route(
        "nvidia/fixture-model", current_provider="custom:nvidia",
        user_providers=custom_config["providers"],
    ) == StartupModelRoute("nvidia/fixture-model", "custom:nvidia")


def test_foreign_configured_slash_route_still_switches(custom_config):
    assert resolve_startup_model_route(
        "nvidia/fixture-model", current_provider="custom:other",
        user_providers=custom_config["providers"],
    ) == StartupModelRoute("fixture-model", "nvidia")
