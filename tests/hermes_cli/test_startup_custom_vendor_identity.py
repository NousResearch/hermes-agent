"""A selected named endpoint owns its vendor-prefixed model ID (#123997)."""
import os
import subprocess
import sys

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


def test_cli_keeps_selected_custom_vendor(custom_config):
    subprocess.run(
        [sys.executable, "-c", "from types import SimpleNamespace; import cli; "
         "from hermes_cli.cli_init_mixin import CLIInitMixin; instance = SimpleNamespace(); "
         "CLIInitMixin._init_model_and_provider(instance, None, None, None, None); "
         "assert (instance.requested_provider, instance.model) == "
         "('custom:nvidia', 'nvidia/fixture-model')"],
        check=True, capture_output=True, text=True, env=os.environ.copy(),
    )


def test_tui_keeps_selected_custom_vendor(custom_config):
    env = os.environ.copy()
    env["HERMES_INFERENCE_MODEL"] = custom_config["model"]["default"]
    env.pop("HERMES_TUI_PROVIDER", None)
    env.pop("HERMES_INFERENCE_PROVIDER", None)
    subprocess.run(
        [sys.executable, "-c", "from tui_gateway import server; "
         "assert server._resolve_startup_runtime() == "
         "('nvidia/fixture-model', 'custom:nvidia')"],
        check=True, capture_output=True, text=True, env=env,
    )


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


def test_selected_custom_provider_accepts_its_display_name_alias():
    providers = {
        "endpoint-key": {
            "name": "vendor-alias",
            "base_url": "https://custom.example.invalid/v1",
        }
    }
    assert resolve_startup_model_route(
        "vendor-alias/fixture-model",
        current_provider="custom:endpoint-key",
        user_providers=providers,
    ) == StartupModelRoute("vendor-alias/fixture-model", "custom:endpoint-key")


@pytest.mark.parametrize("reverse_order", [False, True])
def test_exact_configured_key_wins_over_selected_display_alias(reverse_order):
    providers = {
        "foo": {"name": "bar", "base_url": "https://foo.example.invalid/v1"},
        "bar": {"base_url": "https://bar.example.invalid/v1"},
    }
    if reverse_order:
        providers = dict(reversed(list(providers.items())))
    assert resolve_startup_model_route(
        "bar/fixture-model", current_provider="custom:foo", user_providers=providers,
    ) == StartupModelRoute("fixture-model", "bar")


@pytest.mark.parametrize("url_key", ["base_url", "url", "api"])
def test_selected_legacy_custom_vendor_alias_keeps_runtime_identity(url_key, tmp_path, monkeypatch):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: custom:custom:vendor-alias\n  default: vendor-alias/fixture-model\n"
        "providers:\n  vendor-alias:\n    api: https://foreign.example.invalid/v1\n"
        "custom_providers:\n  - name: custom:vendor-alias\n"
        f"    {url_key}: https://legacy.example.invalid/v1\n"
        "    api_key: legacy-fixture-key\n", encoding="utf-8",
    )
    cfg = load_config()
    route = resolve_startup_model_route(
        cfg["model"]["default"], current_provider=cfg["model"]["provider"],
        user_providers=cfg["providers"], custom_providers=cfg["custom_providers"],
    )
    assert route == StartupModelRoute("vendor-alias/fixture-model", "custom:custom:vendor-alias")
    assert route is not None
    runtime = resolve_runtime_provider(requested=route.provider)
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == "https://legacy.example.invalid/v1"
    assert runtime["api_key"] == "legacy-fixture-key"


@pytest.mark.parametrize("provider", ["nvidia", "openai"])
@pytest.mark.parametrize("current_kind", ["selected_custom", "foreign_custom", "builtin"])
def test_tuning_only_provider_routes_to_builtin(provider, current_kind):
    current_provider = {
        "selected_custom": f"custom:{provider}", "foreign_custom": "custom:other", "builtin": provider,
    }[current_kind]
    assert resolve_startup_model_route(
        f"{provider}/fixture-model", current_provider=current_provider,
        user_providers={provider: {"stale_timeout_seconds": 600}},
    ) == StartupModelRoute("fixture-model", provider)


@pytest.mark.parametrize("current_provider", ["custom:openai", "custom:other", "openai"])
def test_tuning_only_openai_does_not_keep_invalid_custom_aggregator(current_provider, monkeypatch):
    monkeypatch.setattr("hermes_cli.models._find_openrouter_slug", lambda raw: raw)
    assert resolve_startup_model_route(
        "openai/gpt-4o", current_provider=current_provider,
        user_providers={"openai": {"stale_timeout_seconds": 600}},
    ) == StartupModelRoute("gpt-4o", "openai")


def test_foreign_configured_slash_route_still_switches(custom_config):
    assert resolve_startup_model_route(
        "nvidia/fixture-model", current_provider="custom:other",
        user_providers=custom_config["providers"],
    ) == StartupModelRoute("fixture-model", "nvidia")
