"""``-m <alias>`` at session.create is decoded by the owner against the profile's own aliases.

Pinned by the C11 truth table (alias_startup_q_api_key / alias_startup_q_key_env /
alias_startup_oneshot_z): the raw alias name must never become the model sent to the profile's
DEFAULT provider with that provider's key.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ALIAS_HOST_KEY", "sk-alias-env-secret")
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:1/v1\n  default: model-main\n"
        "providers:\n  named-host:\n    base_url: http://127.0.0.1:2/v1\n    model: model-named\n"
        "model_aliases:\n"
        "  alias-host: {model: model-alias, provider: custom, base_url: http://127.0.0.1:3/v1, api_key: sk-alias-literal}\n"
        "  alias-env: {model: model-alias-env, provider: custom, base_url: http://127.0.0.1:3/v1, key_env: ALIAS_HOST_KEY}\n",
        encoding="utf-8")
    from gateway import run as gateway_run
    from hermes_cli import model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    model_switch.DIRECT_ALIASES.clear()
    return hermes_home


def _config():
    from gateway.run import _load_gateway_config
    return _load_gateway_config()


def test_alias_launch_carries_alias_endpoint_and_key(home):
    from gateway.session_local_route import resolve_launch_route
    literal = resolve_launch_route({"model": "alias-host", "source": "cli"}, _config())
    assert (literal["model"], literal["provider"], literal["base_url"], literal["api_key"]) == (
        "model-alias", "custom", "http://127.0.0.1:3/v1", "sk-alias-literal")
    env = resolve_launch_route({"model": "alias-env"}, _config())
    assert (env["model"], env["base_url"], env["api_key"]) == ("model-alias-env", "http://127.0.0.1:3/v1", "sk-alias-env-secret")
    # Qualified `provider:model` selects the named provider; a plain id and an explicit
    # endpoint launch pass through untouched (their route is already decided).
    qualified = resolve_launch_route({"model": "custom:named-host:model-named"}, _config())
    assert (qualified["model"], qualified["provider"]) == ("model-named", "custom:named-host")
    plain = {"model": "model-main", "source": "cli"}
    assert resolve_launch_route(plain, _config()) == plain
    explicit = {"model": "alias-host", "base_url": "http://127.0.0.1:9/v1"}
    assert resolve_launch_route(explicit, _config()) == explicit


def test_alias_key_is_a_launch_key_not_durable_policy(home):
    """The alias's credential rides ``api_key`` (authority memory, revoked by restart), and the
    frozen policy config carries the alias endpoint, never the profile default's."""
    from gateway.session_local_route import resolve_launch_route
    from gateway.session_policy import build_policy
    params = resolve_launch_route({"model": "alias-host", "source": "cli", "cwd": str(home)}, _config())
    policy = build_policy(params, _config(), private_secrets={})
    assert policy.model == "model-alias" and policy.base_url == "http://127.0.0.1:3/v1"
    assert "sk-alias-literal" not in policy.config_json + policy.request_json


def test_alias_launch_explicit_provider_wins_without_alias_credential(home):
    from gateway.session_local_route import resolve_launch_route

    result = resolve_launch_route(
        {"model": "alias-host", "provider": "openrouter"},
        _config(),
    )
    assert result["model"] == "model-alias"
    assert result["provider"] == "openrouter"
    assert result["base_url"] == "http://127.0.0.1:3/v1"
    assert "api_key" not in result


def test_configured_provider_slash_keeps_credential_lookup_key():
    from gateway.session_local_route import resolve_launch_route

    config = {
        "model": {"default": "claude-sonnet-4.6", "provider": "anthropic"},
        "providers": {
            "ollama": {"base_url": "http://localhost:11434/v1"},
        },
    }
    result = resolve_launch_route({"model": "ollama/qwen3.5:4b"}, config)
    assert result["model"] == "qwen3.5:4b"
    assert result["provider"] == "ollama"


def test_aggregator_native_slug_is_not_stolen_by_configured_provider(monkeypatch):
    from gateway.session_local_route import resolve_launch_route

    monkeypatch.setattr(
        "gateway.session_local_route.find_static_provider_model_id",
        lambda provider, model: model if provider == "openrouter" else None,
    )
    params = {"model": "anthropic/claude-opus-4.6"}
    config = {
        "model": {"default": "openai/gpt-5.4", "provider": "openrouter"},
        "providers": {
            "anthropic": {"api_key": "sk-test"},
        },
    }
    assert resolve_launch_route(params, config) == params
