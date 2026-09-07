"""Exercise production gateway routing, with credentials and catalog supplied locally."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli import runtime_provider


@pytest.fixture
def copilot_runtime(monkeypatch):
    config = {"provider": "copilot", "default": "gpt-5.5", "api_mode": "codex_responses"}
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: config)
    monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "test-copilot-token")
    monkeypatch.setattr("hermes_cli.models.fetch_github_model_catalog", lambda **kwargs: [
        {"id": "claude-sonnet-4.6", "supported_endpoints": ["/chat/completions", "/v1/messages"]},
        {"id": "gpt-5.5", "supported_endpoints": ["/responses"]},
    ])
    return config


def make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = SimpleNamespace()
    runner._session_model_overrides = {}
    return runner


@pytest.mark.parametrize("channel_provider", [None, "copilot"])
@pytest.mark.parametrize("default,saved,target,expected", [
    ("gpt-5.5", "codex_responses", "claude-sonnet-4.6", "anthropic_messages"),
    ("claude-sonnet-4.6", "anthropic_messages", "gpt-5.5", "codex_responses"),
])
def test_channel_target_reaches_runtime_resolver(
    monkeypatch, copilot_runtime, channel_provider, default, saved, target, expected,
):
    copilot_runtime.update(default=default, api_mode=saved)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: default)
    monkeypatch.setattr(gateway_run, "_get_channel_override", lambda *a, **kw: SimpleNamespace(
        model=target, provider=channel_provider,
    ))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test-channel")
    model, runtime = make_runner()._resolve_session_agent_runtime(source=source, user_config={})
    assert model == target
    assert runtime["provider"] == "copilot"
    assert runtime["api_mode"] == expected
    assert runtime["api_key"] == "test-copilot-token"


def test_model_only_non_copilot_channel_keeps_runtime(monkeypatch):
    original = {"provider": "openai", "api_mode": "codex_responses", "api_key": "test-key"}
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: "gpt-4o")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: dict(original))
    monkeypatch.setattr(gateway_run, "_get_channel_override", lambda *a, **kw: SimpleNamespace(
        model="gpt-5.5", provider=None,
    ))
    resolver = Mock(side_effect=AssertionError("Non-Copilot model-only override must not re-resolve"))
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", resolver)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test-channel")
    model, runtime = make_runner()._resolve_session_agent_runtime(source=source, user_config={})
    assert model == "gpt-5.5"
    assert runtime == original
    resolver.assert_not_called()


@pytest.mark.parametrize("target,expected", [
    ("claude-sonnet-4.6", "anthropic_messages"), ("gpt-5.5", "codex_responses"),
])
def test_rehydrated_session_routes_persisted_model(monkeypatch, copilot_runtime, target, expected):
    copilot_runtime.update(default="gpt-5.5" if target.startswith("claude") else "claude-sonnet-4.6",
                           api_mode="codex_responses" if target.startswith("claude") else "anthropic_messages")
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: copilot_runtime["default"])
    runner = make_runner()
    runner.session_store = SimpleNamespace(get_model_override=lambda key: {
        "provider": "copilot", "model": target, "base_url": None,
    })
    model, runtime = runner._resolve_session_agent_runtime(session_key="restored-session", user_config={})
    assert model == target
    assert runtime["api_mode"] == expected
    assert runtime["api_key"] == "test-copilot-token"


def test_rehydration_preserves_runtime_metadata(monkeypatch):
    pool = object()
    resolved = {
        "provider": "copilot", "requested_provider": "copilot", "api_key": "test-key",
        "api_mode": "anthropic_messages", "base_url": "https://api.githubcopilot.com",
        "credential_pool": pool, "max_output_tokens": 4096,
        "capabilities": {"vision": True}, "request_overrides": {"temperature": 0.2},
    }
    resolver = Mock(return_value=resolved)
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolver)
    runner = make_runner()
    runner.session_store = SimpleNamespace(get_model_override=lambda key: {
        "provider": "copilot", "model": "claude-sonnet-4.6", "base_url": None,
    })
    runner._rehydrate_session_model_override("restored-session")
    override = runner._session_model_override("restored-session")
    resolver.assert_called_once_with(requested="copilot", target_model="claude-sonnet-4.6")
    assert override["credential_pool"] is pool
    assert override["max_tokens"] == 4096
    assert override["capabilities"] == {"vision": True}
    assert override["request_overrides"] == {"temperature": 0.2}
    assert override["requested_provider"] == "copilot"
    assert override["base_url"] == resolved["base_url"]


def test_fallback_entry_model_reaches_real_resolver(monkeypatch, copilot_runtime):
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {
        "fallback_providers": [{"provider": "copilot", "model": "claude-sonnet-4.6"}],
    })
    runtime = gateway_run._try_resolve_fallback_provider()
    assert runtime["model"] == "claude-sonnet-4.6"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["provider"] == "copilot"


def test_primary_auth_failure_routes_fallback_model(monkeypatch, copilot_runtime):
    from hermes_cli.auth import AuthError

    real_resolve = runtime_provider.resolve_runtime_provider

    def fail_primary(**kwargs):
        if not kwargs.get("requested"):
            raise AuthError("Test primary credentials unavailable")
        return real_resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fail_primary)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: "gpt-5.5")
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {
        "fallback_providers": [{"provider": "copilot", "model": "claude-sonnet-4.6"}],
    })
    model, runtime = make_runner()._resolve_session_agent_runtime(user_config={})
    assert model == "claude-sonnet-4.6"
    assert runtime["api_mode"] == "anthropic_messages"
