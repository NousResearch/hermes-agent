"""Provider-neutral runtime routing contracts for ACP sessions."""

from hermes_cli.auth import AuthError
import pytest


class _Agent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_acp_auth_fallback_keeps_each_route_runtime_and_model(monkeypatch, tmp_path):
    """ACP must not turn a plan-backed Claude route into native Anthropic."""

    from acp_adapter.session import SessionManager
    from hermes_cli import config as config_mod
    from hermes_cli import runtime_provider
    import run_agent

    primary = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
        "runtime": "claude_agent_sdk",
    }
    fallback = {
        "provider": "openai-codex",
        "model": "gpt-5.4",
        "runtime": "codex_app_server",
    }
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AuthError("Claude subscription unavailable", provider="anthropic")
        return {
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "api_mode": "codex_responses",
            "runtime": "codex_app_server",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "subscription-token",
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"model": primary, "fallback_providers": [fallback]},
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    agent = SessionManager()._make_agent(
        session_id="acp-claude", cwd=str(tmp_path)
    )

    assert calls == [
        {
            "requested": "anthropic",
            "target_model": "claude-opus-4-6",
            "route_config": primary,
        },
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": fallback,
        },
    ]
    assert agent.kwargs["model"] == "gpt-5.4"
    assert agent.kwargs["runtime"] == "codex_app_server"
    assert agent.kwargs["fallback_model"] == [fallback]


def test_acp_explicit_external_runtime_resolution_fails_closed(monkeypatch, tmp_path):
    from acp_adapter.session import SessionManager
    from hermes_cli import config as config_mod
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "model": {
                "default": "claude-opus-4-6",
                "provider": "anthropic",
                "runtime": "claude_agent_sdk",
            }
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad runtime")),
    )

    with pytest.raises(ValueError, match="bad runtime"):
        SessionManager()._make_agent(
            session_id="acp-fail-closed", cwd=str(tmp_path)
        )


def test_acp_explicit_alternate_provider_does_not_inherit_claude_runtime(
    monkeypatch, tmp_path
):
    from acp_adapter.session import SessionManager
    from hermes_cli import config as config_mod
    from hermes_cli import runtime_provider
    import run_agent

    calls = []
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "model": {
                "default": "claude-opus-4-6",
                "provider": "anthropic",
                "runtime": "claude_agent_sdk",
            }
        },
    )

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "runtime": "hermes",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "subscription-token",
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    SessionManager()._make_agent(
        session_id="acp-codex",
        cwd=str(tmp_path),
        model="gpt-5.4",
        requested_provider="openai-codex",
    )

    assert calls == [
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": None,
        }
    ]
