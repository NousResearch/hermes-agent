"""Copilot catalog routing agrees across primary, auxiliary and fallback clients."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def catalog_route(tmp_path, monkeypatch):
    from hermes_cli import auth, models

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "model": {"provider": "copilot", "default": "gpt-5-mini"},
    }), encoding="utf-8")
    monkeypatch.setattr(auth, "resolve_api_key_provider_credentials", lambda provider: {
        "api_key": "resolved-key" if provider == "copilot" else "",
        "base_url": "https://api.githubcopilot.com",
    })
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *args, **kwargs: {})

    def arrange(model, endpoints):
        fetch = Mock(return_value=[{
            "id": model, "supported_endpoints": endpoints,
            "capabilities": {"type": "chat", "limits": {"max_prompt_tokens": 128000}},
        }])
        monkeypatch.setattr(models, "fetch_github_model_catalog", fetch)
        return fetch

    return arrange


@pytest.mark.parametrize("site", ["primary-explicit-key", "primary-resolved-key", "auxiliary", "auto-auxiliary", "fallback"])
@pytest.mark.parametrize(
    ("model", "endpoints", "expected"),
    [
        ("grok-4.7", ["/responses"], "codex_responses"),
        ("new-response-model", ["/responses"], "codex_responses"),
        ("new-chat-model", ["/chat/completions"], "chat_completions"),
        ("gpt-5-mini", ["/chat/completions", "/responses"], "chat_completions"),
        ("gpt-6.1-sol", ["/chat/completions"], "chat_completions"),
    ],
)
def test_catalog_wire_agrees_across_agent_and_auxiliary_routes(catalog_route, monkeypatch, site, model, endpoints, expected):
    from agent import auxiliary_client as aux
    from run_agent import AIAgent

    fetch = catalog_route(model, endpoints)
    if site in {"auxiliary", "auto-auxiliary"}:
        if site == "auto-auxiliary":
            monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda provider: model if provider == "copilot" else "")
            client, resolved = aux._resolve_api_key_provider()
        else:
            client, resolved = aux.resolve_provider_client("copilot", model, explicit_api_key="aux-key")
        assert resolved == model
        assert isinstance(client, aux.CodexAuxiliaryClient) == (expected == "codex_responses")
        return

    agent = AIAgent(
        provider="copilot", model="gpt-4.1" if site == "fallback" else model,
        api_key=None if site == "primary-resolved-key" else "primary-key",
        base_url="https://api.githubcopilot.com", quiet_mode=True, skip_context_files=True, skip_memory=True,
        fallback_model=[{"provider": "copilot", "model": model, "api_key": "fallback-key"}] if site == "fallback" else None,
    )
    try:
        if site == "fallback":
            fetch.reset_mock()
            assert agent._try_activate_fallback()
            assert any(call.kwargs.get("api_key") == "fallback-key" for call in fetch.call_args_list)
        assert agent.model == model
        assert agent.api_mode == expected
    finally:
        agent.close()


@pytest.mark.parametrize("model", ["grok-4.7", "new-response-model"])
def test_init_fallback_uses_its_own_catalog_credential(catalog_route, model):
    from agent.agent_init_fallback import recompute_init_fallback_api_mode

    fetch = catalog_route(model, ["/responses"])
    agent = SimpleNamespace(provider="copilot", model=model, api_mode="chat_completions", _transport_cache={})
    client = SimpleNamespace(api_key="fallback-key", base_url="https://api.githubcopilot.com")
    recompute_init_fallback_api_mode(agent, client)

    assert agent.api_mode == "codex_responses"
    assert fetch.call_args.kwargs["api_key"] == "fallback-key"
