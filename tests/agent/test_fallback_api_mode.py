import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.chat_completion_helpers import build_api_kwargs


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_codex_primary_agent(monkeypatch, fallback_entry):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
        fallback_model=[fallback_entry],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._emit_status = lambda *_args, **_kwargs: None
    agent.context_compressor = None
    return agent


def test_fallback_entry_api_mode_override_wins_for_direct_openai(monkeypatch):
    """A Codex primary can fall back to OpenAI Chat Completions.

    Direct api.openai.com normally routes to codex_responses for GPT-5-ish
    models, but fallback_providers entries must be able to pin
    chat_completions so Responses-only encrypted reasoning metadata is not
    sent to chat-only fallback models.
    """
    fallback_entry = {
        "provider": "custom",
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "fallback-key",
        "api_mode": "chat_completions",
    }
    agent = _build_codex_primary_agent(monkeypatch, fallback_entry)

    fallback_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://api.openai.com/v1",
        _custom_headers={},
    )
    resolve_calls = []

    def resolver(*args, **kwargs):
        resolve_calls.append(kwargs)
        return fallback_client, "gpt-4.1-mini"

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolver)

    assert agent._try_activate_fallback() is True
    assert resolve_calls[0]["api_mode"] == "chat_completions"

    assert agent.provider == "custom"
    assert agent.model == "gpt-4.1-mini"
    assert agent.base_url == "https://api.openai.com/v1"
    assert agent.api_mode == "chat_completions"

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])
    assert "input" not in kwargs
    assert "include" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_fallback_without_api_mode_still_uses_direct_openai_responses(monkeypatch):
    """Keep the existing heuristic when no fallback api_mode override exists."""
    fallback_entry = {
        "provider": "custom",
        "model": "gpt-5.1",
        "base_url": "https://api.openai.com/v1",
        "api_key": "fallback-key",
    }
    agent = _build_codex_primary_agent(monkeypatch, fallback_entry)

    fallback_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://api.openai.com/v1",
        _custom_headers={},
    )
    resolve_calls = []

    def resolver(*args, **kwargs):
        resolve_calls.append(kwargs)
        return fallback_client, "gpt-5.1"

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolver)

    assert agent._try_activate_fallback() is True
    assert resolve_calls[0]["api_mode"] is None
    assert agent.api_mode == "codex_responses"


def test_invalid_fallback_api_mode_is_ignored(monkeypatch):
    """Invalid api_mode values should not override normal routing heuristics."""
    fallback_entry = {
        "provider": "custom",
        "model": "gpt-5.1",
        "base_url": "https://api.openai.com/v1",
        "api_key": "***",
        "api_mode": "not-a-real-mode",
    }
    agent = _build_codex_primary_agent(monkeypatch, fallback_entry)

    fallback_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://api.openai.com/v1",
        _custom_headers={},
    )
    resolve_calls = []

    def resolver(*args, **kwargs):
        resolve_calls.append(kwargs)
        return fallback_client, "gpt-5.1"

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolver)

    assert agent._try_activate_fallback() is True
    assert resolve_calls[0]["api_mode"] is None
    assert agent.api_mode == "codex_responses"


def test_context_compressor_receives_fallback_api_mode(monkeypatch):
    """Compression must track the fallback transport, not the primary transport."""
    fallback_entry = {
        "provider": "custom",
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "***",
        "api_mode": "chat_completions",
    }
    agent = _build_codex_primary_agent(monkeypatch, fallback_entry)
    agent.context_compressor = SimpleNamespace(update_model=MagicMock())

    fallback_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://api.openai.com/v1",
        _custom_headers={},
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda *args, **kwargs: (fallback_client, "gpt-4.1-mini"),
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: 128000,
    )

    assert agent._try_activate_fallback() is True
    agent.context_compressor.update_model.assert_called_once()
    assert agent.context_compressor.update_model.call_args.kwargs["api_mode"] == "chat_completions"
