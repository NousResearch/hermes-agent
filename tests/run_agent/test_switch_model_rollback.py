"""Regression tests for #33175: ``switch_model`` must restore the agent's
pre-swap state when the client rebuild fails partway through.

Before the fix, ``agent.model`` and ``agent.provider`` were updated several
lines before the client rebuild ran, so any exception during the rebuild
(bad API key, network error, MiniMax OAuth failure, import failure) left
the agent with the *new* model/provider names against the *old* client.
The next turn would hit a model/provider mismatch (e.g. HTTP 400
"claude-sonnet-4-6 is not supported on openai-codex").
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_chat_completions_agent() -> AIAgent:
    """Build a minimal AIAgent for chat_completions tests."""
    agent = AIAgent.__new__(AIAgent)

    agent.model = "claude-sonnet-4-6"
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.api_key = "old-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="old_client")
    agent._client_kwargs = {
        "api_key": "old-key",
        "base_url": "https://api.anthropic.com",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._config_context_length = None
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._create_openai_client = MagicMock(name="_create_openai_client")
    return agent


def _make_anthropic_agent() -> AIAgent:
    """Build a minimal AIAgent currently on chat_completions, about to switch
    into anthropic_messages — the path used by the issue reporter (codex →
    anthropic-style provider)."""
    agent = AIAgent.__new__(AIAgent)

    agent.model = "gpt-5"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "old-codex-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="old_codex_client")
    agent._client_kwargs = {
        "api_key": "old-codex-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._config_context_length = None
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    return agent


def _snapshot(agent: AIAgent) -> dict:
    return {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": agent.api_key,
        "client": agent.client,
        "_client_kwargs": dict(agent._client_kwargs),
        "_anthropic_api_key": agent._anthropic_api_key,
        "_anthropic_base_url": agent._anthropic_base_url,
        "_anthropic_client": agent._anthropic_client,
        "_is_anthropic_oauth": agent._is_anthropic_oauth,
        "_config_context_length": agent._config_context_length,
        "_use_prompt_caching": agent._use_prompt_caching,
        "_use_native_cache_layout": agent._use_native_cache_layout,
    }


def test_chat_completions_client_failure_rolls_back_state():
    """Bad credentials at ``_create_openai_client`` time must leave the
    agent on the old provider, model, base_url, and client — not in a
    torn state with new model/provider against the stale client."""
    agent = _make_chat_completions_agent()
    before = _snapshot(agent)

    agent._create_openai_client = MagicMock(
        name="_create_openai_client",
        side_effect=RuntimeError("bad credentials"),
    )

    with pytest.raises(RuntimeError, match="bad credentials"):
        agent.switch_model(
            new_model="x-ai/grok-4",
            new_provider="openrouter",
            api_key="bad-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    after = _snapshot(agent)
    assert after == before, (
        "switch_model must restore pre-swap state on client-build failure; "
        f"diverged fields: {[k for k in before if before[k] != after[k]]}"
    )


def test_anthropic_client_failure_rolls_back_state():
    """Anthropic-side rebuild failure (e.g. ``build_anthropic_client``
    raising on a malformed token) must roll back to the old provider."""
    agent = _make_anthropic_agent()
    before = _snapshot(agent)

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=ConnectionError("network error"),
    ), patch(
        "agent.anthropic_adapter.resolve_anthropic_token",
        return_value="sk-ant-x",
    ), patch(
        "agent.anthropic_adapter._is_oauth_token",
        return_value=False,
    ):
        with pytest.raises(ConnectionError, match="network error"):
            agent.switch_model(
                new_model="claude-sonnet-4-6",
                new_provider="anthropic",
                api_key="sk-ant-x",
                base_url="https://api.anthropic.com",
                api_mode="anthropic_messages",
            )

    after = _snapshot(agent)
    assert after == before, (
        "switch_model must restore pre-swap state on anthropic-client-build "
        f"failure; diverged fields: {[k for k in before if before[k] != after[k]]}"
    )


def test_successful_switch_commits_new_state():
    """Happy path: when the client rebuild succeeds, the new state must
    persist (no regression from the rollback wrapper)."""
    agent = _make_chat_completions_agent()
    new_client = MagicMock(name="new_client")
    agent._create_openai_client = MagicMock(return_value=new_client)
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()

    agent.switch_model(
        new_model="x-ai/grok-4",
        new_provider="openrouter",
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
    )

    assert agent.model == "x-ai/grok-4"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "or-key"
    assert agent.client is new_client
