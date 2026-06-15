"""Tests for agent.agent_runtime_helpers.switch_api_key."""

from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import switch_api_key
from run_agent import AIAgent


def _make_agent():
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalClient")
    agent._client_kwargs = {"api_key": "or-key-original", "base_url": "https://openrouter.ai/api/v1"}
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None
    return agent


def test_switch_api_key_reuses_switch_model_with_same_provider():
    """switch_api_key keeps model/provider and forwards to switch_model."""
    agent = _make_agent()
    new_client = MagicMock(name="NewClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client

    switch_api_key(agent, "or-key-new")

    assert agent.provider == "openrouter"
    assert agent.model == "x-ai/grok-4"
    assert agent.api_key == "or-key-new"
    assert agent.client is new_client
    assert agent._client_kwargs["api_key"] == "or-key-new"


def test_switch_api_key_changes_provider_when_requested():
    """switch_api_key can move to a different provider with the same model."""
    agent = _make_agent()
    new_client = MagicMock(name="NewClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client

    switch_api_key(agent, "ds-key-new", provider="deepseek")

    assert agent.provider == "deepseek"
    assert agent.model == "x-ai/grok-4"
    assert agent.api_key == "ds-key-new"


def test_switch_api_key_rejects_empty_key():
    """switch_api_key raises ValueError for an empty key."""
    agent = _make_agent()
    with pytest.raises(ValueError, match="api_key is required"):
        switch_api_key(agent, "")
