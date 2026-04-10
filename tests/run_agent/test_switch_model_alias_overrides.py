"""Tests for AIAgent.switch_model alias-specific overrides."""

from unittest.mock import MagicMock

from agent.context_compressor import ContextCompressor
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "old-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-old"
    agent.api_mode = "openai_compat"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "sk-old", "base_url": "https://openrouter.ai/api/v1"}
    agent._use_prompt_caching = False
    agent._cached_system_prompt = "cached"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._create_openai_client = lambda kwargs, reason, shared: MagicMock()
    agent.context_compressor = ContextCompressor(
        model="old-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-old",
        provider="openrouter",
        quiet_mode=True,
    )
    return agent


def test_switch_model_applies_alias_context_and_max_tokens():
    """Explicit alias overrides should update runtime state in-place."""
    agent = _make_agent()

    agent.switch_model(
        "custom-model:latest",
        "custom",
        api_key="",
        base_url="https://example.com/v1",
        api_mode="openai_compat",
        context_length=204800,
        max_tokens=131072,
    )

    assert agent.model == "custom-model:latest"
    assert agent.provider == "custom"
    assert agent.base_url == "https://example.com/v1"
    assert agent.max_tokens == 131072
    assert agent.context_compressor.context_length == 204800
    assert agent.context_compressor.threshold_tokens == int(204800 * agent.context_compressor.threshold_percent)
    assert agent._primary_runtime["max_tokens"] == 131072
    assert agent._primary_runtime["compressor_context_length"] == 204800
    assert agent._cached_system_prompt is None
