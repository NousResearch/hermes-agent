from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import try_activate_fallback
from agent.error_classifier import FailoverReason
from provider_gateway.config import GatewayConfig
from provider_gateway.runtime import get_circuit_breaker, get_provider_router


class MockAgent:
    def __init__(self) -> None:
        self.provider = "anthropic"
        self.model = "claude-sonnet-4-6"
        self.base_url = "https://api.anthropic.com"
        self.api_mode = "anthropic_messages"
        self.api_key = "key-primary"
        self._fallback_index = 0
        self._fallback_chain = [
            {"provider": "openrouter", "model": "gpt-4o", "base_url": "https://openrouter.ai/api/v1", "api_key": "key-or"},
            {"provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434", "api_key": "key-ollama"},
        ]
        self._provider_gateway_config = GatewayConfig(enabled=False)
        self._rate_limited_until = 0.0
        self._fallback_activated = False
        self._config_context_length = None
        self._client_kwargs = {}
        self.client = MagicMock()
        self._anthropic_client = MagicMock()
        self._anthropic_api_key = "key-primary"
        self._anthropic_base_url = "https://api.anthropic.com"
        self._is_anthropic_oauth = False
        self._primary_runtime = {"provider": "anthropic"}

    def _is_azure_openai_url(self, url: str) -> bool:
        return False

    def _is_direct_openai_url(self, url: str) -> bool:
        return False

    def _provider_model_requires_responses_api(self, model: str, provider: str) -> bool:
        return False

    def _anthropic_prompt_cache_policy(self, **kwargs) -> tuple[bool, str]:
        return False, "none"

    def _ensure_lmstudio_runtime_loaded(self) -> None:
        pass

    def _buffer_status(self, message: str) -> None:
        pass


@patch("agent.auxiliary_client.resolve_provider_client")
def test_fallback_standard_linear_when_gateway_disabled(mock_resolve) -> None:
    """Test that try_activate_fallback remains linear when gateway is disabled."""
    agent = MockAgent()
    agent._provider_gateway_config = GatewayConfig(enabled=False)

    fake_client = MagicMock()
    fake_client.api_key = "key-or"
    fake_client.base_url = "https://openrouter.ai/api/v1"
    mock_resolve.return_value = (fake_client, "gpt-4o")

    # Call fallback
    success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)

    assert success is True
    assert agent._fallback_index == 1
    # Standard linear selection selected the first item (openrouter/gpt-4o)
    assert agent.provider == "openrouter"
    assert agent.model == "openai/gpt-4o"
    assert agent.base_url == "https://openrouter.ai/api/v1"


@patch("agent.auxiliary_client.resolve_provider_client")
def test_fallback_dynamic_routing_when_gateway_enabled(mock_resolve) -> None:
    """Test that try_activate_fallback routes intelligently when gateway is enabled."""
    agent = MockAgent()
    agent._provider_gateway_config = GatewayConfig(
        enabled=True,
        routing_strategy="lowest-cost",
    )

    # Let's seed Circuit Breaker with latencies
    breaker = get_circuit_breaker(agent)
    breaker.record_success("openrouter", latency_ms=400.0)
    breaker.record_success("ollama", latency_ms=50.0)

    fake_client = MagicMock()
    fake_client.api_key = "key-ollama"
    fake_client.base_url = "http://localhost:11434"
    mock_resolve.return_value = (fake_client, "llama3.2")

    # With lowest-cost (or round-robin skipping to free Ollama), dynamic routing should prefer Ollama!
    success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)

    assert success is True
    # The dynamic router selected "ollama" (index 2 in chain, so index becomes 2)
    assert agent.provider == "ollama"
    assert agent.model == "llama3.2"
    assert agent.base_url == "http://localhost:11434"
    assert agent._fallback_index == 2  # Aligned to second item in fallback chain


@patch("agent.auxiliary_client.resolve_provider_client")
def test_fallback_skips_circuit_open_provider(mock_resolve) -> None:
    """Test that fallback skips a provider if its circuit breaker is OPEN."""
    agent = MockAgent()
    agent._provider_gateway_config = GatewayConfig(
        enabled=True,
        routing_strategy="round-robin",
    )

    # Trip circuit for openrouter (the first fallback candidate)
    breaker = get_circuit_breaker(agent)
    for _ in range(5):
        breaker.record_failure("openrouter")
    assert breaker.is_available("openrouter") is False

    fake_client = MagicMock()
    fake_client.api_key = "key-ollama"
    fake_client.base_url = "http://localhost:11434"
    mock_resolve.return_value = (fake_client, "llama3.2")

    # Try fallback. Since openrouter is circuit-open, it should skip it and select ollama!
    success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)

    assert success is True
    assert agent.provider == "ollama"
    assert agent.model == "llama3.2"
    assert agent._fallback_index == 2
