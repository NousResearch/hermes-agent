"""A plugin claiming a host in ``hostnames`` (catalog / aux routing) never replaces the reasoning gate core
owns for that host (OpenRouter, Ollama Cloud): only hosts no core gate covers follow ``reasoning_payload``."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def claiming_profile(request):
    import providers
    from providers.base import ProviderProfile

    providers.list_providers()  # discovery first, so the fake survives it
    providers.register_provider(ProviderProfile(name="hostclaimer", hostnames=("openrouter.ai", "ollama.com")))

    def _cleanup():
        providers._REGISTRY.pop("hostclaimer", None)
        providers._SOURCES.pop("hostclaimer", None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)


def _agent(provider, base_url, model):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        return AIAgent(api_key="k", base_url=base_url, provider=provider, api_mode="chat_completions", model=model,
                       quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                       reasoning_config={"enabled": True, "effort": "high"})


def test_openrouter_keeps_its_reasoning_gate_and_body_when_a_plugin_claims_the_host(claiming_profile):
    agent = _agent("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5")
    with patch("hermes_cli.models_reasoning_caps.openrouter_model_reasoning_capabilities",
               return_value={"supports_reasoning": True}):
        assert agent._supports_reasoning_extra_body() is True
        extra_body = agent._build_api_kwargs([{"role": "user", "content": "x"}]).get("extra_body") or {}
    assert extra_body.get("reasoning") == {"enabled": True, "effort": "high"}


def test_ollama_cloud_keeps_its_capability_gate_when_a_plugin_claims_the_host(claiming_profile):
    agent = _agent("custom", "https://ollama.com/v1", "gpt-oss:120b")
    with patch.object(type(agent), "_ollama_supports_thinking_cached", return_value=True):
        assert agent._supports_reasoning_extra_body() is True
