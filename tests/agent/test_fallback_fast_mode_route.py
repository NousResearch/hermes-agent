"""Fast-mode parameters follow the active fallback route (regression for #122010)."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_fast_mode_rederived_across_fallback_chain():
    from hermes_cli.models import resolve_fast_mode_overrides

    entries = [
        {"provider": "custom", "model": "local-model", "base_url": "http://127.0.0.1:8080/v1", "api_key": "k"},
        {"provider": "openai", "model": "gpt-5.4", "base_url": "https://api.openai.com/v1", "api_key": "k"},
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        fallback_model=entries)
    agent.model = "claude-opus-5"
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.service_tier = "priority"
    agent.request_overrides = {"speed": "fast", "temperature": 0.2}

    def resolve(provider, **kwargs):
        client = MagicMock()
        client.base_url = kwargs["explicit_base_url"]
        client.api_key = "k"
        return client, kwargs["model"]

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
    ):
        assert agent._try_activate_fallback()
        assert agent.request_overrides == {"temperature": 0.2}
        assert agent._try_activate_fallback()
        assert agent.request_overrides == {
            "temperature": 0.2,
            **(resolve_fast_mode_overrides(agent.model, provider=agent.provider, base_url=agent.base_url) or {}),
        }
