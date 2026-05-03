"""Tests for plugin-driven pre-turn model routing."""

from unittest.mock import MagicMock, patch

from hermes_cli.model_switch import ModelSwitchResult
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "session-1"
    agent.model = "glm-5.1"
    agent.provider = "zai"
    agent.base_url = "https://api.z.ai/api/coding/paas/v4"
    agent.api_key = "zai-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {
        "api_key": "zai-key",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
    }
    agent.context_compressor = None
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = [
        {"provider": "openai-codex", "model": "gpt-5.4"},
        {"provider": "zai", "model": "glm-5.1"},
    ]
    agent._fallback_model = agent._fallback_chain[0]
    agent._user_id = "user-1"
    agent.platform = "slack"
    return agent


def test_pre_model_route_switches_with_model_switch_pipeline():
    agent = _make_agent()

    route = {
        "provider": "openai-codex",
        "model": "gpt-5.4",
        "reason": "coding task",
    }
    result = ModelSwitchResult(
        success=True,
        new_model="gpt-5.4",
        target_provider="openai-codex",
        api_key="codex-key",
        base_url="https://api.openai.com/v1",
        api_mode="codex_responses",
    )

    with (
        patch("hermes_cli.plugins.invoke_hook", return_value=[route]) as hook,
        patch("hermes_cli.config.load_config", return_value={"providers": {}}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.model_switch.switch_model", return_value=result) as switch,
        patch.object(agent, "_create_openai_client", return_value=MagicMock()),
        patch.object(agent, "_anthropic_prompt_cache_policy", return_value=(False, False)),
        patch.object(agent, "_ensure_lmstudio_runtime_loaded"),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent._apply_pre_model_route_hook(
            "please review this code",
            [{"role": "user", "content": "please review this code"}],
            is_first_turn=True,
        )

    hook.assert_called_once()
    switch.assert_called_once()
    assert switch.call_args.kwargs["raw_input"] == "gpt-5.4"
    assert switch.call_args.kwargs["explicit_provider"] == "openai-codex"
    assert switch.call_args.kwargs["current_provider"] == "zai"
    assert agent.model == "gpt-5.4"
    assert agent.provider == "openai-codex"
    assert agent.api_mode == "codex_responses"
    assert agent._fallback_chain == [
        {"provider": "openai-codex", "model": "gpt-5.4"},
        {"provider": "zai", "model": "glm-5.1"},
    ]


def test_pre_model_route_ignores_missing_model():
    agent = _make_agent()

    with (
        patch("hermes_cli.plugins.invoke_hook", return_value=[{"provider": "openai-codex"}]),
        patch("hermes_cli.model_switch.switch_model") as switch,
    ):
        agent._apply_pre_model_route_hook(
            "hello",
            [{"role": "user", "content": "hello"}],
            is_first_turn=False,
        )

    switch.assert_not_called()
    assert agent.model == "glm-5.1"
    assert agent.provider == "zai"

