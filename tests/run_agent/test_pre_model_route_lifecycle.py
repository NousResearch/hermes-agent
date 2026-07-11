"""End-to-end turn-lifecycle coverage for the pre_model_route hook."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_agent
from hermes_cli.model_switch import ModelSwitchResult


def _make_agent(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **_kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock(return_value=MagicMock()))

    agent = run_agent.AIAgent(
        model="old-model",
        provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=2,
    )
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._disable_streaming = True
    return agent


def _response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="ok",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
        ),
        model="routed-model",
    )


def test_pre_model_route_runs_once_before_prompt_and_first_request(monkeypatch):
    agent = _make_agent(monkeypatch)
    events = []
    route_calls = []

    def invoke_hook(name, **kwargs):
        if name != "pre_model_route":
            return []
        events.append("hook")
        route_calls.append(kwargs)
        return [{"model": "routed-model", "provider": "routed-provider"}]

    resolved = ModelSwitchResult(
        success=True,
        new_model="routed-model",
        target_provider="routed-provider",
        api_key="resolved-key",
        base_url="https://routed.example/v1",
        api_mode="chat_completions",
    )

    def apply_switch(**kwargs):
        events.append("switch")
        agent.model = kwargs["new_model"]
        agent.provider = kwargs["new_provider"]
        agent.base_url = kwargs["base_url"]
        agent.api_mode = kwargs["api_mode"]
        agent._cached_system_prompt = None

    def build_prompt(target, _system_message, _history):
        events.append(("prompt", target.model, target.provider))
        target._cached_system_prompt = f"SYSTEM:{target.model}"

    def provider_request(_kwargs):
        events.append((
            "request",
            agent.model,
            agent.provider,
            agent._cached_system_prompt,
        ))
        return _response()

    agent._cached_system_prompt = "SYSTEM:old-model"
    agent.switch_model = MagicMock(side_effect=apply_switch)
    agent._interruptible_api_call = provider_request

    with (
        patch(
            "hermes_cli.plugins.has_hook",
            side_effect=lambda name: name == "pre_model_route",
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=invoke_hook),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.model_switch.switch_model", return_value=resolved),
        patch(
            "agent.conversation_loop._restore_or_build_system_prompt",
            side_effect=build_prompt,
        ),
    ):
        result = agent.run_conversation("route this turn")

    assert result["final_response"] == "ok"
    assert len(route_calls) == 1
    assert route_calls[0]["is_first_turn"] is True
    assert events[:4] == [
        "hook",
        "switch",
        ("prompt", "routed-model", "routed-provider"),
        ("request", "routed-model", "routed-provider", "SYSTEM:routed-model"),
    ]
    agent.switch_model.assert_called_once_with(
        new_model="routed-model",
        new_provider="routed-provider",
        api_key="resolved-key",
        base_url="https://routed.example/v1",
        api_mode="chat_completions",
        prune_fallback_chain=False,
    )
