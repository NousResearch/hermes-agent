"""Integration coverage for the turn-scoped pre_model_route contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import run_agent


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
            completion_tokens=2,
            total_tokens=12,
        ),
        model="test-response-model",
    )


def test_route_first_turn_then_restore_primary_for_second_api_call(monkeypatch):
    """A routed turn must not replace the next turn's provider/model."""
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **_kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "t",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 32768,
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"providers": {}})
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers",
        lambda _config: {},
    )

    route_results = iter(
        [
            [
                {
                    "model": "route-model",
                    "provider": "custom",
                    "reason": "integration test",
                }
            ],
            [],
        ]
    )

    def _invoke_hook(name, **_kwargs):
        if name == "pre_model_route":
            return next(route_results)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            new_model="route-model",
            target_provider="custom",
            api_key="route-key",
            base_url="https://route.example/v1",
            api_mode="chat_completions",
        ),
    )

    api_calls = []

    agent = run_agent.AIAgent(
        model="primary-model",
        api_key="primary-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        skip_context_files=True,
        skip_memory=True,
        max_iterations=4,
    )
    agent._session_db = None
    agent._session_db_created = True
    agent._credential_pool = None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._disable_streaming = True

    def _call(api_kwargs):
        api_calls.append(
            (agent.provider, agent.model, api_kwargs.get("model"))
        )
        return _response()

    agent._interruptible_api_call = _call
    primary_runtime = dict(agent._primary_runtime)
    primary_cache = "primary-cache-sentinel"
    agent._cached_system_prompt = primary_cache

    first = agent.run_conversation("route this turn")

    assert api_calls[0] == ("custom", "route-model", "route-model")
    assert agent._primary_runtime == primary_runtime
    assert agent._pre_model_route_restore_state is not None

    agent.run_conversation(
        "use the default route",
        conversation_history=first["messages"],
    )

    assert api_calls[1] == ("openrouter", "primary-model", "primary-model")
    assert agent.provider == "openrouter"
    assert agent.model == "primary-model"
    assert agent._primary_runtime == primary_runtime
    assert agent._cached_system_prompt == primary_cache
    assert agent._pre_model_route_restore_state is None
