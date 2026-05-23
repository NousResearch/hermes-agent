from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_response(content="ok"):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None, reasoning_content=None, reasoning_details=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


BUDGET_CONFIG = {
    "agent": {
        "max_turns": 32,
        "turn_budgets": {
            "gateway": 32,
            "lightweight_followup": 6,
            "coding": 90,
            "ship_mode": 128,
            "cron": 90,
        },
    }
}


def _agent(platform="discord", configured=32):
    agent = AIAgent.__new__(AIAgent)
    setattr(agent, "platform", platform)
    setattr(agent, "max_iterations", configured)
    setattr(agent, "_configured_max_iterations", configured)
    setattr(agent, "_external_turn_ceiling_active", False)
    setattr(agent, "_last_resolved_turn_max_iterations", configured)
    return agent


def test_serious_app_build_request_gets_ship_mode_guard():
    from agent.ship_mode_guard import build_ship_mode_routing_context

    ctx = build_ship_mode_routing_context(
        "Build this app end-to-end, boot it, smoke-test it, and ship the preview"
    )

    assert ctx
    assert "Ship-mode routing guard" in ctx
    assert "todo" in ctx
    assert "source-of-truth" in ctx
    assert "Kanban" in ctx
    assert "review" in ctx


def test_tiny_fix_does_not_get_ship_mode_guard():
    from agent.ship_mode_guard import build_ship_mode_routing_context

    assert build_ship_mode_routing_context("fix typo in README") == ""
    assert build_ship_mode_routing_context("change the button copy to Continue") == ""
    assert build_ship_mode_routing_context("what should we build next?") == ""


def test_product_feature_ship_request_escalates_to_ship_budget(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()

    result = agent._resolve_turn_max_iterations(
        "Implement the product onboarding feature across UI and API, verify the preview, and don't stop"
    )

    assert result == 128
    assert agent._last_turn_budget_classification == "ship_mode"


def test_tiny_fix_stays_out_of_ship_mode(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()

    result = agent._resolve_turn_max_iterations("fix typo in README")

    assert result != 128
    assert agent._last_turn_budget_classification != "ship_mode"


def test_ship_mode_guard_is_injected_into_api_only_message(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("todo", "delegate_task")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="test",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
        )

    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response("ok")
    agent._cached_system_prompt = "You are helpful."

    result = agent.run_conversation(
        "Build this app end-to-end, boot it, smoke-test it, and ship the preview",
        conversation_history=[],
    )

    sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    sent_user = [m for m in sent_messages if m.get("role") == "user"][-1]["content"]
    assert "Ship-mode routing guard" in sent_user
    assert "Use the todo tool" in sent_user
    assert "source-of-truth" in sent_user

    stored_user = [m for m in result["messages"] if m.get("role") == "user"][-1]["content"]
    assert "Ship-mode routing guard" not in stored_user


def test_kanban_worker_does_not_get_nested_ship_mode_guard(monkeypatch):
    from agent.ship_mode_guard import build_ship_mode_routing_context

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

    assert build_ship_mode_routing_context(
        "Build this app end-to-end and ship it", platform="discord"
    ) == ""
