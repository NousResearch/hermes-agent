"""Tests for Forge-style workflow guardrails."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.workflow_guardrails import (
    WorkflowGuardrailConfig,
    WorkflowGuardrailController,
    append_workflow_advisory,
)
from run_agent import AIAgent


def _mock_response(content="done", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(config: dict | None = None, max_iterations: int = 5) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_workflow_config_defaults_to_legacy_safe_off():
    cfg = WorkflowGuardrailConfig()

    assert cfg.enabled is False
    assert cfg.final_gate_mode == "off"
    assert cfg.max_nudges == 1


def test_workflow_config_parses_modes_and_invalid_values():
    cfg = WorkflowGuardrailConfig.from_mapping({"enabled": False, "final_gate_mode": "nudge", "max_nudges": 2})
    assert cfg.enabled is False
    assert cfg.final_gate_mode == "nudge"
    assert cfg.max_nudges == 2

    fallback = WorkflowGuardrailConfig.from_mapping({"enabled": True, "final_gate_mode": "explode"})
    assert fallback.final_gate_mode == "off"


def test_repo_review_missing_verify_returns_advisory():
    controller = WorkflowGuardrailController(
        WorkflowGuardrailConfig(enabled=True, final_gate_mode="advisory")
    )
    controller.reset_for_turn("다음 GitHub 레포지토리를 검토해줘 https://github.com/example/repo")
    controller.record_tool_result("read_file")
    controller.record_tool_result("search_files")

    decision = controller.evaluate_final_response("review done")

    assert decision.action == "advisory"
    assert decision.workflow_key == "repo_review"
    assert [step.key for step in decision.missing_steps] == ["inspect_metadata", "verify"]
    assert "Run tests or smoke checks" in decision.message


def test_devflow_all_required_steps_allow_final():
    controller = WorkflowGuardrailController()
    controller.reset_for_turn("구현 진행해서 완료해")
    for tool in ["read_file", "patch", "terminal"]:
        controller.record_tool_result(tool)

    decision = controller.evaluate_final_response("done")

    assert decision.action == "allow"
    assert decision.missing_steps == ()


def test_nudge_mode_nudges_once_then_advises():
    controller = WorkflowGuardrailController(
        WorkflowGuardrailConfig(enabled=True, final_gate_mode="nudge", max_nudges=1)
    )
    controller.reset_for_turn("구현 진행해서 완료해")
    controller.record_tool_result("read_file")
    controller.record_tool_result("patch")

    first = controller.evaluate_final_response("done")
    second = controller.evaluate_final_response("done")

    assert first.action == "nudge"
    assert second.action == "advisory"


def test_append_workflow_advisory_adds_footer_only_for_advisory():
    controller = WorkflowGuardrailController(
        WorkflowGuardrailConfig(enabled=True, final_gate_mode="advisory")
    )
    controller.reset_for_turn("구현 진행해서 완료해")
    decision = controller.evaluate_final_response("done")

    text = append_workflow_advisory("done", decision)

    assert text.startswith("done")
    assert "Workflow guardrail advisory" in text


def test_runtime_default_workflow_guardrail_is_off_for_legacy_compatibility():
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response("done")

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("구현 진행해서 완료해")

    assert result["final_response"] == "done"
    assert "Workflow guardrail advisory" not in result["final_response"]
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"


def test_runtime_opt_in_workflow_advisory_preserves_final_response_and_metadata():
    config = {"workflow_guardrails": {"enabled": True, "final_gate_mode": "advisory"}}
    agent = _make_agent(config=config)
    agent.client.chat.completions.create.return_value = _mock_response("done")

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("구현 진행해서 완료해")

    assert result["final_response"].startswith("done")
    assert "Workflow guardrail advisory" in result["final_response"]
    assert result["turn_exit_reason"] == "workflow_guardrail_advisory(devflow)"
    assert result["workflow_guardrail"]["active"] == ["devflow"]


def test_runtime_workflow_nudge_continues_then_allows_after_tool_evidence():
    config = {"workflow_guardrails": {"enabled": True, "final_gate_mode": "nudge", "max_nudges": 1}}
    agent = _make_agent(config=config, max_iterations=4)
    agent.client.chat.completions.create.side_effect = [
        _mock_response("premature final"),
        _mock_response("done"),
    ]
    # Simulate the missing tool evidence being supplied between iterations by
    # the runtime/tool layer. This keeps the test deterministic and focused on
    # final-gate role sequencing.
    original_eval = agent._workflow_guardrails.evaluate_final_response
    calls = {"n": 0}

    def eval_with_evidence(text):
        calls["n"] += 1
        if calls["n"] == 2:
            for tool in ["read_file", "patch", "terminal"]:
                agent._workflow_guardrails.record_tool_result(tool)
        return original_eval(text)

    agent._workflow_guardrails.evaluate_final_response = eval_with_evidence

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("구현 진행해서 완료해")

    assert result["final_response"] == "done"
    roles = [m["role"] for m in result["messages"]]
    assert roles[-3:] == ["assistant", "user", "assistant"]
    assert result["completed"] is True
