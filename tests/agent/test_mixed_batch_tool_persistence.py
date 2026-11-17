"""Tests for tool-call batch persistence ordering with invalid calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_tool_round import run_tool_round
from run_agent import AIAgent


def test_mixed_batch_pre_execution_flush_excludes_invalid_tool_results(monkeypatch):
    """The pre-execution flush must only write the assistant tool-call turn.

    Synthetic error results for invalid batch calls must not be flushed ahead
    of unexecuted valid tools, ensuring an interrupted tool execution does not
    leave asymmetric partial tool results in the session database.
    """
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = (
        "nous",
        "test-model",
        "s1",
        [],
    )
    agent._base_url = "https://openrouter.ai/api/v1"
    agent._base_url_lower = agent._base_url
    agent.api_mode = "chat"
    agent.valid_tool_names = {"valid_tool"}
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.stream_delta_callback = None
    agent._tool_guardrail_halt_decision = None
    agent.compression_enabled = False
    agent.iteration_budget = SimpleNamespace(refund=MagicMock())
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent._emit_interim_assistant_message = MagicMock()
    agent._execute_tool_calls = MagicMock()
    agent._touch_activity = MagicMock()

    valid_tc = SimpleNamespace(
        id="call_valid",
        type="function",
        function=SimpleNamespace(name="valid_tool", arguments="{}"),
    )
    invalid_tc = SimpleNamespace(
        id="call_invalid",
        type="function",
        function=SimpleNamespace(name="non_existent_tool", arguments="{}"),
    )
    assistant_message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[valid_tc, invalid_tc],
    )

    messages = []
    conversation_history = []

    flushed_snapshots = []
    agent._flush_messages_to_session_db = MagicMock(
        side_effect=lambda msgs, *args: flushed_snapshots.append(list(msgs)) or True
    )

    # Mock post-tool compression so it's a simple passthrough
    monkeypatch.setattr(
        "agent.turn_tool_round.compress_after_tool_results",
        lambda *args, **kwargs: SimpleNamespace(
            messages=kwargs.get("messages", messages),
            active_system_prompt="system",
            conversation_history=[],
            compression_attempts=0,
            final_response="",
            turn_exit_reason="continue",
            current_turn_user_idx=0,
            end_turn=False,
        ),
    )

    run_tool_round(
        agent,
        assistant_message=assistant_message,
        finish_reason="tool_calls",
        messages=messages,
        conversation_history=conversation_history,
        api_call_count=1,
        effective_task_id="",
        user_message="run tools",
        system_message=None,
        active_system_prompt="system",
        compression_attempts=0,
        max_compression_attempts=3,
        final_response="",
        failed=False,
        _turn_exit_reason="continue",
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
    )

    # First flush (before tool execution) must only contain the assistant turn
    assert len(flushed_snapshots) >= 1
    first_flush = flushed_snapshots[0]
    assert any(m.get("role") == "assistant" for m in first_flush)
    assert not any(m.get("role") == "tool" for m in first_flush)

    # Final messages list must contain the synthetic tool result for the invalid call
    assert any(
        m.get("role") == "tool" and m.get("tool_call_id") == "call_invalid"
        for m in messages
    )
