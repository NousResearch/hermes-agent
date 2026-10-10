"""Loop-level integration tests for the fallback short-reply stall guard.

Exercises the real ``run_conversation`` loop (mocked LLM client) to prove the
new detector — fallback active + short no-tool reply after tool history —
re-prompts via the bounded continuation path instead of ending the turn.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _response(*, content, finish_reason, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call(call_id):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}
    agent.client = MagicMock()
    return agent


def _run(agent, responses):
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch("model_tools.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("do the full task")


def test_short_reply_on_fallback_continues_instead_of_stalling():
    """Mid-task short reply with fallback active → nudge → work completes."""
    agent = _make_agent()
    agent._provider_fallback_active = True  # fallback answering right now

    result = _run(agent, [
        # 1. model calls a tool (empty content + tool_calls is the normal shape)
        _response(content="", finish_reason="tool_calls",
                  tool_calls=[_tool_call("call1")]),
        # 2. after the tool result: terse stop, no tool calls — the bug shape
        _response(content="Sure.", finish_reason="stop"),
        # 3. continuation nudge lands, model finishes the actual work with a
        #    substantive (>200 chars) answer that the guard must accept as final
        _response(content=(
            "Task fully completed. Here are the details: I listed every file "
            "in the target directory, verified permissions, cross-checked the "
            "manifest against the on-disk state, reconciled two mismatches, "
            "and produced the summary artifact at the agreed location."
        ), finish_reason="stop"),
    ])

    final = "Task fully completed. Here are the details:"
    assert result["final_response"].startswith(final), (
        f"Got: {result['final_response']!r} — detector did not re-prompt "
        f"(turn stalled at the short fallback reply)."
    )
    assert result["api_calls"] == 3, (
        f"Expected 3 API calls (call, short reply, continuation), "
        f"got {result['api_calls']}."
    )


def test_short_reply_without_fallback_ends_turn_normally():
    """Same short reply with NO fallback active → turn ends, no re-prompt."""
    agent = _make_agent()
    agent._provider_fallback_active = False

    result = _run(agent, [
        _response(content="", finish_reason="tool_calls",
                  tool_calls=[_tool_call("call1")]),
        _response(content="Sure.", finish_reason="stop"),
    ])

    assert result["final_response"] == "Sure."
    assert result["api_calls"] == 2


def test_disabled_flag_ends_turn_even_on_fallback():
    """Operator knob off → old (pre-fix) behavior preserved."""
    agent = _make_agent()
    agent._provider_fallback_active = True
    agent.fallback_short_reply_continue = False

    result = _run(agent, [
        _response(content="", finish_reason="tool_calls",
                  tool_calls=[_tool_call("call1")]),
        _response(content="Sure.", finish_reason="stop"),
    ])

    assert result["final_response"] == "Sure."
    assert result["api_calls"] == 2
