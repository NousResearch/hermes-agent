"""Regression tests for the background-review aggregate input budget (#93057).

The review fork replays its snapshot on every provider request in its tool
loop. Detached in-memory compaction bounds any SINGLE request; the aggregate
input budget (``_review_input_token_budget``, set by
``_run_review_in_thread`` from ``auxiliary.background_review.max_input_tokens``)
bounds the WHOLE review: the tool loop stops before the provider call that
would cross it, mirroring the iteration-budget exit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call() -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(prompt_tokens: int) -> SimpleNamespace:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call()],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            total_tokens=prompt_tokens + 1,
        ),
    )


def _final_response() -> SimpleNamespace:
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _make_loop_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool_definition()]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=10,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.max_compression_attempts = 1

    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 999_999_999  # never fire compaction here
    compressor.context_length = 1_000_000_000
    compressor.last_prompt_tokens = -1
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""
    agent.compression_enabled = False  # isolate the budget behavior under test
    agent.context_compressor = compressor

    def _fake_execute_tool_calls(assistant_message, messages, *_args):
        tool_call = assistant_message.tool_calls[0]
        messages.append(
            {
                "role": "tool",
                "name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "content": "ok",
            }
        )

    agent._execute_tool_calls = _fake_execute_tool_calls
    return agent


def _run_with_responses(agent, responses):
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do some tool work")
    return result


def test_review_input_budget_stops_tool_loop_before_next_provider_call():
    """Once a fork's cumulative input crosses its budget, no further provider
    call is made — the crossing request completes, then the loop stops."""
    agent = _make_loop_agent()
    agent._review_input_token_budget = 100_000

    responses = [
        _tool_response(50_000),
        _tool_response(50_000),  # cumulative 100_000 -> budget crossed
        _tool_response(50_000),  # must never be consumed
        _final_response(),
    ]
    result = _run_with_responses(agent, responses)

    create = agent.client.chat.completions.create
    assert create.call_count == 2, (
        f"expected the loop to stop after crossing the input budget, "
        f"but {create.call_count} provider calls were made (budget "
        f"{agent._review_input_token_budget}, "
        f"used {agent.session_input_tokens})"
    )
    assert agent.session_input_tokens == 100_000
    assert result["completed"] is False


def test_no_budget_attribute_leaves_tool_loop_unbounded():
    """Agents without ``_review_input_token_budget`` (every normal agent)
    are unaffected by the gate and consume all scripted responses."""
    agent = _make_loop_agent()

    responses = [
        _tool_response(50_000),
        _tool_response(50_000),
        _tool_response(50_000),
        _final_response(),
    ]
    result = _run_with_responses(agent, responses)

    assert agent.client.chat.completions.create.call_count == 4
    assert result["completed"] is True
    assert result["final_response"] == "done"


def test_review_input_budget_exhausted_predicate_edge_cases():
    """The gate only arms for a positive int budget and a real token count."""
    from agent.conversation_loop import _review_input_budget_exhausted

    class _Agent:
        pass

    agent = _Agent()
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = None
    agent.session_input_tokens = 999_999
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = 0
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = -1
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = "100"
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = True
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = 100_000
    agent.session_input_tokens = 99_999
    assert _review_input_budget_exhausted(agent) is False

    agent.session_input_tokens = 100_000
    assert _review_input_budget_exhausted(agent) is True


@pytest.mark.parametrize(
    ("config_value", "expected"),
    [
        ({}, 600_000),
        ({"max_input_tokens": 1_000_000}, 1_000_000),
        ({"max_input_tokens": 0}, None),
        ({"max_input_tokens": -5}, None),
        ({"max_input_tokens": "not-a-number"}, 600_000),
        ({"max_input_tokens": "300000"}, 300_000),
    ],
)
def test_review_input_token_budget_resolution(config_value, expected):
    """Config parsing: default, override, explicit disable, garbage fallback."""
    from agent.background_review import _review_input_token_budget

    assert _review_input_token_budget(config_value) == expected
