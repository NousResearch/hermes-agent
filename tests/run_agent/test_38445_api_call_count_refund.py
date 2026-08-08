"""Regression guard for #38445: ``api_calls`` must not count a failed call.

``run_conversation`` bumps ``api_call_count`` before issuing the request so
attempt numbering is available to logs and hooks during retries. When every
retry fails the loop returns through the max-retries-exhausted path, which
reports ``api_calls`` as-is — so a turn where no call ever succeeded is
logged, displayed and persisted with a count one too high.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _build_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = set()
    agent.api_max_retries = 2
    return agent


def test_exhausted_retries_do_not_count_an_api_call():
    """No call succeeded, so the reported count must stay at zero."""
    agent = _build_agent()
    agent.client = MagicMock()
    # Every attempt fails — a plain connection error is retryable, so the loop
    # burns its retries and exits through the max-retries-exhausted return.
    agent.client.chat.completions.create.side_effect = Exception("Connection error.")

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("time.sleep", return_value=None),
    ):
        result = agent.run_conversation("do the thing")

    assert result["failed"] is True
    assert "API call failed after" in result["final_response"]
    assert result["api_calls"] == 0, (
        f"Expected 0 successful API calls, got {result['api_calls']}. "
        "The optimistic pre-call increment was not refunded, so a turn "
        "where nothing succeeded reports one successful call."
    )
    assert agent._api_call_count == 0, (
        "agent._api_call_count must stay in sync with the refunded value"
    )
