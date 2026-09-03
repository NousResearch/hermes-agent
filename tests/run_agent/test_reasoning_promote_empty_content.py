"""Hard-path recovery when thinking models leave content empty.

Qwen/Ollama-style thinking models often put the entire answer in
``reasoning`` / ``reasoning_content`` and stop with blank ``content``.
That used to surface as an abrupt blank TUI reply (sometimes the literal
``(empty)`` sentinel). These tests lock the hard path:

1. ``(empty)`` is never treated as successful visible content.
2. After thinking-prefill is exhausted, full reasoning is promoted to the
   visible reply instead of ending on a blank turn.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tests.run_agent.test_run_agent import _mock_response


def _make_agent(max_iterations: int = 12) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
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
    agent._fallback_chain = []
    return agent


def _thinking_only(text: str = "The answer is 42 because that is the known value."):
    return _mock_response(content="", finish_reason="stop", reasoning=text)


def test_empty_sentinel_is_not_visible_content():
    agent = _make_agent()
    assert agent._has_content_after_think_block("") is False
    assert agent._has_content_after_think_block(None) is False
    assert agent._has_content_after_think_block("(empty)") is False
    assert agent._has_content_after_think_block("  (empty)  ") is False
    assert agent._has_content_after_think_block("real answer") is True


def test_reasoning_promoted_after_prefill_exhaustion():
    """Two thinking-only prefills + one promote nudge + still thinking-only
    → hard-promote full reasoning as the final visible answer."""
    agent = _make_agent()
    reasoning = (
        "Himalaya is misconfigured: no default account. "
        "Tell the user to set default = true and retry."
    )
    agent.client.chat.completions.create.side_effect = [
        _thinking_only(reasoning),  # prefill 1
        _thinking_only(reasoning),  # prefill 2
        _thinking_only(reasoning),  # promote nudge still empty
        _thinking_only(reasoning),  # hard promote uses this reasoning
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("check my email")

    assert result["turn_exit_reason"] == "reasoning_promoted_to_content"
    assert result["final_response"] == reasoning
    assert "(empty)" not in (result["final_response"] or "")
    assert "No reply:" not in (result["final_response"] or "")


def test_echoed_empty_sentinel_does_not_end_turn_as_text_response():
    """If the model echoes the recovery sentinel as content, keep recovering
    instead of treating it as a successful ``text_response``."""
    agent = _make_agent()
    reasoning = "Use himalaya envelope list after fixing the account config."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="(empty)", finish_reason="stop", reasoning=reasoning),
        _thinking_only(reasoning),
        _thinking_only(reasoning),
        _thinking_only(reasoning),
        _thinking_only(reasoning),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue")

    assert result["turn_exit_reason"] == "reasoning_promoted_to_content"
    assert result["final_response"] == reasoning
