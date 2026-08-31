"""The turn-start preflight timeout surfaces as a typed result, not a traceback.

Companion to the #98424 fail-closed boundary and the #98741/#99710 typed
timeout chain: when ``build_turn_context`` raises
``PreflightCompressionTimedOut``, ``run_conversation`` must convert it into
the same typed recovery dict the in-loop timeout consumers return —
``failed=True`` + ``compression_exhausted=True`` + the actionable message —
because the gateway's generic exception handler deliberately hides raw
exception text from users (it would deliver "Sorry, I encountered an
unexpected error" and skip the clean-session recovery contract).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_turn_start_preflight_timeout_returns_typed_result_not_exception():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor.threshold_tokens = 1
    agent.context_compressor.should_compress = MagicMock(return_value=True)
    agent.context_compressor.should_defer_preflight_to_real_usage = MagicMock(
        return_value=False
    )
    agent.context_compressor.get_active_compression_failure_cooldown = MagicMock(
        return_value=None
    )

    def _timed_out(messages, _system_message, **_kwargs):
        from agent.conversation_compression import (
            mark_context_compression_timed_out,
            reset_context_compression_timeout_outcome,
        )

        reset_context_compression_timeout_outcome(agent)
        mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt

    agent._compress_context = _timed_out

    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old response"},
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        # Must NOT raise PreflightCompressionTimedOut out of run_conversation.
        result = agent.run_conversation("continue", conversation_history=history)

    agent.client.chat.completions.create.assert_not_called()
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["compression_exhausted"] is True
    assert result["turn_exit_reason"] == "context_compression_timeout"
    # The actionable preflight guidance survives into both user-facing fields.
    assert "provider call was not sent" in result["final_response"]
    assert "provider call was not sent" in result["error"]
