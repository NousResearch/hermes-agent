"""Host compression timeout terminates the turn before provider re-entry (#98722).

Salvaged from #98741 and composed with the already-merged #98424 turn-start
fail-closed boundary:

- #98424 covers the TURN-START preflight (raises
  ``PreflightCompressionTimedOut`` before the loop starts).
- These tests pin the two consumers unique to #98741: the provider-overflow
  recovery path must not re-enter compression / re-send the unchanged request
  once the wait budget was spent, and the mid-turn pre-API pass must end the
  turn with the typed ``compression_exhausted`` recovery contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _make_agent():
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
    return agent


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def test_overflow_recovery_timeout_ends_turn_without_provider_reentry():
    """Provider 400 overflow + host-timed-out compression = typed terminal.

    Before the fix, the timed-out pass was indistinguishable from an
    ordinary no-op: the loop re-sent the unchanged oversized request, the
    provider overflowed again, and compression was re-entered in the same
    turn (#98722 "Summarizing" loop).
    """
    agent = _make_agent()

    err_400 = Exception(
        "This model's maximum context length is 8192 tokens. However, "
        "your messages resulted in 95000 tokens. Please reduce the length "
        "of the messages."
    )
    err_400.status_code = 400
    agent.client.chat.completions.create.side_effect = [err_400, err_400]

    compression_calls = []

    def _timed_out(messages, _system_message, **_kwargs):
        compression_calls.append(1)
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
        result = agent.run_conversation("continue", conversation_history=history)

    # Exactly one doomed send proved the overflow; the timed-out recovery
    # pass must not be followed by a second identical send.
    assert compression_calls == [1]
    assert agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["compression_exhausted"] is True
    assert "No messages were dropped" in result["final_response"]
    assert result["error"] == result["final_response"]


def test_pre_api_compression_timeout_is_typed_terminal():
    """Mid-turn pre-API pass that hits the host timeout ends the turn."""
    agent = _make_agent()
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor._threshold_tokens = 1
    agent.context_compressor.should_compress = MagicMock(return_value=True)
    agent.context_compressor.should_compress_info = MagicMock(
        return_value=(True, None)
    )
    agent.context_compressor.should_defer_preflight_to_real_usage = MagicMock(
        return_value=False
    )
    agent.context_compressor.get_active_compression_failure_cooldown = MagicMock(
        return_value=None
    )

    compression_calls = []

    def _timed_out(messages, _system_message, **_kwargs):
        compression_calls.append(1)
        from agent.conversation_compression import (
            mark_context_compression_timed_out,
            reset_context_compression_timeout_outcome,
        )

        reset_context_compression_timeout_outcome(agent)
        mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt

    agent._compress_context = _timed_out

    from agent.turn_context import PreflightCompressionTimedOut

    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old response"},
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        # The turn-start boundary (#98424) may fire first and raise; both
        # outcomes satisfy the invariant under test: the unchanged oversized
        # request never reaches the provider after a host timeout.
        try:
            result = agent.run_conversation(
                "continue", conversation_history=history
            )
        except PreflightCompressionTimedOut:
            result = None

    assert compression_calls == [1]
    agent.client.chat.completions.create.assert_not_called()
    if result is not None:
        assert result["failed"] is True
        assert result["compression_exhausted"] is True
        assert result["turn_exit_reason"] == "context_compression_timeout"
