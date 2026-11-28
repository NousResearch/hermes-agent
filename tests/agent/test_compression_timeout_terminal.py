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

from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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


def test_pre_api_compression_timeout_is_typed_terminal():
    """Mid-turn pre-API pass that hits the host timeout ends the turn."""
    agent = _make_agent()
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.context_compressor._threshold_tokens = 1
    # The typed terminal is the over-window outcome; a request that fits its window is sent uncompressed.
    agent.context_compressor._resolved_context_length = 1
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


def _post_tool_double(*, abort_on_summary_failure: bool):
    class Compressor:
        last_prompt_tokens = 100
        threshold_tokens = 50

        @staticmethod
        def should_compress(_tokens):
            return True

    Compressor.abort_on_summary_failure = abort_on_summary_failure
    return SimpleNamespace(
        context_compressor=Compressor(),
        compression_enabled=True,
        _clear_context_overflow_warn=lambda: None,
        _safe_print=lambda *_args: None,
        # A host timeout hands the caller's own list back and leaves the timeout mark set.
        _compress_context=lambda messages, *_args, **_kwargs: (messages, "system"),
        _last_compression_timed_out=True,
        _persist_user_message_idx=0,
    )


def _compress_after_tool_results(agent, messages):
    from agent.turn_preflight import compress_after_tool_results

    return compress_after_tool_results(
        agent, messages=messages, system_message="system", user_message="current ask",
        active_system_prompt="system", conversation_history=[], compression_attempts=0,
        max_compression_attempts=3, effective_task_id="task", final_response="",
        turn_exit_reason=None, current_turn_user_idx=0,
    )


def test_post_tool_compression_timeout_ends_turn_under_abort_on_summary_failure():
    """Regression for #160484: with compression.abort_on_summary_failure the post-tool pass must
    stop after a host timeout instead of sending the unchanged request, keeping the tool results."""
    messages = [
        {"role": "user", "content": "current ask"},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "fresh result", "tool_call_id": "1"},
    ]
    before = list(messages)

    verdict = _compress_after_tool_results(_post_tool_double(abort_on_summary_failure=True), messages)

    assert verdict.end_turn is True
    assert verdict.failed is True
    assert verdict.turn_exit_reason == "context_compression_timeout"
    assert "timed out" in verdict.final_response
    assert verdict.messages is messages and messages == before


def test_post_tool_compression_timeout_continues_without_abort_on_summary_failure(monkeypatch):
    """Default config keeps the documented "continuing without compression" outcome."""
    monkeypatch.setattr(
        "agent.turn_preflight.conversation_history_after_compression",
        lambda _agent, _messages, history: history,
    )
    monkeypatch.setattr(
        "agent.conversation_loop._should_skip_model_call_for_reference_handoff",
        lambda _messages, _user_message: False,
    )
    messages = [{"role": "user", "content": "current ask"}]

    verdict = _compress_after_tool_results(_post_tool_double(abort_on_summary_failure=False), messages)

    assert verdict.end_turn is False
    assert verdict.failed is False
    assert verdict.turn_exit_reason is None


def test_post_tool_compression_timeout_fails_the_turn_after_one_tool_round():
    """Regression for #160484 through the real loop: the tool runs once, its result is kept, the
    turn reports the timeout, and no second provider request is sent."""
    tool_def = {"type": "function", "function": {
        "name": "web_search", "description": "s",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_def]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=6,
        )
    tool_call = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )
    tool_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, reasoning_content=None, reasoning=None, tool_calls=[tool_call]),
            finish_reason="tool_calls",
        )],
        model="test/model", usage=None,
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [tool_response, _mock_response("must not be requested")]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False

    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 100
    compressor.context_length = 1_000
    compressor.last_prompt_tokens = 500  # real usage over threshold: the post-tool gate owns the pass
    compressor.abort_on_summary_failure = True
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.side_effect = lambda tokens: tokens >= 100
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""
    agent.compression_enabled = True
    agent.context_compressor = compressor

    def _timed_out(messages, _system_message, **_kwargs):
        from agent.conversation_compression import mark_context_compression_timed_out

        mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt

    executions = []

    def _execute(_assistant_message, messages, *_args):
        executions.append(1)
        messages.append({"role": "tool", "name": "web_search", "tool_call_id": "call_1", "content": "RESULT-1"})

    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=10),
        patch("agent.conversation_loop._estimate_tools_tokens_rough", return_value=0),
        patch.object(agent, "_compress_context", side_effect=_timed_out) as compress,
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do tool work", conversation_history=[])

    assert compress.call_count == 1
    assert compress.call_args.kwargs["trigger"] == "post_tool"
    assert executions == [1]
    assert agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "context_compression_timeout"
    assert any(m.get("role") == "tool" and m.get("content") == "RESULT-1" for m in result["messages"])
