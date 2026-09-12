"""Tests for stream emission during abnormal turn recoveries (#31448, #31449).

Verifies that when a turn completes via:
1. Guardrail halt
2. Partial stream recovery
3. Prior turn content fallback

The synthesized or recovered final response is delivered through ``stream_delta_callback``
before loop break, so SSE/TUI clients receive the explanation rather than a silent stream
close with zero content delta (indistinguishable from a crash).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_empty_response import recover_empty_response
from agent.turn_recovery import emit_terminal_stream_response


def test_emit_terminal_stream_response_delivers_text_and_eof():
    agent = SimpleNamespace()
    deltas = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    emit_terminal_stream_response(agent, "Test response", safe_print=False)

    assert deltas == ["Test response", None]


def test_emit_terminal_stream_response_noop_on_empty_text():
    agent = SimpleNamespace()
    deltas = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    emit_terminal_stream_response(agent, "", safe_print=False)
    emit_terminal_stream_response(agent, None, safe_print=False)

    assert deltas == []


def test_emit_terminal_stream_response_swallows_callback_exception():
    agent = SimpleNamespace()

    def buggy_callback(d):
        raise RuntimeError("stream broken")

    agent.stream_delta_callback = buggy_callback
    # Should not raise
    emit_terminal_stream_response(agent, "Test response", safe_print=False)


def test_partial_stream_recovery_emits_through_stream_delta_callback():
    """Regression test for #31449: partial_stream_recovery must push recovered
    text through stream_delta_callback before breaking out of the turn loop."""
    agent = MagicMock()
    agent._current_streamed_assistant_text = "<think>pondering</think>Recovered partial stream"
    agent._has_content_after_think_block = lambda text: "Recovered" in text
    agent._strip_think_blocks = lambda text: "Recovered partial stream"

    deltas = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    verdict = recover_empty_response(
        agent,
        assistant_message=SimpleNamespace(content=""),
        response=None,
        finish_reason="stop",
        final_response="",
        messages=[],
        api_messages=[],
        conversation_history=[],
        active_system_prompt="",
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )

    assert verdict.action == "break"
    assert verdict.turn_exit_reason == "partial_stream_recovery"
    assert verdict.final_response == "Recovered partial stream"
    assert deltas == ["Recovered partial stream", None]


def test_fallback_prior_turn_content_emits_through_stream_delta_callback():
    """Regression test for #31449: fallback_prior_turn_content must push prior
    turn content through stream_delta_callback before breaking out of the turn loop."""
    agent = MagicMock()
    agent._current_streamed_assistant_text = ""
    agent._has_content_after_think_block = lambda text: bool(text)
    agent._strip_think_blocks = lambda text: text
    agent._last_content_with_tools = "Previous answer after housekeeping tools"
    agent._last_content_tools_all_housekeeping = True

    deltas = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    verdict = recover_empty_response(
        agent,
        assistant_message=SimpleNamespace(content=""),
        response=None,
        finish_reason="stop",
        final_response="",
        messages=[],
        api_messages=[],
        conversation_history=[],
        active_system_prompt="",
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )

    assert verdict.action == "break"
    assert verdict.turn_exit_reason == "fallback_prior_turn_content"
    assert verdict.final_response == "Previous answer after housekeeping tools"
    assert deltas == ["Previous answer after housekeeping tools", None]
