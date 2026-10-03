"""Repro for the 2026-09-22 YS-OH-CORE review on #84236.

An explicit ``stop_kind="user_stop"`` that also carries a diagnostic
message (``_emit_interrupted_session_end`` calls
``agent.interrupt("keyboard interrupt", stop_kind="user_stop")``) must NOT
be mistaken for an incoming-message redirect: the turn should still get
the visible "⚡ Turn interrupted" closing message.  Only an untyped
interrupt (no stop_kind) is a redirect and stays silent.
"""

from tests.agent.test_turn_finalizer_interrupted_fallback import (
    _StubAgent,
    _killed_tool_transcript,
    _finalize,
)


def test_user_stop_with_diagnostic_message_is_not_a_redirect():
    agent = _StubAgent(stop_kind="user_stop")
    agent._interrupt_message = "keyboard interrupt"
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert isinstance(result["final_response"], str) and result["final_response"].strip(), (
        "explicit user_stop with a message was silently swallowed "
        f"(final_response={result['final_response']!r})"
    )
    assert "interrupt" in result["final_response"].lower()
    assert "disconnect" not in result["final_response"].lower()


def test_untyped_interrupt_message_stays_silent_redirect():
    agent = _StubAgent(stop_kind=None)
    agent._interrupt_message = "queued incoming message"
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert not result["final_response"], (
        "a genuine redirect must stay silent, got "
        f"{result['final_response']!r}"
    )


def test_user_stop_without_message_still_gets_fallback():
    agent = _StubAgent(stop_kind="user_stop")
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert "stopped" in (result["final_response"] or "").lower()


def test_client_disconnect_with_message_keeps_disconnect_wording():
    agent = _StubAgent(stop_kind="client_disconnect")
    agent._interrupt_message = "SSE client disconnected"
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert "disconnect" in (result["final_response"] or "").lower()


def test_partial_response_still_never_clobbered():
    agent = _StubAgent(stop_kind="user_stop")
    agent._interrupt_message = "keyboard interrupt"
    result = _finalize(
        agent, _killed_tool_transcript(), final_response="partial streamed answer"
    )
    assert result["final_response"] == "partial streamed answer"
