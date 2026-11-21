"""User-visible interrupt reasons must never be the bare placeholder alone."""

from agent.interrupt_control import (
    BARE_INTERRUPT_PLACEHOLDER,
    ensure_interrupt_final_response,
    format_interrupt_user_text,
    is_bare_interrupt_text,
)


def test_bare_placeholder_detected():
    assert is_bare_interrupt_text(None)
    assert is_bare_interrupt_text("")
    assert is_bare_interrupt_text(BARE_INTERRUPT_PLACEHOLDER)
    assert is_bare_interrupt_text("Operation interrupted")
    assert not is_bare_interrupt_text("Operation interrupted: stopped by user.")
    assert not is_bare_interrupt_text("partial answer already streamed")


def test_format_never_bare_without_inputs():
    text = format_interrupt_user_text(session_id="sess-abc")
    assert text.startswith("Operation interrupted:")
    assert "sess-abc" in text
    assert not is_bare_interrupt_text(text)


def test_format_prefers_existing_reasoned_response():
    existing = "Operation interrupted: waiting for model response (1.2s elapsed)."
    assert format_interrupt_user_text(final_response=existing) == existing


def test_format_uses_tool_reason_and_exit_reason():
    text = format_interrupt_user_text(
        tool_reason="terminal batch timeout",
        turn_exit_reason="interrupted_by_system(terminal_batch_timeout)",
    )
    assert text.startswith("Operation interrupted:")
    assert "terminal batch timeout" in text.lower() or "system stop" in text.lower()
    assert not is_bare_interrupt_text(text)


def test_format_user_stop_reads_as_user_stop():
    text = format_interrupt_user_text(
        tool_reason="explicit stop requested",
        turn_exit_reason="interrupted_by_user",
    )
    assert "explicit stop requested" in text or "stopped by user" in text
    assert text.startswith("Operation interrupted:")


def test_ensure_interrupt_final_response_fills_empty():
    class _Agent:
        _tool_interrupt_reason = "explicit stop requested"
        _interrupt_message = None
        session_id = "s1"

    out = ensure_interrupt_final_response(_Agent(), "", turn_exit_reason="interrupted_by_user")
    assert out.startswith("Operation interrupted:")
    assert not is_bare_interrupt_text(out)


def test_ensure_keeps_partial_assistant_prose():
    class _Agent:
        _tool_interrupt_reason = "explicit stop requested"
        _interrupt_message = None
        session_id = "s1"

    partial = "Here is what I found before stop."
    assert ensure_interrupt_final_response(_Agent(), partial) == partial
