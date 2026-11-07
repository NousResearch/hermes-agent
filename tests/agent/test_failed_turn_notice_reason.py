"""Failed-turn boundary copy must name why the turn stopped (not only that it stopped)."""

from agent.turn_failure_copy import (
    FAILED_TURN_NOTICE,
    PARTIAL_FAILED_TURN_NOTICE,
    failed_turn_notice,
    untyped_failed_turn_display_kind,
)


def test_partial_notice_includes_timeout_cause():
    messages = [
        {"role": "user", "content": "run the build"},
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "terminal"}}]},
        {"role": "tool", "content": "timed out"},
    ]
    text = failed_turn_notice(
        messages,
        failure_reason="timeout",
        turn_exit_reason="all_retries_exhausted_no_response",
    )
    assert text.startswith(PARTIAL_FAILED_TURN_NOTICE) or PARTIAL_FAILED_TURN_NOTICE in text
    assert "did not complete" in text.lower()
    # Why: timeout class in plain words — not the bare static notice alone.
    assert text.strip() != PARTIAL_FAILED_TURN_NOTICE
    assert "time" in text.lower() or "timeout" in text.lower() or "respond" in text.lower()


def test_empty_turn_notice_includes_interrupt_cause():
    messages = [{"role": "user", "content": "hello"}]
    text = failed_turn_notice(
        messages,
        failure_reason="host_error",
        turn_exit_reason="interrupted_by_system(gateway_shutdown)",
        tool_reason="gateway shutdown",
    )
    assert "not processed" in text.lower() or text.startswith(FAILED_TURN_NOTICE)
    assert text.strip() != FAILED_TURN_NOTICE
    assert "shutdown" in text.lower() or "system" in text.lower() or "interrupt" in text.lower()


def test_bare_call_without_reason_keeps_legacy_static_copy():
    """Callers that pass no reason still get the stable legacy strings (tests + old paths)."""
    assert failed_turn_notice([{"role": "user", "content": "x"}]) == FAILED_TURN_NOTICE
    assert failed_turn_notice(
        [{"role": "tool", "content": "ok"}]
    ) == PARTIAL_FAILED_TURN_NOTICE


def test_unglossed_loop_error_still_names_a_why():
    """A site code absent from FAILURE_CAUSE_GLOSS must not drop the Why clause."""
    text = failed_turn_notice(
        [{"role": "user", "content": "run it"}],
        failure_reason="loop_error",
    )
    assert text.startswith(FAILED_TURN_NOTICE)
    assert "Why:" in text
    assert text.strip() != FAILED_TURN_NOTICE


def test_unglossed_gateway_connection_still_names_a_why():
    """A second unglossed site code is also kept, not only loop_error."""
    text = failed_turn_notice(
        [{"role": "user", "content": "run it"}],
        failure_reason="gateway_connection",
    )
    assert "Why:" in text
    assert text.strip() != FAILED_TURN_NOTICE


def test_untyped_matcher_accepts_reasoned_suffix():
    base = failed_turn_notice(
        [{"role": "tool", "content": "x"}],
        failure_reason="timeout",
    )
    assert untyped_failed_turn_display_kind("assistant", base) == "failed_turn"
    # A real reply that only quotes the notice must not match.
    assert untyped_failed_turn_display_kind(
        "assistant", f"Quoting Hermes: {PARTIAL_FAILED_TURN_NOTICE}"
    ) is None
