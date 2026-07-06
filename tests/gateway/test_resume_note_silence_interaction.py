"""Resume-note refactor + silence-marker interaction.

Covers three concerns for the Phase-0 auto-continue refactor in
``gateway/run.py``:

1. The extracted note builders (:func:`_build_interruption_system_note`,
   :func:`_build_tool_tail_system_note`) reproduce the historical inline
   f-strings **byte-for-byte** — a golden-string regression guard so a future
   wording tweak is a deliberate, reviewed change, not an accident.
2. The extracted dispatch predicate (:func:`_resolve_resume_note_kind`) is a
   pure truth table, including the "two freshness signals disagree" edge case.
3. Resume/recovery notes and the intentional-silence classifier compose
   correctly in both directions: a recovery report is never swallowed as
   silence, and a literal silence token on a resumed turn is still silence.
"""

import pytest

from gateway.response_filters import (
    LIVE_GATEWAY_SILENT_MARKERS,
    is_intentional_silence_response,
)
from gateway.run import (
    _build_interruption_system_note,
    _build_tool_tail_system_note,
    _resolve_resume_note_kind,
)


# --------------------------------------------------------------------------
# 1. Golden-string guards — the note text pre-refactor, captured verbatim.
# --------------------------------------------------------------------------


def _golden_resume_note(reason, has_new_message):
    """The exact inline f-string the three sites produced before extraction."""
    reason_phrase = (
        "a gateway restart"
        if reason == "restart_timeout"
        else "a gateway shutdown"
        if reason == "shutdown_timeout"
        else "a gateway interruption"
    )
    if has_new_message:
        guidance = (
            "Address the user's NEW message below FIRST and focus "
            "on what the user is asking now."
        )
    else:
        guidance = (
            "Report to the user that the session was restored "
            "successfully and ask what they would like to do next."
        )
    return (
        f"[System note: The previous turn was interrupted by "
        f"{reason_phrase}; the gateway is now back online. "
        f"Any restart/shutdown command in the history has already "
        f"run — do NOT re-execute or verify it. {guidance} "
        f"Do NOT re-execute old tool calls — skip any unfinished "
        f"work from the conversation history.]"
    )


_GOLDEN_TOOL_TAIL_NOTE = (
    "[System note: A new message has arrived. The conversation "
    "history contains pending tool outputs from an interrupted turn. "
    "IGNORE those pending results. Address the user's NEW message "
    "below FIRST. Do NOT re-execute old tool calls from the history.]"
)


@pytest.mark.parametrize(
    "reason",
    ["restart_timeout", "shutdown_timeout", "unknown_reason", None],
)
@pytest.mark.parametrize("has_new_message", [True, False])
def test_interruption_note_is_byte_identical(reason, has_new_message):
    assert _build_interruption_system_note(
        reason, has_new_message
    ) == _golden_resume_note(reason, has_new_message)


def test_unknown_and_missing_reason_map_to_generic_interruption():
    # Both the sentinel-default path (unmapped reason) and None resolve to the
    # generic "a gateway interruption" phrasing.
    assert "a gateway interruption" in _build_interruption_system_note("weird", True)
    assert "a gateway interruption" in _build_interruption_system_note(None, True)


def test_tool_tail_note_is_byte_identical():
    assert _build_tool_tail_system_note() == _GOLDEN_TOOL_TAIL_NOTE


def test_note_builders_return_only_the_bracketed_note():
    # The builder returns just the note; the caller appends the real message.
    note = _build_interruption_system_note("restart_timeout", True)
    assert note.startswith("[System note:") and note.endswith("]")
    assert "\n" not in note


# --------------------------------------------------------------------------
# 2. Dispatch predicate truth table.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_resume_pending", "has_fresh_tool_tail", "message_is_empty", "expected"),
    [
        # Fresh resume mark always wins, regardless of the other signals.
        (True, True, True, "resume"),
        (True, True, False, "resume"),
        (True, False, True, "resume"),
        (True, False, False, "resume"),
        # No resume mark, fresh tool tail → tool_tail.
        (False, True, True, "tool_tail"),
        (False, True, False, "tool_tail"),
        # Neither fresh signal, but an empty turn → safety net (caller also
        # guards on the raw resume_pending marker). This IS the "two freshness
        # signals disagree" case documented inline at the call site.
        (False, False, True, "safety_net"),
        # Ordinary populated turn → no note.
        (False, False, False, "none"),
    ],
)
def test_resolve_resume_note_kind_truth_table(
    is_resume_pending, has_fresh_tool_tail, message_is_empty, expected
):
    assert (
        _resolve_resume_note_kind(
            is_resume_pending, has_fresh_tool_tail, message_is_empty
        )
        == expected
    )


# --------------------------------------------------------------------------
# 3. Silence classifier vs. resume/recovery notes — both directions.
# --------------------------------------------------------------------------


def test_recovery_report_is_not_classified_as_silence():
    # A genuine post-restart recovery report must be delivered, never swallowed
    # as an intentional-silence marker.
    report = (
        "The gateway restarted and is back online now, nothing pending on my "
        "side, what would you like to do next?"
    )
    assert is_intentional_silence_response(report) is False


def test_resume_notes_are_not_silence_markers():
    # The injected system notes themselves are substantive text, not silence.
    assert (
        is_intentional_silence_response(
            _build_interruption_system_note("restart_timeout", False)
        )
        is False
    )
    assert is_intentional_silence_response(_build_tool_tail_system_note()) is False


@pytest.mark.parametrize("marker", sorted(LIVE_GATEWAY_SILENT_MARKERS))
def test_literal_silence_token_on_resumed_turn_is_still_silence(marker):
    # The reverse direction: even after a resume, a model turn whose entire
    # output is the literal silence token still classifies as silence, so the
    # two mechanisms compose without corrupting each other.
    assert is_intentional_silence_response(marker) is True


def test_report_that_merely_mentions_a_marker_still_delivers():
    # Prose that mentions a token mid-sentence is real content, not silence.
    assert (
        is_intentional_silence_response(
            "I nearly stayed NO_REPLY but here's the recovery summary instead."
        )
        is False
    )
