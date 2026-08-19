"""Regression coverage for #80622: a reference-only compaction handoff must
never become the active user turn after a completed assistant response.

Failure mode (real report): assistant finished with ``finish_reason=stop``,
Hermes inserted a standalone ``role=user`` ``[CONTEXT COMPACTION — REFERENCE
ONLY]`` handoff containing a Historical Task Snapshot, no new human request
followed, and the agent immediately started a tool-calling turn that resumed
the already-completed work.
"""

from __future__ import annotations

from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    COMPRESSION_CONTINUATION_USER_CONTENT,
    ContextCompressor,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
    is_compaction_summary_message,
    is_user_originated_turn,
    reference_handoff_would_drive_next_model_call,
)
from agent.conversation_loop import (
    _should_skip_model_call_for_reference_handoff,
)
from agent.turn_context import reanchor_current_turn_user_idx


def _standalone_handoff(task: str = "finish the already-done refactor") -> dict:
    return {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\n"
            f"User asked: '{task}'\n\n{_SUMMARY_END_MARKER}"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: True,
    }


def _merged_assistant_carrier(
    *, finish_reason: str = "stop", tool_calls: list[dict] | None = None
) -> dict:
    """Production merge-into-tail shape with the carrier's completion state."""
    carrier = {
        "role": "assistant",
        "content": (
            f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
            "Refactor complete.\n\n"
            f"{_MERGED_SUMMARY_DELIMITER}\n"
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\n"
            "User asked: 'finish the already-done refactor'\n\n"
            f"{_SUMMARY_END_MARKER}"
        ),
        "finish_reason": finish_reason,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: True,
    }
    if tool_calls is not None:
        carrier["tool_calls"] = tool_calls
    return carrier


class TestReferenceHandoffWouldDriveNextModelCall:
    def test_standalone_handoff_alone_drives(self):
        messages = [_standalone_handoff()]
        assert reference_handoff_would_drive_next_model_call(messages) is True

    def test_handoff_after_completed_stop_drives(self):
        """The reported sequence: assistant stop, then synthetic handoff."""
        messages = [
            {"role": "user", "content": "please finish the refactor"},
            {
                "role": "assistant",
                "content": "Refactor complete.",
                "finish_reason": "stop",
            },
            _standalone_handoff(),
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is True

    def test_real_user_after_handoff_does_not_drive(self):
        messages = [
            _standalone_handoff(),
            {"role": "user", "content": "what's the capital of France?"},
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_mid_tool_loop_after_handoff_does_not_drive(self):
        messages = [
            _standalone_handoff(),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_completed_merged_assistant_carrier_drives_without_adjacent_stop(self):
        """The carrier's own stop marks preserved assistant prose as completed."""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "please finish the refactor"},
            _merged_assistant_carrier(),
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is True

    def test_live_tool_call_carrier_after_completed_stop_stays_in_flight(self):
        """Pending tool_calls on the carrier are live even after an earlier stop."""
        messages = [
            {
                "role": "assistant",
                "content": "Earlier turn complete.",
                "finish_reason": "stop",
            },
            _merged_assistant_carrier(
                finish_reason="tool_calls",
                tool_calls=[{"id": "live-c1", "function": {"name": "terminal"}}],
            ),
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_standalone_assistant_handoff_after_stop_uses_existing_guard(self):
        handoff = _standalone_handoff()
        handoff["role"] = "assistant"
        messages = [
            {
                "role": "assistant",
                "content": "Refactor complete.",
                "finish_reason": "stop",
            },
            handoff,
        ]
        assert ContextCompressor.classify_summary_content(handoff["content"]) == (
            "standalone"
        )
        assert reference_handoff_would_drive_next_model_call(messages) is True

    def test_real_user_after_merged_assistant_carrier_does_not_drive(self):
        messages = [
            {
                "role": "assistant",
                "content": "Refactor complete.",
                "finish_reason": "stop",
            },
            _merged_assistant_carrier(),
            {"role": "user", "content": "start a different task"},
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_distinct_tool_call_after_merged_carrier_stays_in_flight(self):
        messages = [
            {
                "role": "assistant",
                "content": "Refactor complete.",
                "finish_reason": "stop",
            },
            _merged_assistant_carrier(),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "live-c2", "function": {"name": "terminal"}}],
            },
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_embedded_remainder_after_end_marker_does_not_drive(self):
        messages = [
            {
                "role": "user",
                "content": (
                    f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold\n\n"
                    f"{_SUMMARY_END_MARKER}\n\nwhat's the capital of France?"
                ),
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            }
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is False

    def test_continuation_marker_alone_still_drives(self):
        """Synthetic continuation is not a human ask — sole-handoff +
        continuation must not license a fresh tool loop after stop."""
        messages = [
            _standalone_handoff(),
            {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        ]
        assert reference_handoff_would_drive_next_model_call(messages) is True


class TestSkipGuardRestoresRealUser:
    def test_restores_pending_user_then_allows_continue(self):
        messages = [
            {
                "role": "assistant",
                "content": "done",
                "finish_reason": "stop",
            },
            _standalone_handoff(),
        ]
        assert _should_skip_model_call_for_reference_handoff(
            messages, "new ask after compaction"
        ) is False
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "new ask after compaction"

    def test_skips_when_no_real_user_to_restore(self):
        messages = [
            {
                "role": "assistant",
                "content": "Refactor complete.",
                "finish_reason": "stop",
            },
            _standalone_handoff(),
        ]
        assert _should_skip_model_call_for_reference_handoff(messages, None) is True


class TestUserOriginatedTurnPredicate:
    def test_standalone_handoff_not_user_originated_even_with_has_user_turn(self):
        handoff = _standalone_handoff()
        assert is_compaction_summary_message(handoff) is True
        assert is_user_originated_turn(handoff) is False

    def test_plain_user_is_originated(self):
        assert is_user_originated_turn({"role": "user", "content": "hello"}) is True

    def test_display_kind_hidden_not_originated(self):
        assert (
            is_user_originated_turn(
                {
                    "role": "user",
                    "content": "opaque",
                    "display_kind": "hidden",
                }
            )
            is False
        )


class TestReanchorSkipsHandoffFallback:
    def test_fallback_skips_standalone_handoff(self):
        messages = [
            {"role": "system", "content": "sys"},
            _standalone_handoff(),
        ]
        assert reanchor_current_turn_user_idx(messages, "missing ask") == -1

    def test_exact_match_still_wins(self):
        messages = [
            _standalone_handoff(),
            {"role": "user", "content": "live ask"},
        ]
        assert reanchor_current_turn_user_idx(messages, "live ask") == 1

    def test_fallback_prefers_real_user_over_handoff(self):
        messages = [
            {"role": "user", "content": "original ask"},
            _standalone_handoff(),
        ]
        # Exact content rewritten by merge — fall back must not land on handoff.
        assert reanchor_current_turn_user_idx(messages, "rewritten ask") == 0


class TestNoToolCallsWithoutLaterRealUser:
    def test_historical_snapshot_alone_is_not_actionable(self):
        """Suggested regression #3: executable-looking snapshot must not
        count as a user-originated turn."""
        handoff = _standalone_handoff(
            "run npm test and fix every failure in the suite"
        )
        assert is_user_originated_turn(handoff) is False
        assert reference_handoff_would_drive_next_model_call([handoff]) is True
