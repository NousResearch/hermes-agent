"""Regression coverage for #80622: a reference-only compaction handoff must
never become the active user turn after a completed assistant response.

Failure mode (real report): assistant finished with ``finish_reason=stop``,
Hermes inserted a standalone ``role=user`` ``[CONTEXT COMPACTION — REFERENCE
ONLY]`` handoff containing a Historical Task Snapshot, no new human request
followed, and the agent immediately started a tool-calling turn that resumed
the already-completed work.
"""

from __future__ import annotations

import pytest

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
    history_before_user_originated_turn,
    is_compaction_summary_message,
    is_user_originated_turn,
    reference_handoff_would_drive_next_model_call,
    retryable_user_text,
    split_user_originated_turn,
    user_originated_turn_view,
)
from agent.conversation_loop import (
    _should_skip_model_call_for_reference_handoff,
)
from agent.agent_runtime_helpers import repair_message_sequence
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


def _composite_handoff(ask: str = "REAL ASK") -> dict:
    handoff = _standalone_handoff()
    handoff["content"] += f"\n\n{ask}"
    return handoff


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

    def test_force_user_leading_carrier_is_originated_via_live_view(self):
        carrier = _composite_handoff()

        handoff, live = split_user_originated_turn(carrier)

        assert is_user_originated_turn(carrier) is True
        assert user_originated_turn_view(carrier)["content"] == "REAL ASK"
        assert live["content"] == "REAL ASK"
        assert handoff["display_kind"] == "hidden"
        assert "REAL ASK" not in handoff["content"]

    def test_hidden_legacy_carrier_still_projects_but_typed_carrier_does_not(self):
        hidden = {**_composite_handoff(), "display_kind": "hidden"}
        typed = {**_composite_handoff(), "display_kind": "auto_continue"}

        assert user_originated_turn_view(hidden)["content"] == "REAL ASK"
        assert user_originated_turn_view(typed) is None

    def test_rewind_prefix_preserves_only_the_carrier_scaffold(self):
        messages = [
            {"role": "assistant", "content": "prefix"},
            _composite_handoff(),
            {"role": "assistant", "content": "failed"},
        ]

        prefix, live = history_before_user_originated_turn(messages, 1)

        assert live["content"] == "REAL ASK"
        assert prefix[0] == messages[0]
        assert prefix[1]["display_kind"] == "hidden"
        assert "REAL ASK" not in prefix[1]["content"]
        assert messages[1]["content"].endswith("REAL ASK")

    def test_live_projection_keeps_reactions_but_drops_wrapper_metadata(self):
        carrier = _composite_handoff()
        carrier["display_metadata"] = {
            "reactions": [{"emoji": "👍", "author": "user"}],
            "synthetic_only": "do not project",
        }

        live = user_originated_turn_view(carrier)

        assert live["display_metadata"] == {
            "reactions": [{"emoji": "👍", "author": "user"}]
        }

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

    def test_exact_live_projection_reanchors_to_composite_carrier(self):
        messages = [
            {"role": "user", "content": "older ask"},
            _composite_handoff("rewritten ask"),
        ]

        assert reanchor_current_turn_user_idx(messages, "rewritten ask") == 1


class TestRetryableUserText:
    def test_losslessly_flattens_text_only_parts(self):
        assert retryable_user_text(
            [
                {"type": "text", "text": "hello "},
                {"type": "input_text", "text": "world"},
            ]
        ) == "hello world"

    def test_rejects_structured_media_parts(self):
        with pytest.raises(ValueError, match="media or unknown"):
            retryable_user_text(
                [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]
            )

    @pytest.mark.parametrize(
        "text",
        [
            "Explain why ![alt](https://example.test/a.png) renders",
            "Review @file:README.md and this data:image/png example",
            "Explain the literal [screenshot] marker",
            "Compare [file: README.md] with [image|ybres:RID]",
        ],
    )
    def test_keeps_ordinary_text_that_uses_media_like_syntax(self, text):
        assert retryable_user_text(text) == text


class TestCarrierAlternationRepair:
    def test_fresh_user_does_not_mutate_persisted_carrier_dict(self):
        carrier = _composite_handoff()
        original = carrier.copy()
        fresh = {"role": "user", "content": "NEXT ASK"}
        messages = [carrier, fresh]

        assert repair_message_sequence(None, messages) == 0

        assert messages == [carrier, fresh]
        assert messages[0] is carrier
        assert carrier == original


class TestNoToolCallsWithoutLaterRealUser:
    def test_historical_snapshot_alone_is_not_actionable(self):
        """Suggested regression #3: executable-looking snapshot must not
        count as a user-originated turn."""
        handoff = _standalone_handoff(
            "run npm test and fix every failure in the suite"
        )
        assert is_user_originated_turn(handoff) is False
        assert reference_handoff_would_drive_next_model_call([handoff]) is True
