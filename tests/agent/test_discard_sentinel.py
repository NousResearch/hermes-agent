"""Tests for the shared [DISCARD] sentinel and its two consumers.

The sentinel lets background-produced output (the self-improvement review's
notification, or a redundant parallel subagent's result) be silently dropped
while side effects still persist -- Hermes' analogue of Poke's "wait" tool,
modeled on the cron [SILENT] save-but-don't-deliver contract.
"""

import json

from agent.discard_sentinel import (
    DISCARD_MARKER,
    is_discard_marked,
    strip_discard_marker,
)


class TestSentinelHelpers:
    def test_plain_marker_detected(self):
        assert is_discard_marked("[DISCARD]")

    def test_marker_in_surrounding_text_detected(self):
        assert is_discard_marked("nothing useful here, dropping. [DISCARD]")

    def test_case_insensitive(self):
        assert is_discard_marked("[discard]")
        assert is_discard_marked("[Discard]")

    def test_absent_marker_not_detected(self):
        assert not is_discard_marked("a normal helpful summary")

    def test_non_string_is_not_marked(self):
        assert not is_discard_marked(None)
        assert not is_discard_marked(123)
        assert not is_discard_marked({"x": 1})

    def test_strip_removes_marker(self):
        assert strip_discard_marker("done. [DISCARD]") == "done."
        assert strip_discard_marker("[discard] leftover") == "leftover"

    def test_strip_non_string_returns_empty(self):
        assert strip_discard_marker(None) == ""

    def test_marker_constant_shape(self):
        assert DISCARD_MARKER == "[DISCARD]"


class TestBackgroundReviewDiscard:
    def _msgs(self, final_assistant_text):
        return [
            {"role": "user", "content": "review prompt"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": json.dumps({"success": True, "message": "Memory updated"})},
            {"role": "assistant", "content": final_assistant_text},
        ]

    def test_discard_when_final_message_marked(self):
        from agent.background_review import _review_requested_discard

        assert _review_requested_discard(self._msgs("trivial dedupe [DISCARD]"))

    def test_no_discard_when_final_message_clean(self):
        from agent.background_review import _review_requested_discard

        assert not _review_requested_discard(self._msgs("Saved a durable preference."))

    def test_uses_last_assistant_message(self):
        from agent.background_review import _review_requested_discard

        msgs = [
            {"role": "assistant", "content": "[DISCARD] earlier"},
            {"role": "tool", "tool_call_id": "t1", "content": "{}"},
            {"role": "assistant", "content": "final clean reply"},
        ]
        # The LAST assistant message wins -> not discarded.
        assert not _review_requested_discard(msgs)

    def test_empty_messages_not_discarded(self):
        from agent.background_review import _review_requested_discard

        assert not _review_requested_discard([])
        assert not _review_requested_discard(None)


class TestDelegateBatchHint:
    def test_hint_appended_to_goal(self):
        from tools.delegate_tool import _append_batch_discard_hint

        out = _append_batch_discard_hint("Research topic A")
        assert out.startswith("Research topic A")
        assert "[DISCARD]" in out
        assert "Parallel-task note" in out
