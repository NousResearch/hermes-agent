"""Tests for background-review session-store isolation (hermes_state).

The background skill/memory review fork shares the parent's ``session_id`` for
prompt-cache warmth. Without the ``_persist_disabled`` isolation it wrote its
harness turn ("Review the conversation above and update the skill library…")
plus its curator-mode reply into the user's REAL session, and the next live
turn re-read that injected user message as a standing instruction — the agent
"became" the curator and refused the actual task.

``_strip_background_review_harness`` is the load-on-read defense-in-depth that
removes any such stray harness message (and the assistant reply that followed
it) so a polluted session resumes clean.
"""

from hermes_state import (
    _is_background_review_harness_message,
    _strip_background_review_harness,
)


class TestIsBackgroundReviewHarnessMessage:
    def test_matches_skill_review_prompt(self):
        msg = {"role": "user", "content": "Review the conversation above and update the skill library now."}
        assert _is_background_review_harness_message(msg) is True

    def test_matches_memory_review_prompt(self):
        msg = {"role": "system", "content": "Review the conversation above and consider saving to memory."}
        assert _is_background_review_harness_message(msg) is True

    def test_matches_after_leading_whitespace(self):
        msg = {"role": "user", "content": "\n\n   Review the conversation above and update the skill library."}
        assert _is_background_review_harness_message(msg) is True

    def test_ignores_normal_user_message(self):
        msg = {"role": "user", "content": "Please review my PR and update the changelog."}
        assert _is_background_review_harness_message(msg) is False

    def test_ignores_assistant_role(self):
        # An assistant message that quotes the harness text is not itself a harness prompt.
        msg = {"role": "assistant", "content": "Review the conversation above and update the skill library"}
        assert _is_background_review_harness_message(msg) is False

    def test_ignores_non_string_content(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "Review the conversation above and update the skill library"}]}
        assert _is_background_review_harness_message(msg) is False

    def test_ignores_non_dict(self):
        assert _is_background_review_harness_message("not a dict") is False  # type: ignore[arg-type]


class TestStripBackgroundReviewHarness:
    def test_strips_harness_and_following_assistant_reply(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."},
            {"role": "user", "content": "Review the conversation above and update the skill library."},
            {"role": "assistant", "content": "Nothing to save."},
            {"role": "user", "content": "Thanks, now book a flight."},
        ]
        out = _strip_background_review_harness(messages)
        contents = [m["content"] for m in out]
        assert contents == ["What's the weather?", "It's sunny.", "Thanks, now book a flight."]

    def test_strips_harness_without_following_assistant(self):
        # Harness message is the last turn — nothing to skip after it.
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Review the conversation above and consider saving to memory."},
        ]
        out = _strip_background_review_harness(messages)
        assert out == [{"role": "user", "content": "Hi"}]

    def test_does_not_skip_user_turn_after_harness(self):
        # If the message after the harness is a USER turn (not the curator reply),
        # it must be preserved — only the immediately-following ASSISTANT reply is dropped.
        messages = [
            {"role": "user", "content": "Review the conversation above and update the skill library."},
            {"role": "user", "content": "Actually, ignore that and help me debug."},
        ]
        out = _strip_background_review_harness(messages)
        assert out == [{"role": "user", "content": "Actually, ignore that and help me debug."}]

    def test_clean_history_passes_through_unchanged(self):
        messages = [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Question two"},
        ]
        assert _strip_background_review_harness(messages) == messages

    def test_empty_list(self):
        assert _strip_background_review_harness([]) == []

    def test_multiple_harness_pairs(self):
        messages = [
            {"role": "user", "content": "Review the conversation above and update the skill library."},
            {"role": "assistant", "content": "Nothing to save."},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "real answer"},
            {"role": "user", "content": "Review the conversation above and consider saving to memory."},
            {"role": "assistant", "content": "Saved one entry."},
        ]
        out = _strip_background_review_harness(messages)
        assert [m["content"] for m in out] == ["real question", "real answer"]
