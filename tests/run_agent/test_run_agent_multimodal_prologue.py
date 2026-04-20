"""Regression tests for run_conversation's prologue handling of multimodal content.

PR #5621 and earlier multimodal PRs hit an ``AttributeError`` in
``run_agent.run_conversation`` because the prologue unconditionally called
``user_message[:80] + "..."`` / ``.replace()`` / ``_safe_print(f"...{user_message[:60]}")``
on what was now a list.  These tests cover the two fixes:

  1. ``_summarize_user_message_for_log`` accepts strings, lists, and ``None``.
  2. ``_chat_content_to_responses_parts`` converts chat-style content to the
     Responses API ``input_text`` / ``input_image`` shape.

They do NOT boot the full AIAgent — the prologue-fix guarantees are pure
function contracts at module scope.
"""

from run_agent import _chat_content_to_responses_parts, _summarize_user_message_for_log


class TestSummarizeUserMessageForLog:
    def test_plain_string_passthrough(self):
        assert _summarize_user_message_for_log("hello world") == "hello world"

    def test_none_returns_empty_string(self):
        assert _summarize_user_message_for_log(None) == ""

    def test_text_only_list(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _summarize_user_message_for_log(content) == "hi there"

    def test_list_with_image_only(self):
        content = [{"type": "image_url", "image_url": {"url": "https://x"}}]
        # Image-only: "[1 image]" marker, no trailing space.
        assert _summarize_user_message_for_log(content) == "[1 image]"

    def test_list_with_text_and_image(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://x"}},
        ]
        summary = _summarize_user_message_for_log(content)
        assert "[1 image]" in summary
        assert "describe this" in summary

    def test_list_with_multiple_images(self):
        content = [
            {"type": "text", "text": "compare these"},
            {"type": "image_url", "image_url": {"url": "a"}},
            {"type": "image_url", "image_url": {"url": "b"}},
        ]
        summary = _summarize_user_message_for_log(content)
        assert "[2 images]" in summary

    def test_scalar_fallback(self):
        assert _summarize_user_message_for_log(42) == "42"

    def test_list_supports_slice_and_replace(self):
        """The whole point of this helper: its output must be a plain str."""
        content = [{"type": "text", "text": "x" * 200}, {"type": "image_url", "image_url": {"url": "y"}}]
        summary = _summarize_user_message_for_log(content)
        # These are the operations the run_conversation prologue performs.
        _ = summary[:80] + "..."
        _ = summary.replace("\n", " ")


class TestChatContentToResponsesParts:
    def test_non_list_returns_empty(self):
        assert _chat_content_to_responses_parts("hi") == []
        assert _chat_content_to_responses_parts(None) == []

    def test_text_parts_become_input_text(self):
        content = [{"type": "text", "text": "hello"}]
        assert _chat_content_to_responses_parts(content) == [{"type": "input_text", "text": "hello"}]

    def test_image_url_object_becomes_input_image(self):
        content = [{"type": "image_url", "image_url": {"url": "https://x", "detail": "high"}}]
        assert _chat_content_to_responses_parts(content) == [
            {"type": "input_image", "image_url": "https://x", "detail": "high"},
        ]

    def test_bare_string_image_url(self):
        content = [{"type": "image_url", "image_url": "https://x"}]
        assert _chat_content_to_responses_parts(content) == [{"type": "input_image", "image_url": "https://x"}]

    def test_responses_format_passthrough(self):
        """Input already in Responses format should round-trip cleanly."""
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://x"},
        ]
        assert _chat_content_to_responses_parts(content) == [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://x"},
        ]

    def test_unknown_parts_skipped(self):
        """Unknown types shouldn't crash — filtered silently at this level
        (the API server's normalizer rejects them earlier)."""
        content = [{"type": "text", "text": "ok"}, {"type": "audio", "x": "y"}]
        assert _chat_content_to_responses_parts(content) == [{"type": "input_text", "text": "ok"}]

    def test_empty_url_image_skipped(self):
        content = [{"type": "image_url", "image_url": {"url": ""}}]
        assert _chat_content_to_responses_parts(content) == []
