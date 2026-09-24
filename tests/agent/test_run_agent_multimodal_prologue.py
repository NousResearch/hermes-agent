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

from run_agent import _summarize_user_message_for_log
from agent.codex_responses_adapter import _chat_content_to_responses_parts


class TestSummarizeUserMessageForLog:
    def test_plain_string_passthrough(self):
        assert _summarize_user_message_for_log("hello world") == "hello world"


    def test_text_only_list(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _summarize_user_message_for_log(content) == "hi there"







class TestChatContentToResponsesParts:
    def test_non_list_returns_empty(self):
        assert _chat_content_to_responses_parts("hi") == []
        assert _chat_content_to_responses_parts(None) == []

    def test_text_parts_become_input_text(self):
        content = [{"type": "text", "text": "hello"}]
        assert _chat_content_to_responses_parts(content) == [{"type": "input_text", "text": "hello"}]





