"""Regression tests for hermes_cli/cli_content_scrub.py (c3 cluster).

Wave 1 godfile extraction, shard s1 cluster c3: ``_strip_reasoning_tags``,
``_assistant_content_as_text``, ``_assistant_copy_text`` and the
``_REASONING_TAGS`` constant, moved verbatim from cli.py. Expected values in
this file are pinned to the live cli.py behavior (verified against the live
functions on 2026-08-05 before extraction). The existing
``tests/run_agent/test_strip_reasoning_tags_cli.py`` covers the same
functions through the ``from cli import _strip_reasoning_tags`` re-export;
these tests exercise the moved module directly.
"""

from hermes_cli.cli_content_scrub import (
    _REASONING_TAGS,
    _assistant_content_as_text,
    _assistant_copy_text,
    _strip_reasoning_tags,
)


class TestStripReasoningTags:
    def test_closed_pair(self):
        # Trailing ``\s*`` after the close tag eats the space after it.
        assert _strip_reasoning_tags("a <think>hidden</think> b") == "a b"

    def test_closed_pair_multiline(self):
        text = "a <think>\nline1\nline2\n</think> b"
        assert _strip_reasoning_tags(text) == "a b"

    def test_unterminated_open_tag_runs_to_eof(self):
        assert _strip_reasoning_tags("a <thinking>never closed") == "a"

    def test_stray_orphan_close_tag(self):
        assert _strip_reasoning_tags("stuff</think>answer") == "stuffanswer"

    def test_case_insensitive(self):
        assert _strip_reasoning_tags("a <THINK>x</THINK> b") == "a b"

    def test_all_known_tags(self):
        for tag in _REASONING_TAGS:
            assert _strip_reasoning_tags(f"<{tag}>x</{tag}>") == ""

    def test_tool_call_block_stripped(self):
        text = '<tool_call>{"name": "x"}</tool_call>result'
        result = _strip_reasoning_tags(text)
        assert "<tool_call>" not in result
        assert "result" in result

    def test_tool_calls_plural(self):
        assert _strip_reasoning_tags("a <tool_calls>payload</tool_calls> b") == "a b"

    def test_function_named_block_needs_line_boundary(self):
        # The named-function regex is boundary-gated (start / after
        # newline/.!?:); mid-line only the stray close tag is stripped.
        assert _strip_reasoning_tags('a <function name="f">body</function> b') == (
            'a <function name="f">bodyb'
        )

    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""


class TestAssistantContentAsText:
    def test_none(self):
        assert _assistant_content_as_text(None) == ""

    def test_str(self):
        assert _assistant_content_as_text("hello") == "hello"

    def test_list_of_text_dicts(self):
        content = [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]
        assert _assistant_content_as_text(content) == "one\ntwo"

    def test_list_skips_non_text_and_empty(self):
        content = [
            {"type": "image", "text": "ignored"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "keep"},
        ]
        assert _assistant_content_as_text(content) == "keep"

    def test_other_types_str_fallback(self):
        assert _assistant_content_as_text(42) == "42"


class TestAssistantCopyText:
    def test_strips_reasoning_from_text_content(self):
        content = [{"type": "text", "text": "a <think>x</think> b"}]
        assert _assistant_copy_text(content) == "a b"

    def test_passthrough_plain(self):
        assert _assistant_copy_text("plain") == "plain"
