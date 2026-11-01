"""Markdown code regions survive ``strip_think_blocks`` (#133845).

A literal ``<tool_call>`` quoted inside a fenced block or inline code is user-visible
text (documentation, examples, logs), not model output. The sanitizer used to apply
``_UNTERMINATED_TOOL_CALL_PATTERN`` across the whole message, so a quoted tag ate
everything from its line to the end of the response — the "truncated at backticks"
report. These tests lock the split: code regions pass through verbatim, everything
outside them is sanitized exactly as before.
"""

from __future__ import annotations

from agent.agent_runtime_helpers import strip_think_blocks


class TestFencedCodeBlocks:
    def test_backtick_fence_tool_call_literal_preserved(self) -> None:
        text = (
            "Here is how the tool call looks:\n"
            "```text\n"
            "<tool_call>\n"
            "{'name': 'web_search', 'arguments': {'q': 'hermes'}}\n"
            "</tool_call>\n"
            "```\n"
            "That snippet is an example, not a real invocation."
        )
        out = strip_think_blocks(None, text)
        assert out == text

    def test_tilde_fence_tool_call_literal_preserved(self) -> None:
        text = "~~~\n<function_call>{'name': 'f'}</function_call>\n~~~\nAfter."
        assert strip_think_blocks(None, text) == text

    def test_variable_length_fence_with_shorter_run_inside(self) -> None:
        # A 4-backtick fence stays open across a bare 3-backtick line; the inner run is
        # content, not the closing fence.
        text = "````\n```\n<tool_call>x</tool_call>\n```\n````\ntail"
        assert strip_think_blocks(None, text) == text

    def test_reasoning_tag_literal_inside_fence_preserved(self) -> None:
        text = "```xml\n<think>scratch</think>\n```"
        assert strip_think_blocks(None, text) == text

    def test_unclosed_fence_runs_to_end(self) -> None:
        text = "intro\n```python\n<tool_call>\nstill literal"
        assert strip_think_blocks(None, text) == text

    def test_fence_info_string_ignored(self) -> None:
        text = "```python\n<tool_call>y</tool_call>\n```"
        assert strip_think_blocks(None, text) == text


class TestInlineCode:
    def test_inline_code_tool_call_preserved(self) -> None:
        text = "Write `<tool_call>{'name': 'f'}</tool_call>` to call the tool."
        assert strip_think_blocks(None, text) == text

    def test_inline_code_surrounded_prose_sanitized_independently(self) -> None:
        text = "use `<tool_call>x</tool_call>` then <think>secret</think> done"
        out = strip_think_blocks(None, text)
        assert "`<tool_call>x</tool_call>`" in out
        assert "secret" not in out
        assert "done" in out

    def test_unpaired_backticks_leave_prose_untouched(self) -> None:
        text = "a ` b <think>hidden</think> c"
        out = strip_think_blocks(None, text)
        assert "a ` b" in out
        assert "hidden" not in out


class TestSanitizationOutsideCodeStillApplies:
    def test_real_tool_call_block_before_fence_removed(self) -> None:
        text = (
            "<tool_call>\n"
            "{'name': 'f', 'arguments': {}}\n"
            "</tool_call>\n"
            "```text\n"
            "<tool_call>quoted</tool_call>\n"
            "```"
        )
        out = strip_think_blocks(None, text)
        assert "{'name': 'f', 'arguments': {}}" not in out
        assert "<tool_call>quoted</tool_call>" in out

    def test_unterminated_real_tool_call_still_dropped_to_segment_end(self) -> None:
        # A line-start unclosed <tool_call> outside any code region is a real truncated
        # tool call (#101899): the prose segment from it on is dropped. The fence after
        # it is a separate segment and survives.
        text = "before\n<tool_call>\n{'name': 'f'}\n```text\nliteral\n```"
        out = strip_think_blocks(None, text)
        assert "before" in out
        assert "{'name': 'f'}" not in out
        assert "literal" in out

    def test_think_block_between_fences_removed(self) -> None:
        text = "```text\na\n```\n<think>reasoning</think>\nvisible\n```text\nb\n```"
        out = strip_think_blocks(None, text)
        assert "reasoning" not in out
        assert "visible" in out
        assert "a" in out
        assert "b" in out

    def test_stray_closer_after_inline_code_still_removed(self) -> None:
        text = "see `docs` then\n</tool_call>\nend"
        out = strip_think_blocks(None, text)
        assert "</tool_call>" not in out
        assert "see `docs` then" in out
        assert "end" in out

    def test_no_code_text_behaviour_unchanged(self) -> None:
        text = "<think>hidden reasoning</think>answer\n<tool_call>\n{'name': 'f'}\n"
        out = strip_think_blocks(None, text)
        assert "hidden" not in out
        assert "answer" in out
        assert "{'name': 'f'}" not in out

    def test_none_and_empty(self) -> None:
        assert strip_think_blocks(None, None) == ""
        assert strip_think_blocks(None, "") == ""
