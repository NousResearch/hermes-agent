"""Tests for cli.py::_strip_reasoning_tags — specifically the tool-call
XML stripping added in openclaw/openclaw#67318 port.

The CLI has its own copy of the stripper because it needs to run on the
final displayed assistant text (after streaming) without depending on the
AIAgent instance. It must stay in sync with run_agent.py::_strip_think_blocks
for tool-call tag coverage."""

import pytest

from cli import _strip_reasoning_tags


class TestToolCallStripping:
    def test_tool_call_block_stripped(self):
        text = '<tool_call>{"name": "x"}</tool_call>result'
        result = _strip_reasoning_tags(text)
        assert "<tool_call>" not in result
        assert "result" in result

    def test_function_calls_block_stripped(self):
        text = '<function_calls>[{}]</function_calls>\nanswer'
        result = _strip_reasoning_tags(text)
        assert "<function_calls>" not in result
        assert "answer" in result

    def test_gemma_function_name_block_stripped(self):
        text = (
            'Reading.\n'
            '<function name="r"><parameter name="p">/tmp/x</parameter></function>\n'
            'Done.'
        )
        result = _strip_reasoning_tags(text)
        assert '<function name="r">' not in result
        assert "/tmp/x" not in result
        assert "Reading." in result
        assert "Done." in result

    def test_prose_mention_of_function_preserved(self):
        text = "Use <function> declarations in JavaScript."
        result = _strip_reasoning_tags(text)
        assert "JavaScript" in result

    def test_reasoning_still_stripped(self):
        """Regression: make sure existing think-tag stripping still works."""
        text = "<think>reasoning</think> answer"
        result = _strip_reasoning_tags(text)
        assert "reasoning" not in result
        assert "answer" in result

    def test_mixed_reasoning_and_tool_call(self):
        text = '<think>plan</think><tool_call>{"x":1}</tool_call>final'
        result = _strip_reasoning_tags(text)
        assert "plan" not in result
        assert "<tool_call>" not in result
        assert "final" in result

    def test_stray_function_close(self):
        text = "visible</function> tail"
        result = _strip_reasoning_tags(text)
        assert "</function>" not in result
        assert "visible" in result
        assert "tail" in result

    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""

    def test_plain_text_unchanged(self):
        assert _strip_reasoning_tags("just text") == "just text"
