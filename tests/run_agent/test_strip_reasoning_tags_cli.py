"""Tests for cli.py::_strip_reasoning_tags — specifically the tool-call
XML stripping added in openclaw/openclaw#67318 port.

The CLI has its own copy of the stripper because it needs to run on the
final displayed assistant text (after streaming) without depending on the
AIAgent instance. It must stay in sync with run_agent.py::_strip_think_blocks
for tool-call tag coverage.
"""

from functools import partial

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from cli import _strip_reasoning_tags

# GLM text-channel tool call cut mid-serialization by a stream drop (#101899):
# the first key and call name never arrived, only orphan argument markup.
_CUT_FRAGMENT = (
    "Both gates started.\n"
    "wait</arg_value>\n<arg_key>session_id</arg_key>\n<arg_value>abc</arg_value>\n"
    "<arg_key>timeout</arg_key>\n<arg_value>59"
)
_COMPLETE_WITH_PROSE = (
    "Use <function> in JS. The arg_key field maps to arg_value.\n"
    "<tool_call>x<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>\nDone."
)

_STRIPPERS = [
    pytest.param(_strip_reasoning_tags, id="display"),
    pytest.param(partial(strip_think_blocks, None), id="storage"),
]


class TestToolCallStripping:
    def test_tool_call_block_stripped(self):
        text = '<tool_call>{"name": "x"}</tool_call>result'
        result = _strip_reasoning_tags(text)
        assert "<tool_call>" not in result
        assert "result" in result

    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""

    @pytest.mark.parametrize(
        "stripper", [_strip_reasoning_tags, partial(strip_think_blocks, None)],
        ids=["display", "storage"],
    )
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (_CUT_FRAGMENT, "Both gates started."),
            (_CUT_FRAGMENT.split("\n", 1)[1], ""),
            (_CUT_FRAGMENT + "\nThird line must survive.",
             "Both gates started.\nThird line must survive."),
            ("Waiting.\n<tool_call>process_manage", "Waiting."),
        ],
        ids=["original-fragment", "fragment-only", "prose-after-fragment", "unclosed-call"],
    )
    def test_cut_tool_call_stripped_to_visible_prefix(self, stripper, text, expected):
        """A cut call is unrecoverable, but unrelated prose must not be lost."""
        assert stripper(text).strip() == expected

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    @pytest.mark.parametrize("tag", ["arg_key", "arg_value", "/arg_key", "/arg_value"])
    def test_bracketed_arg_tag_prose_and_tail_survive(self, stripper, tag):
        text = (
            "First line stays.\n"
            f"The <{tag}> holds the parameter name.\n"
            "Third line must survive."
        )
        assert stripper(text) == text

    def test_complete_block_and_inline_prose_mentions_untouched(self):
        for out in (_strip_reasoning_tags(_COMPLETE_WITH_PROSE),
                    strip_think_blocks(None, _COMPLETE_WITH_PROSE)):
            assert "Use <function> in JS. The arg_key field maps to arg_value." in out
            assert out.rstrip().endswith("Done.")
            assert "<tool_call>" not in out


class TestEndOfLineArgTagStripping:
    """The new third alternative in the regex matches </arg_value> only at
    end of line (followed by whitespace + newline or EOF)."""

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_at_eol_with_trailing_newline(self, stripper):
        """</arg_value> at end of line followed by newline → stripped."""
        text = "Both gates started.\nwait</arg_value>\nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_at_eof_no_newline(self, stripper):
        """</arg_value> at very end of text (no trailing newline) → stripped."""
        text = "Both gates started.\nwait</arg_value>"
        result = stripper(text).strip()
        assert result == "Both gates started."

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_with_trailing_spaces_then_newline(self, stripper):
        """</arg_value> followed by spaces then newline → stripped."""
        text = "Both gates started.\nwait</arg_value>   \nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_with_crlf(self, stripper):
        """Windows \r\n line endings: </arg_value>\r\n → stripped."""
        text = "Both gates started.\r\nwait</arg_value>\r\nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_mid_line_with_more_text(self, stripper):
        """</arg_value> in the middle of a line (more text after) → NOT stripped."""
        text = "The result was wait</arg_value> and then more."
        result = stripper(text).strip()
        assert result == text

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_mid_line_then_newline(self, stripper):
        """</arg_value> followed by more text on same line → NOT stripped."""
        text = "First line.\nThe result was wait</arg_value> and then more.\nThird line."
        result = stripper(text).strip()
        assert "First line." in result
        assert "The result was wait</arg_value> and then more." in result
        assert "Third line." in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_key_at_eol(self, stripper):
        """</arg_key> at end of line → stripped (same as </arg_value>)."""
        text = "Both gates started.\nwait</arg_key>\nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_key>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_multiple_fragment_lines(self, stripper):
        """Multiple consecutive fragment lines → all stripped, prose survives."""
        text = (
            "Both gates started.\n"
            "wait</arg_value>\n"
            "<arg_key>session_id</arg_key>\n"
            "<arg_value>abc</arg_value>\n"
            "Third line."
        )
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result
        assert "<arg_key>" not in result
        assert "<arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_empty_line_between_fragments(self, stripper):
        """Empty lines between fragments should not affect stripping."""
        text = "Both gates started.\n\nwait</arg_value>\n\nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_arg_value_at_eol_with_tabs(self, stripper):
        """</arg_value> with tab indentation at end of line → stripped."""
        text = "Both gates started.\n\twait</arg_value>\nThird line."
        result = stripper(text).strip()
        assert "Both gates started." in result
        assert "Third line." in result
        assert "wait</arg_value>" not in result

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_prose_mention_of_arg_value_not_stripped(self, stripper):
        """Inline prose like 'the arg_value field' must NOT be stripped."""
        text = "The arg_value field contains the parameter value."
        result = stripper(text).strip()
        assert result == text

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    def test_prose_mention_with_tool_call(self, stripper):
        """Prose mentioning arg_value alongside a real tool call."""
        text = (
            "The arg_value field maps to the value.\n"
            "<tool_call>process_data<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
            "Done."
        )
        result = stripper(text).strip()
        assert "The arg_value field maps to the value." in result
        assert "Done." in result
        assert "<tool_call>" not in result
