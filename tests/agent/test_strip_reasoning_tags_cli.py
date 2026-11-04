"""Tests for cli.py::_strip_reasoning_tags — specifically the tool-call
XML stripping added in openclaw/openclaw#67318 port.

The CLI has its own copy of the stripper because it needs to run on the
final displayed assistant text (after streaming) without depending on the
AIAgent instance. It must stay in sync with run_agent.py::_strip_think_blocks
for tool-call tag coverage."""

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

    def test_namespace_prefixed_tool_call_block_stripped(self):
        # muse-spark (opencode-go, Responses wire) serializes a native call onto the text
        # channel as <atem:function_calls>…</atem:function_calls>; without a namespace-aware
        # pattern the literal XML leaks into the delivered final (the "not covered yet"
        # shape tracked in #103483).
        text = (
            "Checking the queue.\n"
            "<atem:function_calls>\n"
            '<atem:invoke name="default.terminal">\n'
            '<atem:parameter name="command">echo hi</atem:parameter>\n'
            "</atem:invoke>\n"
            "</atem:function_calls>"
        )
        for out in (_strip_reasoning_tags(text), strip_think_blocks(None, text)):
            assert "atem:" not in out
            assert "function_calls" not in out
            assert "Checking the queue." in out.strip()

    def test_cut_namespace_prefixed_tool_call_stripped_to_visible_prefix(self):
        text = 'Waiting.\n<atem:function_calls>\n<atem:invoke name="default.terminal">'
        for out in (_strip_reasoning_tags(text), strip_think_blocks(None, text)):
            assert out.strip() == "Waiting."


    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""

    @pytest.mark.parametrize("stripper", _STRIPPERS)
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (_CUT_FRAGMENT, "Both gates started."),
            (_CUT_FRAGMENT.split("\n", 1)[1], ""),
            (_CUT_FRAGMENT + "\nThird line must survive.",
             "Both gates started.\nThird line must survive."),
            ("Waiting.\n<tool_call>process_manage", "Waiting."),
            ("Done.\nprocess_manage<arg_key>action</arg_key><arg_value>wait</arg_value>", "Done."),
            ("Hi.\nterminal<arg_key>command</arg_key><arg_value>ls -la", "Hi."),
        ],
        ids=["original-fragment", "fragment-only", "prose-after-fragment", "unclosed-call",
             "name-prefixed-fragment", "name-prefixed-fragment-cut-value"],
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
