"""Tests for cli.py::_strip_reasoning_tags — specifically the tool-call
XML stripping added in openclaw/openclaw#67318 port.

The CLI has its own copy of the stripper because it needs to run on the
final displayed assistant text (after streaming) without depending on the
AIAgent instance. It must stay in sync with run_agent.py::_strip_think_blocks
for tool-call tag coverage."""


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


class TestToolCallStripping:
    def test_tool_call_block_stripped(self):
        text = '<tool_call>{"name": "x"}</tool_call>result'
        result = _strip_reasoning_tags(text)
        assert "<tool_call>" not in result
        assert "result" in result







    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""

    def test_cut_tool_call_stripped_to_visible_prefix(self):
        """Both strippers drop the unrecoverable tail; only prose survives."""
        assert _strip_reasoning_tags(_CUT_FRAGMENT) == "Both gates started."
        assert strip_think_blocks(None, _CUT_FRAGMENT).strip() == "Both gates started."
        assert strip_think_blocks(None, "Waiting.\n<tool_call>process_manage").strip() == "Waiting."

    def test_complete_block_and_inline_prose_mentions_untouched(self):
        for out in (_strip_reasoning_tags(_COMPLETE_WITH_PROSE),
                    strip_think_blocks(None, _COMPLETE_WITH_PROSE)):
            assert "Use <function> in JS. The arg_key field maps to arg_value." in out
            assert out.rstrip().endswith("Done.")
            assert "<tool_call>" not in out

