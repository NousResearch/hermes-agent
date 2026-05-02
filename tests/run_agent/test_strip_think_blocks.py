import re
import pytest

def _make_strip_fn():
    import ast as _ast
    import pathlib
    src = (pathlib.Path(__file__).parents[2] / "run_agent.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)
    exec_ns: dict = {"re": re}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == "AIAgent":
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    if item.name in ("_strip_tag_with_nesting", "_strip_think_blocks"):
                        lines = src.split("\n")[item.lineno - 1 : item.end_lineno]
                        dedented = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
                        exec(dedented, exec_ns)
    class _FakeAgent:
        _strip_tag_with_nesting = staticmethod(exec_ns["_strip_tag_with_nesting"])
    exec_ns["AIAgent"] = _FakeAgent
    raw_fn = exec_ns["_strip_think_blocks"]
    def strip(content: str) -> str:
        return raw_fn(None, content).strip()
    return strip

_strip = _make_strip_fn()

class TestStripThinkBlocksRegression:
    def test_simple_closed_pair(self):
        assert _strip("<think>reasoning</think>Here is your answer.") == "Here is your answer."
    def test_closed_pair_with_newline_boundary(self):
        assert _strip("I will help you.\n<think>Let me reason...</think>\nHere is the result.") == "I will help you.\n\nHere is the result."
    def test_only_think_block_returns_empty(self):
        assert _strip("<think>Only reasoning</think>") == ""
    def test_think_with_attributes(self):
        assert _strip('<think type="deep">reasoning here</think>answer') == "answer"
    def test_multiline_think_block(self):
        assert _strip("<think>\nLine 1\nLine 2\n</think>Answer") == "Answer"
    def test_orphan_close_tag(self):
        assert _strip("Result: 42</think>") == "Result: 42"
    def test_adjacent_closed_pairs(self):
        assert _strip("<think>a</think><think>b</think>visible") == "visible"
    def test_adjacent_blocks_preserve_content_between(self):
        assert _strip("<think>r1</think>text<think>r2</think>more") == "textmore"
    def test_boundary_unterminated_stripped(self):
        assert _strip("\n<think>leaked boundary reasoning") == ""
    def test_empty_string(self):
        assert _strip("") == ""
    def test_plain_text_unchanged(self):
        assert _strip("just some text") == "just some text"
    def test_thinking_variant(self):
        assert _strip("<thinking>inner</thinking>answer") == "answer"
    def test_reasoning_variant(self):
        assert _strip("<reasoning>inner</reasoning>answer") == "answer"
    def test_thought_variant(self):
        assert _strip("<thought>inner</thought>answer") == "answer"
    def test_tool_call_block_stripped(self):
        assert _strip('answer<tool_call>{"name": "x"}</tool_call>') == "answer"
    def test_stray_tool_call_closer(self):
        assert _strip("visible</tool_calls>") == "visible"

class TestStripThinkBlocksBugFixes:
    def test_nested_blocks_outer_body_not_leaked(self):
        inp = "<think>outer <think>inner</think> still outer</think>visible"
        result = _strip(inp)
        assert result == "visible"
        assert "still outer" not in result
        assert "inner" not in result
    def test_inline_unterminated_body_not_leaked(self):
        inp = "Before<think>leaked reasoning"
        result = _strip(inp)
        assert result == "Before"
        assert "leaked" not in result
    def test_deeply_nested_blocks(self):
        inp = "<think>L1 <think>L2 <think>L3</think> back-L2</think> back-L1</think>answer"
        result = _strip(inp)
        assert result == "answer"
        assert "L1" not in result
    def test_inline_unterminated_multiple_tags(self):
        assert _strip("prefix<think>body") == "prefix"
    def test_nested_then_more_visible(self):
        inp = "<think>nested <think>inner</think> outer</think>first\nsecond"
        result = _strip(inp)
        assert "first" in result
        assert "nested" not in result
