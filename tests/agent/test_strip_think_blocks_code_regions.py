"""Regression tests for Issue #133845: strip_think_blocks must not strip literal
tool-call/think tags that appear inside Markdown code regions (fenced backtick,
fenced tilde, variable-length fences, inline code). Code regions survive the
sanitize pass byte-identical; actual sanitize targets outside code are unchanged."""

from agent.agent_runtime_helpers import strip_think_blocks


class _NoAgent:
    pass


def _strip(content: str) -> str:
    return strip_think_blocks(_NoAgent(), content)


def test_backtick_fence_preserves_tool_call_tags():
    content = "Example:\n\n```xml\n<tool_call>\n{'name': 't'}\n</tool_call>\n```\n\ndone."
    assert _strip(content) == content


def test_tilde_fence_preserves_tags():
    content = "~~~\n<think>spoiler</think> ~ok\n~~~"
    assert _strip(content) == content


def test_variable_length_fence_preserves_tags():
    content = "````\n<tool_call>x</tool_call>\n````"
    assert _strip(content) == content


def test_inline_code_preserves_tags():
    content = "write `<tool_call>foo</tool_call>` verbatim."
    assert _strip(content) == content


def test_tag_outside_code_still_stripped():
    content = "Before <tool_call>{'x': 1}</tool_call> after."
    assert "tool_call" not in _strip(content)
    assert "Before" in _strip(content) and "after." in _strip(content)


def test_think_tag_outside_code_still_stripped():
    content = "<think>secret</think> visible"
    assert _strip(content) == " visible"


def test_mixed_code_and_live_tag():
    fenced = "```xml\n<tool_call>\n{'name': 'x'}\n</tool_call>\n```"
    assert _strip(fenced) == fenced
