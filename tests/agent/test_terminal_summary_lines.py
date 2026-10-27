"""Terminal summary line-count regressions for #137105."""

import json

import pytest

from agent.context_compressor import _summarize_tool_result


def test_terminal_summary_counts_output_lines_not_json_envelope_lines():
    """Regression for #137105: JSON escapes terminal-output newlines."""
    content = json.dumps({"output": "first\nsecond\nthird", "exit_code": 0})

    assert _summarize_tool_result("terminal", json.dumps({"command": "probe"}), content) == (
        "[terminal] ran `probe` -> exit 0, 3 lines output"
    )


@pytest.mark.parametrize("output, lines", [
    ("hello", 1), ("hello\n", 1), ("a\nb", 2), ("a\nb\n", 2),
    ("", 0), ("\n", 1), ("hello\r\n", 1), ("a\r\nb\r\n", 2),
    ("\r\n", 1), ("a\n\n", 2),
])
def test_terminal_summary_counts_logical_output_lines(output, lines):
    content = json.dumps({"output": output, "exit_code": 0})
    assert _summarize_tool_result("terminal", '{"command": "probe"}', content) == (
        f"[terminal] ran `probe` -> exit 0, {lines} lines output"
    )


@pytest.mark.parametrize("content, lines", [
    ("hello\n", 2), ('{"exit_code": 0}', 1),
    ('{"output": null}', 1), ("", 0),
])
def test_terminal_summary_preserves_fallback_line_count(content, lines):
    summary = _summarize_tool_result("terminal", '{"command": "probe"}', content)
    assert summary.endswith(f", {lines} lines output")
