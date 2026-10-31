"""Formatting of the ``tool_progress: log`` line written to tool_calls.log."""

from gateway.run_turn_progress import tool_log_line

TS = "2026-10-25 01:17:15"


def test_tool_without_preview_has_no_dangling_separator():
    assert tool_log_line(TS, "skill_manage", None) == f"{TS}  skill_manage"
    assert tool_log_line(TS, "todo", "") == f"{TS}  todo"


def test_preview_is_one_json_string_with_inner_quotes_escaped():
    # A terminal preview embeds shell quoting; raw quotes made the line unparseable.
    line = tool_log_line(TS, "terminal", 'grep -E "hermes.*(gateway)" /数据/a.csv')
    assert line == f'{TS}  terminal: "grep -E \\"hermes.*(gateway)\\" /数据/a.csv"'
