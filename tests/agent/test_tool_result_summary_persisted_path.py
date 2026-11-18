"""Compressed tool-result summaries retain spillover recovery paths."""

from agent.context_compressor import _summarize_tool_result


PERSISTED = "<persisted-output>\nFull output saved to: /Users/test/cache/spill.txt\nPreview\n</persisted-output>"


def test_terminal_summary_keeps_persisted_output_path():
    summary = _summarize_tool_result(
        "terminal", '{"command":"printf hi"}',
        '{"exit_code":0}\n' + PERSISTED,
    )
    assert summary.endswith("; full output: /Users/test/cache/spill.txt")


def test_execute_code_summary_keeps_persisted_output_path():
    summary = _summarize_tool_result(
        "execute_code", '{"code":"print(1)"}',
        PERSISTED,
    )
    assert summary.endswith("; full output: /Users/test/cache/spill.txt")


def test_summary_omits_path_when_result_was_not_persisted():
    summary = _summarize_tool_result(
        "terminal", '{"command":"pwd"}', '{"exit_code":0}\n/workspace\n',
    )
    assert "full output:" not in summary
