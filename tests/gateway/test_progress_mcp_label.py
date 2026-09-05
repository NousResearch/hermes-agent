"""The gateway's no-verb progress line names an MCP tool by its label, never its wire name.

``TurnRunner._progress_build_message`` renders the line through the catalog templates
(``gateway.progress.tool_preview`` / ``tool_head``) fed ``display_tool_label``: the MCP
scaffolding is stripped (a third-party server stays visible as ``linear · get_issue``), and a
preview that merely repeats the tool's own name is said once.
"""
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner

TOOL = "mcp__linear__get_issue"
LABEL = "linear · get_issue"


def _runner():
    ctx = SimpleNamespace(source=None, progress_mode="all", last_was_terminal_block=[False])
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: None)
    return TurnRunner(runner, ctx)


def test_third_party_mcp_tool_renders_its_label_with_the_preview():
    line = _runner()._progress_build_message(TOOL, "ABC-1", {"id": "ABC-1"})
    assert "mcp__" not in line
    assert line.endswith(f'{LABEL}: "ABC-1"')


@pytest.mark.parametrize("preview", [TOOL, LABEL])
def test_a_preview_that_only_repeats_the_name_is_said_once(preview):
    line = _runner()._progress_build_message(TOOL, preview, {})
    assert "mcp__" not in line
    assert line.endswith(LABEL)
    assert line.count("get_issue") == 1
