"""Exact model-authored tool-name contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


VALID = {
    "todo",
    "patch",
    "browser_click",
    "web_search",
    "write_file",
    "terminal",
}


@pytest.fixture
def repair():
    """Return the compatibility shim bound to a minimal agent."""
    from run_agent import AIAgent

    stub = SimpleNamespace(valid_tool_names=VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


@pytest.mark.parametrize("name", sorted(VALID))
def test_exact_registered_name_is_preserved(repair, name):
    assert repair(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "TERMINAL",
        "terminall",
        "web-search",
        "write file",
        "WriteFileTool",
        "Patch_tool",
        'terminal" parameter="command" string="true',
        "browser_click ",
        "",
        None,
    ],
)
def test_non_exact_name_is_never_mapped_to_executable_tool(repair, name):
    assert repair(name) is None
