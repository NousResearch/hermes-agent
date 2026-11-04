"""Regression for #107515: discover deferred tools before declaring them missing."""

from types import SimpleNamespace

import pytest

from agent import system_prompt


@pytest.mark.parametrize(
    "names, expected",
    [
        ({"tool_search", "tool_describe", "tool_call", "terminal"}, True),
        ({"tool_describe", "tool_call", "terminal"}, False),
        (set(), False),
        ({"terminal", "read_file"}, False),
    ],
    ids=["bridge-active", "search-disabled", "empty", "ordinary-tools"],
)
def test_tool_guidance_requires_visible_tool_search(names, expected):
    agent = SimpleNamespace(valid_tool_names=names)
    block = system_prompt._tool_guidance_block(agent) or ""

    assert ("tool_search" in block) is expected
    if expected:
        assert "MCP" in block
        assert "deferred" in block
        assert "before concluding" in block


@pytest.mark.parametrize(
    "names, expected",
    [
        ({"tool_search", "tool_describe", "tool_call", "terminal"}, True),
        ({"tool_describe", "tool_call", "terminal"}, False),
        (set(), False),
        ({"terminal", "read_file"}, False),
    ],
    ids=["bridge-active", "search-disabled", "empty", "ordinary-tools"],
)
def test_full_system_prompt_requires_visible_tool_search(names, expected):
    agent = SimpleNamespace(
        valid_tool_names=names,
        load_soul_identity=False,
        skip_context_files=True,
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_diagnostic_status=lambda *_args, **_kwargs: None,
    )
    prompt = system_prompt.build_system_prompt(agent)

    assert ("tool_search" in prompt) is expected
    if expected:
        from agent.prompt_builder import TOOL_SEARCH_GUIDANCE

        assert prompt.count(TOOL_SEARCH_GUIDANCE) == 1
        assert TOOL_SEARCH_GUIDANCE in agent._cached_system_prompt_static
