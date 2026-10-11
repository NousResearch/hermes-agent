"""Memory guidance and the platform hint each open with their own heading line.

Without a heading, the memory paragraph read as the body of the section before it and the
platform hint as a continuation of whatever block preceded it. Session-search guidance and
SKILLS_GUIDANCE (whose opening sentence is filter-sensitive, #82154) must stay unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import agent.system_prompt as system_prompt
from agent.prompt_builder import (
    MEMORY_GUIDANCE_HEADING,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
)
from agent.system_prompt import PLATFORM_HINT_HEADING, build_system_prompt_parts


def _agent(platform="cli", **overrides):
    base = dict(
        platform=platform,
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names={"memory", "session_search", "skill_manage"},
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _platform_hint_overrides={},
        model="",
        provider="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def test_memory_guidance_heading_is_a_line_of_its_own():
    lines = system_prompt._tool_guidance_block(_agent()).splitlines()
    assert lines[0] == MEMORY_GUIDANCE_HEADING
    # A later volatile block is already titled "MEMORY (your personal notes)".
    assert "memory" not in MEMORY_GUIDANCE_HEADING.lower()


def test_platform_hint_heading_is_a_line_of_its_own_before_the_hint():
    lines = _stable_prompt(_agent()).splitlines()
    i = lines.index(PLATFORM_HINT_HEADING)
    assert lines[i - 1] == ""
    assert lines[i + 1] == PLATFORM_HINTS["cli"].splitlines()[0]


def test_no_platform_heading_without_a_hint():
    assert PLATFORM_HINT_HEADING not in _stable_prompt(_agent(platform=""))


def test_session_search_and_skills_guidance_get_no_heading_and_keep_their_neighbours():
    assert "\n" not in SESSION_SEARCH_GUIDANCE
    assert not SESSION_SEARCH_GUIDANCE.startswith("#")
    assert not SKILLS_GUIDANCE.startswith("#")
    # SKILLS_GUIDANCE's opening sentence is filter-sensitive: it still follows session search
    # after a single space, exactly as before.
    block = system_prompt._tool_guidance_block(_agent())
    assert block.endswith(f"skills. {SESSION_SEARCH_GUIDANCE} {SKILLS_GUIDANCE}")
