"""The Kanban worker protocol must start its own paragraph in the tool-guidance block.

``_tool_guidance_block`` joins the memory, session-search and skills paragraphs with a
space. ``KANBAN_GUIDANCE`` opens with a ``#`` heading, so joining it the same way put
``# Kanban task execution protocol`` mid-line after the memory paragraph: the heading
stopped being a heading and its ``##`` sections read as part of the block above.
"""

from __future__ import annotations

from types import SimpleNamespace

import agent.system_prompt as system_prompt
from agent.prompt_builder import KANBAN_GUIDANCE

_HEADING = KANBAN_GUIDANCE.splitlines()[0]


def _worker(valid_tool_names):
    return SimpleNamespace(valid_tool_names=set(valid_tool_names), _kanban_worker_guidance=KANBAN_GUIDANCE)


def test_kanban_heading_starts_its_own_paragraph_after_memory_guidance():
    block = system_prompt._tool_guidance_block(
        _worker({"memory", "session_search", "skill_manage", "kanban_show"})
    )
    assert block is not None
    assert f"\n\n{_HEADING}\n" in block
    assert _HEADING in block.splitlines()


def test_kanban_guidance_alone_is_returned_unchanged():
    assert system_prompt._tool_guidance_block(_worker({"kanban_show"})) == KANBAN_GUIDANCE
