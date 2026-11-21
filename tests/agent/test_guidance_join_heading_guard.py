"""The tool-guidance join must never glue a heading part onto the previous line.

``_tool_guidance_block`` space-joins its parts. #133698 was the instance: ``KANBAN_GUIDANCE``
opens with ``# Kanban task execution protocol`` and landed mid-line after the memory paragraph.
The guard covers any part whose first line is a heading, not just that one.
"""

from __future__ import annotations

from types import SimpleNamespace

import agent.system_prompt as system_prompt
from agent.prompt_builder import KANBAN_GUIDANCE
from agent.system_prompt import _join_guidance


def test_heading_part_starts_its_own_line_after_a_blank_line():
    joined = _join_guidance(["Prose about memory.", "# Some protocol\nBody text."])
    assert joined == "Prose about memory.\n\n# Some protocol\nBody text."
    assert "# Some protocol" in joined.splitlines()


def test_prose_only_join_is_byte_identical_to_a_space_join():
    parts = ["First paragraph.", "Second sentence.", "Third one."]
    assert _join_guidance(parts) == " ".join(parts)


def test_existing_blank_line_is_not_doubled():
    assert _join_guidance(["Prose.\n\n", "# Heading\nBody."]) == "Prose.\n\n# Heading\nBody."
    assert _join_guidance(["Prose.", "\n\n# Heading\nBody."]) == "Prose.\n\n# Heading\nBody."


def test_hash_inside_a_paragraph_is_not_a_new_section():
    parts = ["Prose.", "See issue #82154; a line like # not-a-heading stays.\n# also mid-part"]
    assert _join_guidance(parts) == " ".join(parts)
    assert _join_guidance(["Prose.", "#82154 is an issue number, not a heading."]) == (
        "Prose. #82154 is an issue number, not a heading."
    )


def test_kanban_worker_guidance_heading_is_on_its_own_line():
    agent = SimpleNamespace(
        valid_tool_names={"memory", "session_search", "skill_manage", "kanban_show"},
        _kanban_worker_guidance=KANBAN_GUIDANCE,
    )
    block = system_prompt._tool_guidance_block(agent)
    assert f"\n\n{KANBAN_GUIDANCE.splitlines()[0]}\n" in block
