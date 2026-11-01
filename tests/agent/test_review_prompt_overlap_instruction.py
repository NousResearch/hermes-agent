"""Regression tests for the review prompts' skill-overlap instruction (#141929).

The review fork's final reply has no consumer: the published summary is built
from succeeded memory/skill tool results (``summarize_background_review_actions``),
so no review prompt may ask the fork to leave overlap notes in that reply. The
never-delete guard must stay in both prompts.
"""

import pytest

from agent.background_review import _COMBINED_REVIEW_PROMPT, _SKILL_REVIEW_PROMPT


@pytest.mark.parametrize(
    "prompt_name,prompt",
    [
        ("skill review", _SKILL_REVIEW_PROMPT),
        ("combined review", _COMBINED_REVIEW_PROMPT),
    ],
)
def test_prompts_do_not_ask_for_reply_only_overlap_notes(prompt_name, prompt):
    lowered = prompt.lower()
    # Old instruction asked the fork to note overlaps in its final reply, which
    # nobody reads — the note went nowhere (and consolidation is off by default).
    assert "note it in your reply" not in lowered
    assert "mention it" not in lowered
    assert "curator handles" not in lowered
    # The replacement points the fork at actions that do have a consumer:
    # patch the existing skill instead of creating a near-duplicate.
    assert "do not leave a note about it in your reply" in lowered
    assert "instead of creating a near-duplicate" in lowered


@pytest.mark.parametrize(
    "prompt_name,prompt",
    [
        ("skill review", _SKILL_REVIEW_PROMPT),
        ("combined review", _COMBINED_REVIEW_PROMPT),
    ],
)
def test_prompts_keep_the_never_delete_guard(prompt_name, prompt):
    lowered = prompt.lower()
    assert "never delete a skill" in lowered