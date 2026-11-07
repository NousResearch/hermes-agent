"""Regression tests for the review prompts' repair-an-unloaded-skill rule (#66180).

When a pitfall is already stated by an existing skill but that skill was never
loaded in the session being reviewed, appending another warning to the skill
changes nothing — the loading decision is made from the description. Both
review prompts must carry the rule: rewrite that skill's description as a
clear "Use when <trigger>" line instead of appending a duplicate warning.
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
def test_both_prompts_check_for_an_already_stated_lesson(prompt_name, prompt):
    lowered = prompt.lower()
    # The condition: the lesson already exists in an unloaded skill.
    assert "already stated by an existing skill" in lowered
    assert "was never loaded this session" in lowered
    # The verdict: another warning is inert.
    assert "appending another warning changes nothing" in lowered
    # The required action: repair the trigger description.
    assert "patch that skill's description instead" in lowered
    assert "'use when" in lowered