"""Prompt invariants for background skill review loop engineering."""


def test_skill_review_prompt_contains_loop_engineering_guardrails():
    from agent.background_review import _SKILL_REVIEW_PROMPT

    assert "Loop engineering rule" in _SKILL_REVIEW_PROMPT
    assert "one focused change" in _SKILL_REVIEW_PROMPT
    assert "Do not claim success unless" in _SKILL_REVIEW_PROMPT


def test_combined_review_prompt_contains_loop_engineering_guardrails():
    from agent.background_review import _COMBINED_REVIEW_PROMPT

    assert "Loop engineering rule" in _COMBINED_REVIEW_PROMPT
    assert "one focused change" in _COMBINED_REVIEW_PROMPT
    assert "Do not claim success unless" in _COMBINED_REVIEW_PROMPT
