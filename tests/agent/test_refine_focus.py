"""Tests for the /refine focus parameter on spawn_background_review_thread."""

from unittest.mock import MagicMock

from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _MEMORY_REVIEW_PROMPT,
    spawn_background_review_thread,
)


def _bare_agent():
    agent = MagicMock()
    # Ensure getattr(agent, "_COMBINED_REVIEW_PROMPT", default) hits the default.
    del agent._COMBINED_REVIEW_PROMPT
    del agent._MEMORY_REVIEW_PROMPT
    del agent._SKILL_REVIEW_PROMPT
    return agent


def test_no_focus_prompt_is_byte_identical():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True
    )
    assert prompt == _COMBINED_REVIEW_PROMPT

    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, focus=None
    )
    assert prompt == _COMBINED_REVIEW_PROMPT

    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, focus="   "
    )
    assert prompt == _COMBINED_REVIEW_PROMPT


def test_focus_is_appended_to_prompt():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True,
        focus="save the deploy workflow as a skill",
    )
    assert prompt.startswith(_COMBINED_REVIEW_PROMPT)
    assert "save the deploy workflow as a skill" in prompt
    assert "explicitly requested" in prompt


def test_focus_works_with_memory_only_prompt():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=False, focus="remember my timezone",
    )
    assert prompt.startswith(_MEMORY_REVIEW_PROMPT)
    assert "remember my timezone" in prompt
