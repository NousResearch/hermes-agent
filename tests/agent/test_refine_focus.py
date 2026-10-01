"""Tests for the /refine focus parameter on spawn_background_review_thread."""

from types import SimpleNamespace

from agent.employee_review import PROMPT
from agent.knowledge import render

from agent.background_review import (
    spawn_background_review_thread,
)


def _bare_agent():
    return SimpleNamespace()

def test_no_focus_prompt_is_byte_identical():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True
    )
    assert prompt == render(PROMPT)

    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, focus=None
    )
    assert prompt == render(PROMPT)

    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, focus="   "
    )
    assert prompt == render(PROMPT)


def test_focus_is_appended_to_prompt():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True,
        focus="save the deploy workflow in the manual",
    )
    assert prompt.startswith(render(PROMPT))
    assert "save the deploy workflow in the manual" in prompt


def test_focus_works_with_memory_only_prompt():
    agent = _bare_agent()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=False, focus="remember my timezone",
    )
    assert prompt.startswith(render(PROMPT))
    assert "remember my timezone" in prompt
