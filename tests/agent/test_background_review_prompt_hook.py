"""A plugin's ``background_review_prompt`` hook replaces the background review request."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent.background_review import _COMBINED_REVIEW_PROMPT, _SKILL_REVIEW_PROMPT, spawn_background_review_thread
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def review_prompt_hooks():
    mgr = get_plugin_manager()
    saved = {key: list(value) for key, value in mgr._hooks.items()}
    callbacks = mgr._hooks.setdefault("background_review_prompt", [])
    try:
        yield callbacks
    finally:
        mgr._hooks = saved


def test_first_plugin_prompt_replaces_the_review_request(review_prompt_hooks):
    calls = []

    def observe(**kwargs):
        calls.append(kwargs)

    review_prompt_hooks.extend([
        observe,
        lambda **_: {"prompt": "   "},
        lambda **_: "not a dict",
        lambda **_: {"prompt": "Custom review."},
        lambda **_: {"prompt": "Later review."},
    ])
    _target, prompt = spawn_background_review_thread(
        SimpleNamespace(session_id="s1"), [], review_memory=True, review_skills=True, task_cfg={},
        focus="the tests", explicit=True,
    )
    assert prompt.startswith("Custom review.") and prompt.endswith("the tests")
    assert "Later review." not in prompt
    assert {key: calls[0][key] for key in ("prompt", "review_memory", "review_skills", "explicit", "session_id")} == {
        "prompt": _COMBINED_REVIEW_PROMPT, "review_memory": True, "review_skills": True,
        "explicit": True, "session_id": "s1",
    }
    assert "background_review_prompt" in VALID_HOOKS


def test_without_a_plugin_prompt_the_default_request_is_kept(review_prompt_hooks):
    review_prompt_hooks.append(lambda **_: None)
    _target, prompt = spawn_background_review_thread(
        SimpleNamespace(session_id="s1"), [], review_skills=True, task_cfg={},
    )
    assert prompt == _SKILL_REVIEW_PROMPT


def test_a_hung_plugin_leaves_the_default_request(review_prompt_hooks, monkeypatch):
    release = threading.Event()

    def hung(**_):
        release.wait(10)
        return {"prompt": "Late review."}

    review_prompt_hooks.append(hung)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.2)
    try:
        _target, prompt = spawn_background_review_thread(
            SimpleNamespace(session_id="s1"), [], review_skills=True, task_cfg={},
        )
    finally:
        release.set()
    assert prompt == _SKILL_REVIEW_PROMPT
