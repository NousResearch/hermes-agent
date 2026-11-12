"""The background review never asks the user for a skill's missing API key.

The review thread inherits the finished turn's ContextVars, including the session owner's
turn-scoped secret prompt. A ``skill_view`` of a skill with a missing required env var inside the
review must resolve to "skipped" instead of opening the masked prompt on the user's screen.
"""

from __future__ import annotations

import hermes_agent.run_agent as run_agent_module
from hermes_agent.run_agent import AIAgent
from tests.agent.test_background_review import ImmediateThread, _bare_agent
from hermes_agent.tools import skills_tool
from hermes_agent.tools.skills_tool_setup import _capture_required_environment_variables

_MISSING = [{"name": "LINEAR_API_KEY", "prompt": "Enter value for LINEAR_API_KEY"}]


def _viewer_prompt(calls):
    def prompt(var_name, prompt, metadata=None):
        calls.append(var_name)
        return {"success": True, "stored_as": var_name, "validated": False, "skipped": False}
    return prompt


def _run_review_that_opens_skill(monkeypatch):
    """Spawn a real background review whose fork loads a skill with a missing key."""
    results = []

    class SkillOpeningReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            results.append(_capture_required_environment_variables("linear", _MISSING))

        def release_clients(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", SkillOpeningReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    AIAgent._spawn_background_review(
        _bare_agent(), messages_snapshot=[{"role": "user", "content": "hello"}], review_skills=True)
    assert len(results) == 1, "the review fork did not run"
    return results[0]


def test_review_does_not_use_the_turns_secret_prompt(monkeypatch):
    calls = []
    with skills_tool.bind_turn_secret_capture(_viewer_prompt(calls)):
        result = _run_review_that_opens_skill(monkeypatch)
        assert skills_tool._turn_secret_capture.get() is not None  # the turn's own binding is intact

    assert calls == []
    assert result["missing_names"] == ["LINEAR_API_KEY"]
    assert result["setup_skipped"] is True


def test_review_does_not_use_the_cli_secret_prompt(monkeypatch):
    calls = []
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", _viewer_prompt(calls))

    result = _run_review_that_opens_skill(monkeypatch)

    assert calls == []
    assert result["setup_skipped"] is True


def test_live_turn_still_prompts_for_the_key(monkeypatch):
    """Control: outside the review the same skill load still reaches the viewer's prompt."""
    calls = []
    with skills_tool.bind_turn_secret_capture(_viewer_prompt(calls)):
        result = _capture_required_environment_variables("linear", _MISSING)

    assert calls == ["LINEAR_API_KEY"]
    assert result["missing_names"] == []
