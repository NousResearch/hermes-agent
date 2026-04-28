"""Tests that the background review agent is restricted to memory+skills toolsets.

Regression coverage for issue #15204: the background skill-review agent
inherited the full default toolset, allowing it to perform non-skill side
effects (terminal, send_message, delegate_task, etc.).
"""

import threading
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent_stub():
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(AIAgent)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_agent_uses_restricted_toolsets():
    """The review agent must only have access to 'memory' and 'skills' toolsets."""
    agent = _make_agent_stub()
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
        raise RuntimeError("stop after capturing init args")

    with patch.object(AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert sorted(captured["enabled_toolsets"]) == ["memory", "skills"]


def test_background_review_agent_tools_are_limited():
    """Verify the resolved memory+skills toolsets only contain memory and skill tools."""
    from toolsets import resolve_multiple_toolsets

    expected_tools = set(resolve_multiple_toolsets(["memory", "skills"]))

    assert "memory" in expected_tools
    assert "skill_manage" in expected_tools
    assert "skill_view" in expected_tools
    assert "skills_list" in expected_tools

    assert "terminal" not in expected_tools
    assert "send_message" not in expected_tools
    assert "delegate_task" not in expected_tools
    assert "web_search" not in expected_tools
    assert "execute_code" not in expected_tools
