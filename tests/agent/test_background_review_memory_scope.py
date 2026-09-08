"""The review fork's memory access must follow the trigger that fired (#105921).

``spawn_background_review_thread`` already received ``review_memory`` /
``review_skills`` but used them only to pick the prompt — the tool whitelist
granted the whole ``memory`` toolset whenever the profile had memory enabled,
so a skill-nudge fork held ``remove``/``replace`` on MEMORY.md it was never
asked to use. These tests pin the scope-aware whitelist and the pass-through
from ``spawn_background_review_thread`` down to it.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg  # noqa: E402


def _review_agent(memory_enabled=True, user_profile_enabled=False) -> SimpleNamespace:
    """The whitelist only reads the profile's memory flags off the fork."""
    return SimpleNamespace(_memory_enabled=memory_enabled, _user_profile_enabled=user_profile_enabled)


class TestReviewToolWhitelistScope:
    def test_skill_only_review_omits_memory_tool(self):
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None, review_memory=False)
        assert "memory" not in whitelist
        assert "skill_manage" in whitelist  # the skill review keeps its own surface

    def test_memory_review_keeps_memory_tool(self):
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None, review_memory=True)
        assert "memory" in whitelist

    def test_memory_disabled_profile_stays_memory_free(self):
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(memory_enabled=False, user_profile_enabled=False), None, review_memory=True)
        assert "memory" not in whitelist

    def test_default_scope_is_memoryless_fail_closed(self):
        # Unknown trigger (default) must not grant memory: the incident fork was a
        # skill review that used memory nobody asked it to touch.
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None)
        assert "memory" not in whitelist


class TestSpawnForwardsScope:
    def test_target_passes_review_memory_to_worker(self):
        captured = {}

        def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None, review_memory=False):
            captured["review_memory"] = review_memory

        agent = SimpleNamespace()
        with patch.object(bg, "_run_review_in_thread", fake_worker):
            target, _prompt = bg.spawn_background_review_thread(
                agent, [], review_memory=False, review_skills=True)
            target()
            assert captured["review_memory"] is False

            target, _prompt = bg.spawn_background_review_thread(
                agent, [], review_memory=True, review_skills=False)
            target()
            assert captured["review_memory"] is True
