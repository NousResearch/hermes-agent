"""``skills.creation_nudge_interval: 0`` must be absolute for unattended review forks (#82708).

The switch only gates the skill-review trigger; the memory-triggered fork still carried the
whole ``skills`` toolset, so autonomous ``skill_manage`` writes continued against an explicit
user configuration. These tests pin the whitelist gate and its pass-through from
``_run_review_fork`` (an explicit ``/refine`` keeps the skill surface — the user asked for it).
"""

from __future__ import annotations

import os
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg


def _review_agent(memory_enabled=True, user_profile_enabled=False) -> SimpleNamespace:
    """The whitelist only reads the profile's memory flags off the fork."""
    return SimpleNamespace(_memory_enabled=memory_enabled, _user_profile_enabled=user_profile_enabled)


class TestKillSwitchWhitelistScope:
    def test_kill_switch_strips_skills_from_memory_fork(self):
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(), None, review_memory=True, skills_creation_on=False)
        assert "skill_manage" not in whitelist
        assert "skill_view" not in whitelist and "skills_list" not in whitelist
        assert "memory" in whitelist  # the memory surface this fork exists for stays
        assert {"read_file", "search_files"} <= whitelist

    def test_kill_switch_keeps_memory_surface(self):
        # The fork must not lose its actual job because the unrelated skill switch is off.
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(), None, review_memory=True, skills_creation_on=False)
        assert "memory" in whitelist

    def test_memoryless_profile_and_kill_switch_leave_read_only_fork(self):
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(memory_enabled=False, user_profile_enabled=False), None,
            review_memory=True, skills_creation_on=False)
        assert "memory" not in whitelist and "skill_manage" not in whitelist
        assert {"read_file", "search_files"} <= whitelist

    def test_default_keeps_skills_surface(self):
        # Regression: with the switch unset (default on) the whitelist is unchanged.
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None, review_memory=True)
        assert "skill_manage" in whitelist and "memory" in whitelist

    def test_skill_only_review_keeps_skills_by_default(self):
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(), None, review_memory=False)
        assert "skill_manage" in whitelist


def _run_fork_capturing_whitelist(parent_agent):
    """Drive ``_run_review_fork`` with the build/teardown seams stubbed; return the whitelist
    handed to ``set_thread_tool_whitelist``."""
    captured = {}

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        fork = SimpleNamespace(
            _memory_enabled=True, _user_profile_enabled=False, _memory_write_origin=write_origin,
            run_conversation=lambda **kw: None, _session_messages=[])
        return fork, {}, False

    from hermes_cli import plugins

    def fake_set(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = whitelist
        captured["deny_msg"] = deny_msg_fmt or ""

    noop = lambda *a, **k: None
    with patch.object(bg, "build_cache_parity_fork", fake_build), \
            patch.object(bg, "_track_review_fork", noop), \
            patch.object(bg, "_snapshot_review_usage", lambda a: {}), \
            patch.object(bg, "_record_review_usage_to_parent", noop), \
            patch.object(bg, "finish_background_review_run", noop), \
            patch.object(bg, "_release_fork_clients", noop), \
            patch.object(plugins, "set_thread_tool_whitelist", fake_set), \
            patch.object(plugins, "clear_thread_tool_whitelist", noop):
        bg._run_review_fork(
            parent_agent, [], "p", None, None, bg._ReviewForkState(), True, False)
    return captured


class TestForkPassThrough:
    def test_unattended_fork_honors_kill_switch(self):
        parent = SimpleNamespace(_skill_nudge_interval=0)
        captured = _run_fork_capturing_whitelist(parent)
        assert "skill_manage" not in captured["whitelist"]
        assert "memory" in captured["whitelist"]
        # The deny wording must not advertise a skill surface the fork no longer has.
        assert "skill_manage" not in captured["deny_msg"]

    def test_unattended_fork_keeps_skills_when_nudge_on(self):
        parent = SimpleNamespace(_skill_nudge_interval=10)
        captured = _run_fork_capturing_whitelist(parent)
        assert "skill_manage" in captured["whitelist"]

    def test_explicit_refine_bypasses_kill_switch(self):
        # /refine is the user's own request: the skill surface rides along even when the
        # autonomous-creation switch is off — the user is in the loop.
        parent = SimpleNamespace(_skill_nudge_interval=0)

        def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
            fork = SimpleNamespace(
                _memory_enabled=True, _user_profile_enabled=False, _memory_write_origin=write_origin,
                run_conversation=lambda **kw: None, _session_messages=[])
            return fork, {}, False

        captured = {}
        from hermes_cli import plugins

        def fake_set(whitelist, deny_msg_fmt=None):
            captured["whitelist"] = whitelist

        noop = lambda *a, **k: None
        with patch.object(bg, "build_cache_parity_fork", fake_build), \
                patch.object(bg, "_track_review_fork", noop), \
                patch.object(bg, "_snapshot_review_usage", lambda a: {}), \
                patch.object(bg, "_record_review_usage_to_parent", noop), \
                patch.object(bg, "finish_background_review_run", noop), \
                patch.object(bg, "_release_fork_clients", noop), \
                patch.object(plugins, "set_thread_tool_whitelist", fake_set), \
                patch.object(plugins, "clear_thread_tool_whitelist", noop):
            bg._run_review_fork(
                parent, [], "p", None, None, bg._ReviewForkState(), True, True)
        assert "skill_manage" in captured["whitelist"]

    def test_parent_without_interval_attribute_keeps_skills(self):
        # Bare SimpleNamespace agents (older callers, tests) must not fail closed on a
        # missing attribute: the config default is 10 (on).
        captured = _run_fork_capturing_whitelist(SimpleNamespace())
        assert "skill_manage" in captured["whitelist"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
