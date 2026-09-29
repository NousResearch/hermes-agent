"""Unified employee review scope, participant handoff and atomic consolidation."""

from __future__ import annotations

import os
import sys
import pytest
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
    @pytest.mark.parametrize("review_memory", [True, False])
    def test_unified_review_has_only_knowledge_tools(self, review_memory):
        whitelist, extra = bg._review_tool_whitelist(
            _review_agent(), {"extra_tools": ["send_message"]}, review_memory=review_memory)
        assert {"memory", "read_file", "search_files", "write_file", "patch"} == whitelist
        assert not extra



class TestSpawnForwardsScope:
    def test_target_passes_review_memory_to_worker(self):
        captured = {}

        def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                        review_memory=False, explicit=False, person_snapshot=None):
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


class TestExplicitRefineOrigin:
    """``/refine`` (explicit) must not inherit the unattended-review origin: the user asked
    for that review, so its fork keeps the full memory operation set and the delete gate
    does not apply (#105921 review follow-up)."""

    def test_target_passes_explicit_to_worker(self):
        captured = {}

        def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                        review_memory=False, explicit=False, person_snapshot=None):
            captured["explicit"] = explicit

        with patch.object(bg, "_run_review_in_thread", fake_worker):
            target, _prompt = bg.spawn_background_review_thread(
                SimpleNamespace(), [], review_memory=True, explicit=True)
            target()
            assert captured["explicit"] is True

    @pytest.mark.parametrize("explicit", [True, False])
    def test_fork_keeps_background_review_origin_and_carries_attendedness(self, explicit):
        """Every curator/skill guard keys on the background_review origin, so an explicit /refine
        must NOT change the origin; attendedness rides on the fork as its own flag."""
        forks = []

        def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
            fork = SimpleNamespace(
                _memory_enabled=True, _user_profile_enabled=False, _memory_write_origin=write_origin,
                run_conversation=lambda **kw: None, _session_messages=[])
            forks.append(fork)
            return fork, {}, False

        noop = lambda *a, **k: None
        with patch.object(bg, "build_cache_parity_fork", fake_build), \
                patch.object(bg, "_track_review_fork", noop), \
                patch.object(bg, "_snapshot_review_usage", lambda a: {}), \
                patch.object(bg, "_record_review_usage_to_parent", noop), \
                patch.object(bg, "finish_background_review_run", noop), \
                patch.object(bg, "_release_fork_clients", noop):
            bg._run_review_fork(SimpleNamespace(), [], "p", None, None, bg._ReviewForkState(), True, explicit)
        (fork,) = forks
        assert fork._memory_write_origin == "background_review"
        assert fork._review_attended is explicit


class TestConsolidationWrites:
    def test_review_consolidates_atomically_after_budget_denial(self, tmp_path, monkeypatch):
        import json
        from tools.memory_tool import memory_tool
        from tools.memory_tool_store import MemoryStore
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = MemoryStore(memory_char_limit=500)
        store.load_from_disk()
        assert store.add("memory", "seed entry one")["success"]
        token = set_current_write_origin("background_review")
        try:
            denied = json.loads(memory_tool(action="add", content="x" * 600, store=store))
            assert not denied["success"]
            assert store._entries_for("memory") == ["seed entry one"]
            result = json.loads(memory_tool(action="replace", old_text="seed entry one",
                                           content="merged entry", store=store))
        finally:
            reset_current_write_origin(token)
        assert result["success"] and not result.get("staged")
        reloaded = MemoryStore(memory_char_limit=500)
        reloaded.load_from_disk()
        assert reloaded._entries_for("memory") == ["merged entry"]
