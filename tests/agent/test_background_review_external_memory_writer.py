"""External typed-memory writer contract for unattended background reviews.

The host application owns account scoping and exposes it through an existing
named tool (usually an MCP tool).  Hermes only schedules the review and permits
that exact tool; it must never fall back to its file-backed ``memory`` tool.
"""

from __future__ import annotations

from types import SimpleNamespace

import agent.background_review as bg
from agent.turn_context import _tick_memory_nudge


class TestExternalMemoryWriterNudge:
    def test_nudge_fires_without_native_store_when_external_writer_is_available(self):
        agent = SimpleNamespace(
            _memory_nudge_interval=2,
            _turns_since_memory=1,
            _memory_store=None,
            _external_memory_writer_tool_name="memory_write",
            valid_tool_names={"memory_write"},
        )

        assert _tick_memory_nudge(agent) is True
        assert agent._turns_since_memory == 0

    def test_nudge_does_not_fire_when_configured_writer_is_unavailable(self):
        agent = SimpleNamespace(
            _memory_nudge_interval=2,
            _turns_since_memory=1,
            _memory_store=None,
            _external_memory_writer_tool_name="memory_write",
            valid_tool_names=set(),
        )

        assert _tick_memory_nudge(agent) is False
        assert agent._turns_since_memory == 1


class TestExternalMemoryWriterReviewScope:
    def test_external_writer_replaces_native_memory_in_the_review_whitelist(self):
        review_agent = SimpleNamespace(
            _memory_enabled=False,
            _user_profile_enabled=False,
            _external_memory_writer_tool_name="memory_write",
        )

        whitelist, configured = bg._review_tool_whitelist(review_agent, None, review_memory=True)

        assert "memory_write" in whitelist
        assert "memory_write" in configured
        assert "memory" not in whitelist

    def test_external_writer_review_prompt_requires_typed_proposal_payload(self):
        prompt = bg.memory_review_prompt_for_writer("memory_write")

        assert "memory_write" in prompt
        assert "typed memory proposal" in prompt
        assert "using the memory tool" not in prompt

    def test_spawn_selects_external_writer_prompt(self):
        agent = SimpleNamespace(_external_memory_writer_tool_name="memory_write")

        _target, prompt = bg.spawn_background_review_thread(agent, [], review_memory=True)

        assert "memory_write" in prompt
        assert "account-scoped" in prompt
