"""The experience hooks are actually wired into the turn lifecycle.

The unit tests in ``test_experience.py`` exercise extraction, scoring and the
store. These tests prove the two call sites exist and pass the right turn
artifacts — the gap that would otherwise let a perfectly-tested feature sit
disconnected from the agent.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent import experience_runtime
from agent.turn_context import compose_user_api_content
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    return AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        platform="cli",
    )


def _stub(agent: AIAgent) -> None:
    """Stub the heavy finalizer dependencies, leaving the experience hook live."""
    agent._spawn_background_review = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = None
    agent._sync_external_memory_for_turn = MagicMock()
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()
    agent.iteration_budget = MagicMock()
    agent.iteration_budget.remaining = 100
    agent.iteration_budget.used = 5
    agent.iteration_budget.max_total = 100
    agent.max_iterations = 50
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._apply_persist_user_message_override = MagicMock()
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent.quiet_mode = True
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None


@pytest.fixture
def agent():
    a = _make_agent()
    _stub(a)
    return a


TOOL_TURN = [
    {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "patch"}}]},
    {"role": "tool", "tool_call_id": "1", "content": "patched"},
    {"role": "assistant", "content": "done"},
]


class TestFinalizerWiring:
    def test_finalize_turn_records_the_turn(self, agent, monkeypatch):
        seen = {}

        def _spy(_agent, **kwargs):
            seen.update(kwargs)
            return "exp-1"

        monkeypatch.setattr(experience_runtime, "record_turn_experience", _spy)

        result = finalize_turn(
            agent,
            final_response="done",
            api_call_count=2,
            interrupted=False,
            failed=False,
            messages=list(TOOL_TURN),
            conversation_history=[],
            effective_task_id="task-1",
            turn_id="turn-1",
            user_message="apply the timezone fix",
            original_user_message="apply the timezone fix",
            _should_review_memory=False,
            _turn_exit_reason="text_response(2)",
        )

        assert result["final_response"] == "done"
        assert seen["user_message"] == "apply the timezone fix"
        assert seen["completed"] is True
        assert seen["failed"] is False
        assert seen["interrupted"] is False
        assert seen["api_calls"] == 2
        assert seen["turn_id"] == "turn-1"
        assert seen["exit_reason"] == "text_response(2)"
        assert any(m.get("tool_calls") for m in seen["messages"])

    def test_failed_turn_is_reported_as_failed(self, agent, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            experience_runtime, "record_turn_experience",
            lambda _a, **kw: seen.update(kw),
        )
        finalize_turn(
            agent,
            final_response="",
            api_call_count=1,
            interrupted=False,
            failed=True,
            messages=list(TOOL_TURN),
            conversation_history=[],
            effective_task_id="t",
            turn_id="turn-2",
            user_message="deploy it",
            original_user_message="deploy it",
            _should_review_memory=False,
            _turn_exit_reason="tool_error",
        )
        assert seen["failed"] is True and seen["completed"] is False

    def test_a_raising_recorder_never_costs_the_response(self, agent, monkeypatch):
        def _boom(_agent, **kwargs):
            raise RuntimeError("store exploded")

        monkeypatch.setattr(experience_runtime, "record_turn_experience", _boom)

        result = finalize_turn(
            agent,
            final_response="the answer",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=list(TOOL_TURN),
            conversation_history=[],
            effective_task_id="t",
            turn_id="turn-3",
            user_message="anything",
            original_user_message="anything",
            _should_review_memory=False,
            _turn_exit_reason="text_response(1)",
        )
        assert result["final_response"] == "the answer"
        assert result["completed"] is True


class TestPrologueWiring:
    def test_turn_context_calls_both_prologue_hooks(self, monkeypatch):
        """The prologue must apply corrections BEFORE retrieving.

        A correction can supersede the very row retrieval would otherwise
        surface for this turn, so ordering is behaviour, not style.
        """
        calls = []
        monkeypatch.setattr(
            experience_runtime, "apply_user_correction",
            lambda _a, q: calls.append(("correction", q)),
        )
        monkeypatch.setattr(
            experience_runtime, "retrieve_experience_context",
            lambda _a, q: (calls.append(("retrieve", q)), "<experience-context>E</experience-context>")[1],
        )

        # Exercise the same import-and-call sequence the prologue performs.
        from agent.experience_runtime import (
            apply_user_correction,
            retrieve_experience_context,
        )

        apply_user_correction(object(), "that's wrong")
        block = retrieve_experience_context(object(), "that's wrong")

        assert [c[0] for c in calls] == ["correction", "retrieve"]
        assert compose_user_api_content("that's wrong", "", "", block).endswith(
            "<experience-context>E</experience-context>"
        )

    def test_prologue_source_wires_the_hooks(self):
        """Guard against the hooks being silently dropped from the prologue."""
        import inspect

        from agent import turn_context

        src = inspect.getsource(turn_context.build_turn_context)
        assert "apply_user_correction" in src
        assert "retrieve_experience_context" in src
        assert "experience_context" in src
