"""Memory and background-review gates consume the canonical turn outcome."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


class _FinalizerAgent:
    def __init__(self, verification_status):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=89, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self._turn_verification_status = verification_status
        self._memory_manager = MagicMock()
        self.background_reviews = []

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **kwargs):
        return getattr(AIAgent, "_sync_external_memory_for_turn")(self, **kwargs)

    def _spawn_background_review(self, **kwargs):
        self.background_reviews.append(kwargs)


def _finalize(agent, *, final_response="Done.", failed=False, interrupted=False, reason="text_response(finish_reason=stop)"):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "do it"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=True,
        _turn_exit_reason=reason,
    )


@pytest.mark.parametrize(
    "outcome, verification_status, kwargs",
    [
        ("completed_unverified", "unverified", {}),
        ("partial", "passed", {"reason": "max_iterations_reached(90/90)"}),
        ("blocked", "passed", {"reason": "approval_blocked"}),
        ("failed", "passed", {"failed": True, "reason": "provider_failure"}),
        ("interrupted", "passed", {"interrupted": True, "reason": "interrupted_by_user"}),
        ("unresolved", "passed", {"reason": "tool_timeout"}),
        ("cancelled", "passed", {"reason": "cancelled"}),
    ],
)
def test_non_verified_outcomes_do_not_sync_or_spawn_review(
    monkeypatch, outcome, verification_status, kwargs
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    agent = _FinalizerAgent(verification_status)

    result = _finalize(agent, **kwargs)

    assert result["outcome"] == outcome
    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()
    assert agent.background_reviews == []


def test_verified_turn_syncs_and_spawns_review(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    agent = _FinalizerAgent("passed")

    result = _finalize(agent)

    assert result["outcome"] == "verified"
    agent._memory_manager.sync_all.assert_called_once()
    agent._memory_manager.queue_prefetch_all.assert_called_once()
    assert len(agent.background_reviews) == 1


# completed_unverified is intentionally not durable until a later policy
# explicitly permits it; this phase only allows the verified outcome.
