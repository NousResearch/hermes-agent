"""Regression tests for budget-exhausted turn finalization."""

from types import SimpleNamespace
from unittest.mock import Mock

from agent.iteration_budget import IterationBudget
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


class FakeAgent:
    def __init__(self, *, max_iterations=10, budget_max=10, budget_used=0, summary="progress"):
        self.max_iterations = max_iterations
        self.iteration_budget = IterationBudget(budget_max)
        for _ in range(budget_used):
            assert self.iteration_budget.consume()
        self.summary = summary
        self.quiet_mode = True
        self.model = "test/model"
        self.session_id = "session-1"
        self.provider = "test-provider"
        self.base_url = "https://example.invalid"
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._interrupt_message = None
        self._stream_callback = Mock()
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self._turn_failed_file_mutations = {}

        self._emit_status = Mock()
        self._safe_print = Mock()
        self._save_trajectory = Mock()
        self._cleanup_task_resources = Mock()
        self._drop_trailing_empty_response_scaffolding = Mock()
        self._persist_session = Mock()
        self.clear_interrupt = Mock()
        self._sync_external_memory_for_turn = Mock()
        self._spawn_background_review = Mock()

    def _handle_max_iterations(self, messages, api_call_count):
        return self.summary

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return True

    @staticmethod
    def _format_turn_completion_explanation(reason):
        return AIAgent._format_turn_completion_explanation(reason)

    def _drain_pending_steer(self):
        return None


def _finalize(agent, *, final_response=None, api_call_count=0, messages=None, reason="loop_exited"):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages or [{"role": "user", "content": "do work"}],
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do work",
        original_user_message="do work",
        _should_review_memory=False,
        _turn_exit_reason=reason,
    )


def test_max_iterations_exhaustion_returns_incomplete_metadata_and_warning():
    agent = FakeAgent(max_iterations=2, budget_max=10, budget_used=1, summary="Done, everything is complete.")

    result = _finalize(agent, api_call_count=2)

    assert result["completed"] is False
    assert result["budget_exhausted"] is True
    assert result["failure_reason"] == "budget_exhausted"
    assert result["turn_exit_reason"] == "max_iterations_reached(2/2)"
    assert result["budget_used"] == 1
    assert result["budget_max"] == 10
    assert "Iteration budget exhausted" in result["final_response"]
    assert "not verified complete" in result["final_response"]
    assert "Progress summary from the model" in result["final_response"]
    assert "Done, everything is complete." in result["final_response"]


def test_shared_iteration_budget_exhaustion_is_not_success_when_api_calls_below_max():
    agent = FakeAgent(max_iterations=10, budget_max=3, budget_used=3, summary="I think this is done.")

    result = _finalize(agent, api_call_count=2)

    assert result["completed"] is False
    assert result["budget_exhausted"] is True
    assert result["failure_reason"] == "budget_exhausted"
    assert result["turn_exit_reason"] == "iteration_budget_exhausted(3/3)"
    assert result["budget_used"] == 3
    assert result["budget_max"] == 3
    assert "Iteration budget exhausted" in result["final_response"]
    assert "I think this is done." in result["final_response"]


def test_budget_exhaustion_after_tool_result_does_not_look_successful():
    agent = FakeAgent(max_iterations=10, budget_max=1, budget_used=1, summary="Done.")
    messages = [
        {"role": "user", "content": "patch it"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
        {"role": "tool", "content": "patched"},
    ]

    result = _finalize(agent, api_call_count=1, messages=messages)

    assert result["completed"] is False
    assert result["budget_exhausted"] is True
    assert result["final_response"].startswith("⚠️ Iteration budget exhausted")
    assert "not verified complete" in result["final_response"]


def test_budget_warning_survives_empty_or_failed_summary_generation():
    class RaisingAgent(FakeAgent):
        def _handle_max_iterations(self, messages, api_call_count):
            raise RuntimeError("summary broke")

    agent = RaisingAgent(max_iterations=1, budget_max=1, budget_used=1)

    result = _finalize(agent, api_call_count=1)

    assert result["completed"] is False
    assert result["budget_exhausted"] is True
    assert "Iteration budget exhausted" in result["final_response"]
    assert "summary broke" not in result["final_response"]
    assert "send `continue`" in result["final_response"].lower()


def test_budget_warning_survives_transform_hook_overclaim(monkeypatch):
    agent = FakeAgent(max_iterations=1, budget_max=1, budget_used=1, summary="progress")

    def fake_invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return ["Done, everything is complete."]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    result = _finalize(agent, api_call_count=1)

    assert result["completed"] is False
    assert result["budget_exhausted"] is True
    assert result["final_response"].startswith("⚠️ Iteration budget exhausted")
    assert "not verified complete" in result["final_response"]
    assert "Progress summary from the model" in result["final_response"]
    assert "Done, everything is complete." in result["final_response"]


def test_non_budget_success_does_not_get_warning():
    agent = FakeAgent(max_iterations=10, budget_max=10, budget_used=1)

    result = _finalize(agent, final_response="Done.", api_call_count=1, reason="text_response(finish_reason=stop)")

    assert result["completed"] is True
    assert result["budget_exhausted"] is False
    assert result["final_response"] == "Done."
