"""Per-turn tool-call cap tied to the iteration limit (#131098).

The turn loop bounds LLM calls (api_call_count < max_iterations) but each
iteration can execute several tools, so total tool executions can blow past the
turn limit (142 tool calls against max_turns=100) and the budget-exhaustion
summary degrades. Total executed tools per turn must stay within the turn
limit, and a tool-capped turn must end as a coherent max_iterations_reached
handoff -- not a failure.
"""

from types import SimpleNamespace

import pytest

from agent.conversation_loop import _LoopState
from agent.turn_failure_copy import is_max_iteration_handoff
from agent.turn_finalizer import finalize_turn
from agent.turn_tool_round import max_turn_tool_calls, run_tool_round


def test_cap_is_tied_to_the_turn_limit():
    assert max_turn_tool_calls(SimpleNamespace(max_iterations=100)) == 100


def test_cap_is_inert_when_the_limit_is_unparseable():
    assert max_turn_tool_calls(SimpleNamespace(max_iterations="zzz")) > 10**9
    assert max_turn_tool_calls(SimpleNamespace()) > 10**9


def test_loop_state_tracks_executed_tools_per_turn():
    assert _LoopState.__dataclass_fields__["tool_call_count"].default == 0


def _capped_agent():
    return SimpleNamespace(
        max_iterations=3,
        quiet_mode=True,
        verbose_logging=False,
        _safe_print=lambda *a, **k: None,
    )


def _assistant_message(n):
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name="t", arguments="{}"),
                id=f"call-{i}",
            )
            for i in range(n)
        ],
        content="",
    )


def test_tool_round_at_cap_executes_nothing_and_ends_the_turn():
    verdict = run_tool_round(
        _capped_agent(),
        assistant_message=_assistant_message(2),
        finish_reason="stop",
        messages=[],
        conversation_history=[],
        api_call_count=1,
        effective_task_id="task",
        user_message="hi",
        system_message=None,
        active_system_prompt=None,
        compression_attempts=0,
        max_compression_attempts=3,
        final_response=None,
        failed=False,
        _turn_exit_reason="unknown",
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        tool_call_count=3,
    )
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "budget_exhausted"
    assert verdict.tool_call_count == 3


class _LimitAgent:
    def __init__(self, *, max_iterations=100, budget_remaining=50):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
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
        self.persisted_messages = None
        self._handle_max_iterations_called = False

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _emit_diagnostic_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def test_tool_capped_turn_gets_a_summary_handoff(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(max_iterations=100, budget_remaining=50)
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=50,
        tool_call_count=100,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )
    assert agent._handle_max_iterations_called is True
    assert result["turn_exit_reason"] == "max_iterations_reached(50/100)"
    assert result["completed"] is False
    assert is_max_iteration_handoff(result) is True


def test_turn_under_the_tool_cap_is_untouched(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(max_iterations=100, budget_remaining=50)
    result = finalize_turn(
        agent,
        final_response="done",
        api_call_count=50,
        tool_call_count=12,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )
    assert agent._handle_max_iterations_called is False
    assert result["completed"] is True
    assert result["final_response"] == "done"
