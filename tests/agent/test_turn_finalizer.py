from types import SimpleNamespace

from agent.turn_finalizer import finalize_turn


class DummyAgent:
    def __init__(self, *, max_iterations=4):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(remaining=1, used=max_iterations, max_total=max_iterations)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = "https://example.invalid/v1"
        self.session_id = "s1"
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
        self.session_cost_source = "test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._interrupt_message = None
        self.valid_tool_names = []
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _handle_max_iterations(self, _messages, _api_call_count):
        return "summary after budget exhaustion"

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *_args, **_kwargs):
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

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(agent, *, final_response, api_call_count, turn_exit_reason):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": final_response or ""}],
        conversation_history=None,
        effective_task_id="default",
        turn_id="t1",
        user_message="u",
        original_user_message="u",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )


def test_text_response_at_iteration_limit_is_completed():
    """A real final answer on the last allowed cron/API iteration is success.

    Cron jobs often run with max_iterations=4 and can legitimately use three
    tool turns plus one final text turn. That should not be surfaced as a
    RuntimeError just because api_call_count == max_iterations.
    """
    agent = DummyAgent(max_iterations=4)

    result = _finalize(
        agent,
        final_response="配信時刻: 2026-06-20 14:00 JST\n遅延なし",
        api_call_count=4,
        turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["completed"] is True
    assert result["failed"] is False


def test_budget_exhaustion_at_iteration_limit_remains_incomplete():
    """No final answer before the loop limit still remains incomplete."""
    agent = DummyAgent(max_iterations=4)

    result = _finalize(
        agent,
        final_response=None,
        api_call_count=4,
        turn_exit_reason="unknown",
    )

    assert result["completed"] is False
    assert result["final_response"] == "summary after budget exhaustion"
