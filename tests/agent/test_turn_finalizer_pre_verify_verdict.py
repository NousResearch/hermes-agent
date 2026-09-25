"""CI-collected regression for the ``pre_verify`` enforced-verdict call site.

The enforcement itself is unit-tested in ``tests/agent/test_verify_hooks.py``
(``apply_pre_verify_verdict`` in isolation). What was NOT covered by any
collected suite is the *call site*: that :func:`agent.turn_finalizer.finalize_turn`
applies the recorded verdict **after** the ``transform_llm_output`` hooks.

That ordering is the whole contract. A transform that wholesale-rewrites the
answer runs first; if the verdict were applied before the transforms (or not at
all), the delivered ``final_response`` would carry none of it. These tests drive
the real ``finalize_turn`` with a real (monkeypatched-at-source) hook dispatcher
and assert on the DELIVERED ``result["final_response"]``, so deleting the
``apply_pre_verify_verdict`` call in ``finalize_turn`` fails them.

Positive control (``test_cleared_verdict_leaves_transform_output_untouched``):
when the last ``pre_verify`` evaluation *cleared* the verdict — the real shape
after a continuation actually resolved the work — the delivered answer is the
transform's output byte-for-byte, with no appended verdict. Enforcement must be
a consequence of a pending verdict, never of the call site existing.
"""

import pytest

from agent.turn_finalizer import finalize_turn
from agent.verify_hooks import PRE_VERIFY_VERDICT_ATTR, record_pre_verify_verdict

VERDICT = "STOP-CHECK: 3 cards unattended; work is NOT complete."
DRAFT = "All done, everything is green."
TRANSFORMED = "Rewritten by a competing plugin: looks good to me!"


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-verdict"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


@pytest.fixture
def hooks(monkeypatch):
    """Route ``invoke_hook`` at its source module so ``finalize_turn``'s own
    lazy import sees it. Returns the recorded hook-name order."""
    seen = []

    def _install(transform_result):
        def fake_invoke(hook_name, **kwargs):
            seen.append(hook_name)
            if hook_name == "transform_llm_output" and transform_result is not None:
                return [transform_result]
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_invoke)
        return seen

    return _install


def _finalize(agent, *, final_response, interrupted=False):
    messages = [
        {"role": "user", "content": "finish the board"},
        {"role": "assistant", "content": final_response or ""},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-v",
        turn_id="turn-v",
        user_message="finish the board",
        original_user_message="finish the board",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )


def test_verdict_survives_a_competing_wholesale_transform(hooks):
    """A transform that replaces the entire answer cannot drop the verdict."""
    seen = hooks(TRANSFORMED)
    agent = _StubAgent()
    record_pre_verify_verdict(agent, VERDICT)

    result = _finalize(agent, final_response=DRAFT)
    delivered = result["final_response"]

    # The transform really ran (call-site ordering is exercised, not stubbed).
    assert "transform_llm_output" in seen
    assert result["response_transformed"] is True
    assert result["pre_transform_response"] == DRAFT
    assert TRANSFORMED in delivered
    # ...and the verdict is still in the DELIVERED answer, first.
    assert VERDICT in delivered
    assert delivered.startswith(VERDICT)
    # Consumed exactly once: nothing pending for the next turn.
    assert getattr(agent, PRE_VERIFY_VERDICT_ATTR) == ""


def test_verdict_is_enforced_with_no_transform_registered(hooks):
    hooks(None)
    agent = _StubAgent()
    record_pre_verify_verdict(agent, VERDICT)

    delivered = _finalize(agent, final_response=DRAFT)["final_response"]
    assert delivered == VERDICT + "\n\n" + DRAFT


def test_cleared_verdict_leaves_transform_output_untouched(hooks):
    """POSITIVE CONTROL — the real 'the continuation actually fixed it' path.

    The last ``pre_verify`` evaluation recorded an empty verdict, so the turn
    ends with exactly what the transform produced: byte-for-byte, no prefix.
    """
    hooks(TRANSFORMED)
    agent = _StubAgent()
    record_pre_verify_verdict(agent, VERDICT)   # earlier evaluation
    record_pre_verify_verdict(agent, "")        # later one clears it

    delivered = _finalize(agent, final_response=DRAFT)["final_response"]
    assert delivered == TRANSFORMED
    assert VERDICT not in delivered


def test_no_hook_verdict_at_all_is_byte_identical(hooks):
    """Default-off: no recorded verdict ⇒ the call site changes nothing."""
    hooks(None)
    agent = _StubAgent()
    delivered = _finalize(agent, final_response=DRAFT)["final_response"]
    assert delivered == DRAFT


def test_interrupt_wins_over_pending_verdict(hooks):
    hooks(None)
    agent = _StubAgent()
    record_pre_verify_verdict(agent, VERDICT)

    delivered = _finalize(agent, final_response=DRAFT, interrupted=True)["final_response"]
    assert VERDICT not in delivered
    # Still consumed — it must not leak into the next turn.
    assert getattr(agent, PRE_VERIFY_VERDICT_ATTR) == ""
