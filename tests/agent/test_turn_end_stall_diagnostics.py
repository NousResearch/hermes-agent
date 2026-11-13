"""Regression coverage for #131740: the turn-end activity stamp and the watchdog
stall stack dump.

A Desktop turn wedged between the ``Turn ended`` log line and ``run_conversation``
returning: the liveness watchdog fired ~600s later naming ``API call #8 completed``
as the last activity — the last stamp before the wedge — so nothing in the log
pinned the blocking frame, and the abort read like a hung API call rather than a
turn that had already finished its loop.

Two guarantees keep the next occurrence diagnosable:

1. ``finalize_turn`` stamps the activity clock at ``Turn ended``
   (``agent/turn_finalizer.py::_log_turn_exit``): a stall in the post-loop tail is
   measured — and surfaced — from when the loop actually ended, and the watchdog's
   ``last activity`` names ``turn end logged`` instead of a stale API call.
2. The watchdog dumps all thread stacks (turn thread marked) the moment it fires
   (``agent/turn_liveness.py``), so the wedged frame lands in errors.log without
   the reporter needing py-spy or faulthandler setup.
"""

from __future__ import annotations

import logging
import threading

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


from agent.status_output import StatusOutputMixin


class _StubAgent(StatusOutputMixin):
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "desktop"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.activity_log = []
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

    # --- fallible / observed surfaces -------------------------------------
    def _touch_activity(self, desc, **_kwargs):
        self.activity_log.append(desc)

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

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


def _finalize(agent):
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "done"},
    ]
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_turn_end_logs_stamp_the_activity_clock():
    """The ``Turn ended`` log line must also stamp the liveness activity clock.

    Red on base: without the stamp, a turn that wedges in the post-loop tail is
    measured from the last API call — the watchdog fires ~600s after the loop
    ended and reports ``API call #N completed`` as the last activity (#131740).
    """
    agent = _StubAgent()
    result = _finalize(agent)
    assert result["completed"] is True
    # The finalizer stamped the clock when the loop ended, so the watchdog's
    # next sample names the finalizer, not the last provider response.
    assert agent.activity_log, "finalize_turn never stamped the activity clock"
    assert agent.activity_log[-1] == "turn end logged"


def test_turn_end_stamp_failure_never_breaks_the_finalizer():
    """A broken activity-clock seam (stub doubles without the mixin) must not
    lose the final response or raise out of ``finalize_turn``."""
    agent = _StubAgent()

    def _explode(desc, **_kwargs):
        raise RuntimeError("activity clock unavailable")

    agent._touch_activity = _explode  # type: ignore[method-assign]
    result = _finalize(agent)
    assert result["final_response"] == "done"
    assert result["completed"] is True


def test_watchdog_dump_the_turn_thread_stack_when_it_fires(caplog):
    """The stall surface must include every thread's stack (turn thread marked),
    so the wedged frame is in errors.log on the next occurrence (#131740)."""
    from agent import turn_liveness

    class _Agent:
        session_id = "stalled-session"
        _last_activity_ts = None
        _turn_liveness_activity_generation = 0
        _last_activity_desc = "API call #8 completed"
        _execution_thread_id: int | None = None

        def _liveness_activity_lock(self):
            lock = getattr(self, "_lock", None)
            if lock is None:
                lock = self._lock = threading.Lock()
            return lock

    agent = _Agent()
    # Park a "turn thread" on a known frame so the dump can prove it was captured.
    parked = threading.Event()
    release = threading.Event()

    def _turn_thread():
        agent._execution_thread_id = threading.current_thread().ident
        parked.set()
        release.wait(10.0)

    worker = threading.Thread(target=_turn_thread, name="prompt-turn-stalled", daemon=True)
    worker.start()
    assert parked.wait(10.0), "turn thread never started"

    watchdog = turn_liveness.TurnLivenessWatchdog(
        agent, session_id="stalled-session", timeout_s=600, poll_s=15,
        stop_event=threading.Event(), activity_lock=agent._liveness_activity_lock(),
        is_turn_active=lambda: True, commit_abort=lambda snapshot, message: True,
        deactivate_turn=lambda: None,
    )
    try:
        with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
            watchdog._surface_stall(
                turn_liveness.ActivitySnapshot(generation=1, activity_ts=None, idle_seconds=610.0)
            )
    finally:
        release.set()
        worker.join(10.0)

    dump = "\n".join(record.getMessage() for record in caplog.records)
    assert "All thread stacks at turn stall" in dump
    assert "prompt-turn-stalled" in dump, "turn thread missing from the stall dump"
    assert "<- turn thread" in dump, "the turn thread was not marked in the stall dump"
    # The stack contains the parked frame the wedged thread was actually in.
    assert "release.wait" in dump or "_turn_thread" in dump
    # Rate-limited per activity generation: a second surface of the SAME
    # generation must not dump again.
    records_before = len(caplog.records)
    watchdog._surface_stall(
        turn_liveness.ActivitySnapshot(generation=1, activity_ts=None, idle_seconds=620.0)
    )
    assert len(caplog.records) == records_before
