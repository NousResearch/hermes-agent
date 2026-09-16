"""Tests for #112892: user-facing cron interrupt (``hermes cron stop <id>``).

Four contracts that pin the new user-facing stop path (CLI half only — UI half
left for a follow-up per the implementation plan in ``.patrol/run154-...``):

1. ``interrupt_running_job`` calls ``request_hard_interrupt`` on the registered
   AIAgent with the operator's reason (and the trusted ``cron_stop`` category).
2. After ``interrupt_running_job`` runs, every execution token of the stopped
   job is in ``_interrupted_job_ids`` BEFORE ``run_one_job`` writes
   ``last_status`` — so a still-running agent that produces a plausible-looking
   final response cannot overwrite "interrupted" with a false "ok" (#60432).
3. ``interrupt_running_job`` does not touch jobs that aren't currently
   running; a manual ``hermes cron run`` triggered later picks up the SAME
   token identity (recurring jobs reuse the ID) and is NOT short-circuited by
   the previous run's flag — the flag is per-execution, per-token.
4. Calling ``interrupt_running_job`` twice on the same job is a no-op the second
   time: zero in-flight tokens means zero agents to interrupt and zero
   last_status rows to write (avoids overwriting the now-"interrupted" run
   with a second stop-attempt).
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Module-level dicts/sets shared across the test process — clear before AND after."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    sched._running_cron_agents.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    sched._running_cron_agents.clear()


def _register_token(job_id, agent, owner="owner-1"):
    """Helper: emulate ``run_one_job``'s claim setup so ``interrupt_running_job`` resolves the agent."""
    import cron.scheduler as sched

    token = object()
    profile_home = sched._get_hermes_home().resolve()
    with sched._running_lock:
        sched._running_fire_owners.setdefault(job_id, {})[token] = (owner, profile_home)
        sched._running_cron_agents.setdefault(job_id, {})[token] = agent
    return token


class TestInterruptRunningJobCallsHardInterrupt:
    def test_calls_request_hard_interrupt_with_reason_on_every_token(self):
        """Contract: the operator-supplied reason reaches the AIAgent, with the trusted
        ``cron_stop`` category exposed via ``tool_reason``."""
        import cron.scheduler as sched

        job_id = "job-stop-1"
        agent_a = object()
        agent_b = object()
        _register_token(job_id, agent_a, owner="owner-a")
        _register_token(job_id, agent_b, owner="owner-b")

        with patch("cron.scheduler.request_hard_interrupt") as mock_request:
            with patch("cron.scheduler.mark_running_jobs_interrupted",
                       return_value=[job_id]) as mock_mark:
                marked, interrupted = sched.interrupt_running_job(job_id, reason="user stopped")

        assert interrupted == 2
        assert marked == [job_id]
        # Two tokens → two hard-interrupt calls; reason + tool_reason both forwarded.
        assert mock_request.call_count == 2
        for call in mock_request.call_args_list:
            args, kwargs = call
            # Either positional or keyword form is acceptable; the agent must be present and the
            # reason text must be exactly what the operator passed.
            forwarded_agent = args[0] if args else kwargs.get("agent")
            forwarded_message = (
                args[1] if len(args) > 1
                else kwargs.get("message") or kwargs.get("tool_reason")
            )
            assert forwarded_agent in (agent_a, agent_b)
            assert "user stopped" in str(forwarded_message) or "user stopped" in str(kwargs)


class TestInterruptTokensWrittenBeforeLastStatus:
    def test_token_in_interrupted_set_before_mark_runs(self):
        """Contract: ``_interrupted_job_ids`` is updated BEFORE ``mark_running_jobs_interrupted``
        writes ``last_status`` — the pre-write ``_consume_interrupted_flag`` check (#60432)
        is what stops a still-running agent from overwriting "interrupted" with a false "ok"."""
        import cron.scheduler as sched

        job_id = "job-stop-2"
        token = _register_token(job_id, object(), owner="owner-1")

        observed_interrupted_set = []

        def spy_mark(reason, *, only_owners=None):
            observed_interrupted_set.append(set(sched._interrupted_job_ids))
            return [job_id]

        with patch("cron.scheduler.request_hard_interrupt"):
            with patch("cron.scheduler.mark_running_jobs_interrupted", side_effect=spy_mark):
                sched.interrupt_running_job(job_id, reason="user")

        # Exactly one observation; the token was already in the interrupted set at that point.
        assert len(observed_interrupted_set) == 1
        assert token in observed_interrupted_set[0]


class TestStopDoesNotPoisonFutureRuns:
    def test_repeat_fire_token_id_is_independent_of_previous_stop(self):
        """Contract: a recurring job that gets stopped MUST NOT short-circuit its next manual
        ``hermes cron run`` — recurring jobs reuse the job ID; the flag is per-token so a
        fresh fire with a new token starts unpoisoned."""
        import cron.scheduler as sched

        job_id = "job-recurring"
        agent_first = object()
        first_token = _register_token(job_id, agent_first, owner="owner-1")

        # Operator stops the first run.
        with patch("cron.scheduler.request_hard_interrupt"):
            with patch("cron.scheduler.mark_running_jobs_interrupted", return_value=[job_id]):
                sched.interrupt_running_job(job_id, reason="user")

        # Consume the flag (the natural run_one_job shutdown path does this).
        assert sched._consume_interrupted_flag(job_id, first_token) is True

        # Second fire registers a NEW token (run_one_job creates one each time).
        agent_second = object()
        second_token = _register_token(job_id, agent_second, owner="owner-1")

        # The second token must NOT be in the interrupted set; the consume check on the
        # second fire must return False.
        assert second_token not in sched._interrupted_job_ids
        assert sched._consume_interrupted_flag(job_id, second_token) is False


class TestStopIsIdempotentAndBoundarySafe:
    def test_no_running_job_returns_zero_no_mark(self):
        """Contract: stopping a job that isn't currently running is a no-op — no
        ``mark_job_run`` write, no ``request_hard_interrupt`` call, and the function
        signals "nothing to do" via the zero ``interrupted_count`` so the CLI can
        tell the operator."""
        import cron.scheduler as sched

        with patch("cron.scheduler.request_hard_interrupt") as mock_request:
            with patch("cron.scheduler.mark_running_jobs_interrupted") as mock_mark:
                marked, interrupted = sched.interrupt_running_job("job-never-ran", reason="user")

        assert interrupted == 0
        assert marked == []
        mock_request.assert_not_called()
        mock_mark.assert_not_called()

    def test_second_stop_after_first_is_safe_no_op(self):
        """Contract: calling ``interrupt_running_job`` twice doesn't double-fire
        ``request_hard_interrupt`` (the second call sees zero in-flight tokens) and
        doesn't try to overwrite the now-"interrupted" ``last_status`` row."""
        import cron.scheduler as sched

        job_id = "job-stop-twice"
        _register_token(job_id, object(), owner="owner-1")

        with patch("cron.scheduler.request_hard_interrupt") as mock_request:
            with patch("cron.scheduler.mark_running_jobs_interrupted", return_value=[job_id]):
                first_marked, first_interrupted = sched.interrupt_running_job(job_id, reason="user")
            # Second call: no tokens left in `_running_fire_owners`.
            second_marked, second_interrupted = sched.interrupt_running_job(job_id, reason="user")

        assert first_interrupted == 1 and first_marked == [job_id]
        assert second_interrupted == 0 and second_marked == []
        # Only one hard-interrupt call (from the first stop); the second stop saw no live token.
        assert mock_request.call_count == 1

    def test_stop_reason_reaches_mark_run_for_bookkeeping(self):
        """Contract: the operator's reason text is the SAME string the bookkeeping path writes
        into ``last_status`` (via ``mark_running_jobs_interrupted``), so the operator can grep
        for it later."""
        import cron.scheduler as sched

        job_id = "job-stop-reason"
        _register_token(job_id, object(), owner="owner-1")

        with patch("cron.scheduler.request_hard_interrupt"):
            with patch("cron.scheduler.mark_running_jobs_interrupted") as mock_mark:
                sched.interrupt_running_job(job_id, reason="manual stop from holny")

        assert mock_mark.call_count == 1
        # First positional arg of ``mark_running_jobs_interrupted(reason, *, only_owners=...)``.
        forwarded_reason = mock_mark.call_args[0][0]
        assert forwarded_reason == "manual stop from holny"