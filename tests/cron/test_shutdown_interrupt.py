"""Tests for #60432: cron jobs must not be silently invisible to gateway
shutdown, and a job whose tool subprocess got killed by shutdown must
never be reported as a successful run.

Covers the cron/scheduler.py primitives directly:
  - get_running_job_ids() -- thread-safe snapshot the gateway drain reads
  - mark_running_jobs_interrupted() -- called by the gateway right after
    it force-kills tool subprocesses
  - the interrupted-flag race guard in run_one_job(), which must win over
    the job's own thread finishing normally with a plausible-looking
    result AFTER its tool was already killed out from under it
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state(monkeypatch):
    """Every test starts from a clean slate and leaves one behind, since
    these sets are module-level globals shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._interrupted_job_reasons.clear()
    sched._active_cron_worker_processes.clear()
    monkeypatch.setattr(sched, "claim_job_for_fire_token", lambda jid: f"claim-{jid}")
    monkeypatch.setattr(sched, "heartbeat_fire_claim", lambda *a, **k: True)
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._interrupted_job_reasons.clear()
    sched._active_cron_worker_processes.clear()


class TestGetRunningJobIds:
    def test_empty_when_nothing_running(self):
        import cron.scheduler as sched

        assert sched.get_running_job_ids() == frozenset()

    def test_reflects_in_flight_jobs(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        sched._running_job_ids.add("job-2")

        result = sched.get_running_job_ids()

        assert result == frozenset({"job-1", "job-2"})

    def test_snapshot_is_immutable_and_independent(self):
        """Mutating _running_job_ids after the call must not change the
        already-returned snapshot -- callers (the gateway drain loop) rely
        on this to safely count in a tight polling loop."""
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        snapshot = sched.get_running_job_ids()
        sched._running_job_ids.add("job-2")

        assert snapshot == frozenset({"job-1"})


class TestMarkRunningJobsInterrupted:
    def test_no_op_when_nothing_running(self):
        import cron.scheduler as sched

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == []
        mock_mark.assert_not_called()

    def test_terminates_direct_worker_but_defers_owner_finalization(self):
        """Shutdown reaps a direct worker not present in the ticker thread set."""
        import cron.scheduler as sched

        worker = object()
        sched._active_cron_worker_processes["direct-job"] = worker
        with patch("cron.scheduler._terminate_cron_worker") as terminate, \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown")

        assert marked == ["direct-job"]
        terminate.assert_called_once_with(worker)
        mock_mark.assert_not_called()
        assert sched._interrupted_job_reasons["direct-job"] == "gateway shutdown"

    def test_defers_finalization_until_worker_acknowledges_interruption(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (final-cleanup)")

        assert sorted(marked) == ["job-1", "job-2"]
        mock_mark.assert_not_called()
        assert sched._interrupted_job_reasons == {
            "job-1": "gateway shutdown (final-cleanup)",
            "job-2": "gateway shutdown (final-cleanup)",
        }

    def test_sets_interrupted_flag_for_consumption_by_run_one_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")

        with patch("cron.scheduler.mark_job_run"):
            sched.mark_running_jobs_interrupted("shutdown")

        assert "job-1" in sched._interrupted_job_ids

    def test_shutdown_signal_does_not_write_runtime_from_non_owner(self):
        """The shutdown thread may flag workers, but only owners may finalize."""
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert sorted(marked) == ["job-1", "job-2"]
        mock_mark.assert_not_called()


class TestIsInterrupted:
    """Peek-only check used at the delivery gate -- must NOT clear the
    flag, unlike _consume_interrupted_flag."""

    def test_false_when_not_marked(self):
        import cron.scheduler as sched

        assert sched._is_interrupted("job-1") is False

    def test_true_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._is_interrupted("job-1") is True

    def test_does_not_clear_the_flag(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        sched._is_interrupted("job-1")

        # Still set -- the later, authoritative check before mark_job_run
        # must still see it.
        assert "job-1" in sched._interrupted_job_ids
        assert sched._is_interrupted("job-1") is True


class TestConsumeInterruptedFlag:

    def test_true_and_clears_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._consume_interrupted_flag("job-1") is True
        # Consumed -- a second check (e.g. a later, unrelated fire of the
        # same recurring job ID) must not still read as interrupted.
        assert sched._consume_interrupted_flag("job-1") is False


class TestRunOneJobHonoursInterruptedFlag:
    """run_one_job() must not let a job's own completion overwrite a
    status the shutdown path already wrote for the same run."""

    def _make_job(self, job_id="job-1"):
        return {"id": job_id, "name": "test job", "prompt": "do work"}

    def test_queued_interrupted_job_never_starts_and_releases_claims(self):
        """A pool-queued job marked during shutdown must not begin afterward."""
        import cron.scheduler as sched

        job = {
            **self._make_job("queued-job"),
            "execution_id": "execution-1",
            "_fire_claim_id": "fire-1",
            "run_claim": {"id": "run-1", "by": "owner", "at": "now"},
        }
        sched._interrupted_job_ids.add(job["id"])
        sched._interrupted_job_reasons[job["id"]] = "gateway shutdown"

        with patch("cron.scheduler.claim_dispatch") as claim_dispatch, \
             patch("cron.scheduler._run_job_in_killable_process") as run_worker, \
             patch("cron.scheduler.release_run_claim", return_value=True) as release_run, \
             patch("cron.scheduler.release_fire_claim", return_value=True) as release_fire, \
             patch("cron.scheduler.finish_execution") as finish:
            result = sched.run_one_job(job)

        assert result is False
        claim_dispatch.assert_not_called()
        run_worker.assert_not_called()
        release_run.assert_called_once_with(
            "queued-job",
            expected_claim_id="run-1",
        )
        release_fire.assert_called_once_with(
            "queued-job",
            expected_claim_id="fire-1",
        )
        finish.assert_called_once_with(
            "execution-1",
            success=False,
            error="gateway shutdown",
            delivery_outcome="suppressed",
        )

    def test_success_path_is_fenced_failure_when_interrupted(self):
        import cron.scheduler as sched

        job = self._make_job()

        def complete_after_interrupt(*_args, **_kwargs):
            sched._interrupted_job_ids.add(job["id"])
            return True, "full output", "final response", None

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler._run_job_in_killable_process",
                 side_effect=complete_after_interrupt,
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[1] is False
        assert "interrupt" in mock_mark.call_args.args[2].lower()
        assert (
            mock_mark.call_args.kwargs["expected_fire_claim_id"]
            == "claim-job-1"
        )
        # Flag is consumed so a later, unrelated fire of the same job ID
        # isn't permanently silenced.
        assert job["id"] not in sched._interrupted_job_ids

    def test_interrupted_job_delivers_failure_summary_not_raw_response(self):
        """The status-write guard alone isn't enough: delivery happens
        BEFORE mark_job_run in run_one_job's own flow, so a job that kept
        running post-kill and produced a plausible-looking final_response
        must not have that response sent to the user just because the
        eventual status write gets suppressed. Interrupted jobs must route
        through the same failure-summary delivery path a real failure
        would."""
        import cron.scheduler as sched

        job = self._make_job()

        def complete_after_interrupt(*_args, **_kwargs):
            sched._interrupted_job_ids.add(job["id"])
            return True, "full output", "a plausible final response", None

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler._run_job_in_killable_process",
                 side_effect=complete_after_interrupt,
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch(
                 "cron.scheduler._summarize_cron_failure_for_delivery",
                 return_value="This run was interrupted.",
             ) as mock_summarize, \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None) as mock_deliver, \
             patch("cron.scheduler.mark_job_run"):
            result = sched.run_one_job(job)

        assert result is True
        mock_summarize.assert_called_once()
        # The summarizer's error argument must mention the interruption,
        # not be silently None / the agent's own (possibly absent) error.
        assert "interrupt" in mock_summarize.call_args.args[1].lower()
        delivered_content = mock_deliver.call_args.args[1]
        assert delivered_content == "This run was interrupted."
        assert "plausible final response" not in delivered_content


    def test_exception_path_also_honours_interrupted_flag(self):
        import cron.scheduler as sched

        job = self._make_job()

        def fail_after_interrupt(*_args, **_kwargs):
            sched._interrupted_job_ids.add(job["id"])
            raise RuntimeError("boom")

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler._run_job_in_killable_process",
                 side_effect=fail_after_interrupt,
             ), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is False
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[1] is False
        assert "interrupt" in mock_mark.call_args.args[2].lower()
        assert (
            mock_mark.call_args.kwargs["expected_fire_claim_id"]
            == "claim-job-1"
        )
