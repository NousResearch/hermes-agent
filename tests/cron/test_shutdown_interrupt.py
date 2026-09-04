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

import threading
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Every test starts from a clean slate and leaves one behind, since
    these sets are module-level globals shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


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

    def test_marks_every_in_flight_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})
        profile_home = sched._get_hermes_home().resolve()
        sched._running_fire_owners.update(
            {
                "job-1": {object(): ("owner-1", profile_home)},
                "job-2": {object(): ("owner-2", profile_home)},
            }
        )

        with patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (final-cleanup)")

        assert sorted(marked) == ["job-1", "job-2"]
        assert mock_mark.call_count == 2
        called_ids = {c.args[0] for c in mock_mark.call_args_list}
        assert called_ids == {"job-1", "job-2"}
        for c in mock_mark.call_args_list:
            # success must be False -- an interrupted run is never "ok".
            assert c.args[1] is False
            assert "gateway shutdown" in c.args[2]
            assert c.kwargs["expected_fire_owner"] in {"owner-1", "owner-2"}

    def test_sets_interrupted_flag_for_consumption_by_run_one_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")

        with patch("cron.scheduler.mark_job_run"):
            sched.mark_running_jobs_interrupted("shutdown")

        assert "job-1" in sched._interrupted_job_ids

    def test_one_job_marking_failure_does_not_block_the_others(self):
        """mark_job_run raising for one job (e.g. a jobs.json write race)
        must not prevent the rest from being marked -- this runs during
        shutdown, there's no retry window."""
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})
        profile_home = sched._get_hermes_home().resolve()
        sched._running_fire_owners.update(
            {
                "job-1": {object(): ("owner-1", profile_home)},
                "job-2": {object(): ("owner-2", profile_home)},
            }
        )

        def _side_effect(job_id, success, reason, **kwargs):
            if job_id == "job-1":
                raise OSError("disk full")
            return True

        with patch("cron.scheduler.mark_job_run", side_effect=_side_effect):
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == ["job-2"]

    def test_stale_shutdown_cannot_clear_replacement_owner(self, tmp_path):
        import cron.jobs as jobs
        import cron.scheduler as sched

        profile_home = tmp_path / "profile"
        profile_home.mkdir()
        with jobs.use_cron_store(profile_home):
            created = jobs.create_job(prompt="x", schedule="every 5m", name="owned")
            claimed = jobs.claim_job_for_fire(created["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            stale_owner = claimed["fire_claim"]["by"]
            original_status = claimed["last_status"]
            replacement_claim = {
                "at": "2026-07-12T12:30:00+00:00",
                "by": "replacement-owner",
            }
            replacement = {**claimed, "fire_claim": replacement_claim}
            jobs.save_jobs([replacement])

            sched._running_job_ids.add(created["id"])
            sched._running_fire_owners[created["id"]] = {
                object(): (stale_owner, profile_home)
            }
            marked = sched.mark_running_jobs_interrupted("shutdown")
            refreshed = jobs.get_job(created["id"])

        assert marked == []
        assert isinstance(refreshed, dict)
        assert refreshed["fire_claim"] == replacement_claim
        assert refreshed["last_status"] == original_status


class TestRunningFireOwnerRegistry:
    def test_run_one_job_registers_owner_only_while_active(self):
        import cron.scheduler as sched

        job = {
            "id": "owned-job",
            "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner-1"},
        }

        def _observe_registry(current_job, run):
            assert list(sched._running_fire_owners[current_job["id"]].values()) == [
                ("owner-1", sched._get_hermes_home().resolve())
            ]
            return True

        with patch("cron.scheduler._run_with_fire_claim_heartbeat", side_effect=_observe_registry):
            assert sched.run_one_job(job) is True

        assert job["id"] not in sched._running_fire_owners

    def test_shutdown_sees_all_concurrent_direct_fire_owners(self, monkeypatch):
        """Direct entry points and replacement owners share one token registry."""
        import cron.scheduler as sched

        entered = threading.Barrier(3)
        release = threading.Event()
        marked_owners: list[str] = []

        def hold_run(_job, _run):
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return True

        def mark(_job_id, _success, _reason, *, expected_fire_owner):
            marked_owners.append(expected_fire_owner)
            return True

        monkeypatch.setattr(sched, "_run_with_fire_claim_heartbeat", hold_run)
        monkeypatch.setattr(sched, "mark_job_run", mark)

        jobs = [
            {"id": "same-job", "fire_claim": {"by": "old-owner"}},
            {"id": "same-job", "fire_claim": {"by": "replacement-owner"}},
        ]
        threads = [threading.Thread(target=sched.run_one_job, args=(job,)) for job in jobs]
        for thread in threads:
            thread.start()
        entered.wait(timeout=2)

        assert sched.get_running_job_ids() == frozenset({"same-job"})
        assert sched.mark_running_jobs_interrupted("shutdown") == ["same-job", "same-job"]
        assert set(marked_owners) == {"old-owner", "replacement-owner"}

        release.set()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
        assert "same-job" not in sched.get_running_job_ids()

    def test_shutdown_marks_each_owner_in_its_profile_store(self, monkeypatch, tmp_path):
        import cron.jobs as cron_jobs
        import cron.scheduler as sched

        profile_a = tmp_path / "a"
        profile_b = tmp_path / "b"
        observed = []
        sched._running_fire_owners["same-job"] = {
            object(): ("owner-a", profile_a),
            object(): ("owner-b", profile_b),
        }

        def mark(job_id, success, reason, *, expected_fire_owner):
            observed.append(
                (
                    job_id,
                    success,
                    expected_fire_owner,
                    cron_jobs._current_cron_store().jobs_file,
                )
            )
            return True

        monkeypatch.setattr(sched, "mark_job_run", mark)

        assert sched.mark_running_jobs_interrupted("shutdown") == ["same-job", "same-job"]
        assert set(observed) == {
            ("same-job", False, "owner-a", profile_a / "cron" / "jobs.json"),
            ("same-job", False, "owner-b", profile_b / "cron" / "jobs.json"),
        }


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


class TestExecutionScopedInterruption:
    """Interruption flags must target ONE execution, not the job ID.

    Owner-registered executions are recorded by their unique execution
    token, so a fresh run that reuses the same job ID (recurring fire,
    replacement claim owner) never consumes a flag that targeted its
    dead predecessor.
    """

    def test_interruption_targets_only_the_interrupted_execution(self):
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        old_token = object()
        sched._running_fire_owners["job-1"] = {
            old_token: ("owner-1", profile_home),
        }

        with patch("cron.scheduler.mark_job_run", return_value=True):
            sched.mark_running_jobs_interrupted("shutdown")

        assert sched._is_interrupted("job-1", old_token) is True
        new_token = object()
        assert sched._is_interrupted("job-1", new_token) is False
        # A new execution must not steal (and thereby clear) the old flag.
        assert sched._consume_interrupted_flag("job-1", new_token) is False
        assert sched._consume_interrupted_flag("job-1", old_token) is True
        assert sched._is_interrupted("job-1", old_token) is False

    def test_only_owners_marks_only_targeted_executions(self):
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        token_a, token_b = object(), object()
        sched._running_fire_owners["job-a"] = {token_a: ("owner-a", profile_home)}
        sched._running_fire_owners["job-b"] = {token_b: ("owner-b", profile_home)}

        with patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            marked = sched.mark_running_jobs_interrupted(
                "dashboard shutdown",
                only_owners={("job-a", "owner-a")},
            )

        assert marked == ["job-a"]
        assert mock_mark.call_count == 1
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-a"
        assert sched._is_interrupted("job-a", token_a) is True
        assert sched._is_interrupted("job-b", token_b) is False

    def test_replacement_execution_of_same_job_is_not_poisoned(self):
        """A replacement owner starting while the stale flag exists must
        complete through the normal mark path, not the interrupted one."""
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        stale_token = object()
        sched._running_fire_owners["job-1"] = {
            stale_token: ("stale-owner", profile_home),
        }
        with patch("cron.scheduler.mark_job_run", return_value=True):
            sched.mark_running_jobs_interrupted("shutdown")
        sched._running_fire_owners.clear()

        job = {
            "id": "job-1",
            "name": "test job",
            "prompt": "do work",
            "fire_claim": {"by": "replacement-owner"},
        }
        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.fire_claim_fence"), \
             patch("cron.scheduler.heartbeat_fire_claim", return_value=True), \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        mock_mark.assert_called_once()


class TestCombinedCancelEvent:
    def test_or_semantics(self):
        import cron.scheduler as sched

        a, b = threading.Event(), threading.Event()
        combined = sched._CombinedCancelEvent(a, b)
        assert combined.is_set() is False
        b.set()
        assert combined.is_set() is True

    def test_set_propagates_to_all(self):
        import cron.scheduler as sched

        a, b = threading.Event(), threading.Event()
        combined = sched._CombinedCancelEvent(a, b)
        combined.set()
        assert a.is_set() and b.is_set()

    def test_run_one_job_forwards_external_cancel_event(self):
        import cron.scheduler as sched

        external = threading.Event()
        job = {"id": "job-x", "name": "x", "prompt": "p"}

        with patch.object(
            sched,
            "_run_with_fire_claim_heartbeat",
            side_effect=lambda job_arg, run: run(threading.Event()),
        ), patch.object(sched, "_run_one_job_body", return_value=True) as body:
            assert sched.run_one_job(job, cancel_event=external) is True

        combined = body.call_args.kwargs["fire_claim_lost"]
        assert combined.is_set() is False
        external.set()
        assert combined.is_set() is True


class TestBaseExceptionThroughOwnerFencedFlow:
    """#73973 (sweeper review on #70638): a BaseException escaping run_job
    must still record a failed run through the owner-fenced terminal path —
    and a stale worker must not record over a replacement claim owner."""

    def _job(self):
        return {
            "id": "job-be",
            "name": "base exc",
            "prompt": "p",
            "fire_claim": {"by": "owner-be"},
        }

    def _patches(self, run_side_effect):
        return (
            patch("cron.scheduler.claim_dispatch", return_value=True),
            patch("agent.secret_scope.set_secret_scope", return_value=None),
            patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
            patch("agent.secret_scope.reset_secret_scope"),
            patch("cron.scheduler.run_job", side_effect=run_side_effect),
            patch("cron.scheduler.heartbeat_fire_claim", return_value=True),
        )

    def test_cancelled_error_records_failure_and_reraises(self):
        import asyncio

        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(asyncio.CancelledError())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark, \
             patch("cron.scheduler.finish_execution") as mock_finish:
            try:
                sched.run_one_job(self._job())
                raised = False
            except asyncio.CancelledError:
                raised = True

        assert raised, "non-Exception BaseException must propagate"
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[:3] == ("job-be", False, "CancelledError")
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"
        assert mock_finish.call_args.kwargs["success"] is False

    def test_keyboard_interrupt_records_failure_and_reraises(self):
        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(KeyboardInterrupt())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark, \
             patch("cron.scheduler.finish_execution"):
            try:
                sched.run_one_job(self._job())
                raised = False
            except KeyboardInterrupt:
                raised = True

        assert raised
        mock_mark.assert_called_once()
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"

    def test_base_exception_from_stale_owner_is_fenced_out(self):
        """A replacement owner reclaimed the job: the stale worker's
        BaseException path must NOT write terminal state over it."""
        import asyncio

        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(asyncio.CancelledError())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=False) as mock_mark, \
             patch("cron.scheduler.finish_execution"):
            try:
                sched.run_one_job(self._job())
            except asyncio.CancelledError:
                pass

        mock_mark.assert_called_once()
        # fenced write was attempted with the stale owner and discarded by
        # the store (return False) — and the code accepted that verdict
        # without retrying or writing anything else.
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"


class TestCallerLossAfterClaimAcquisition:
    """cirwel's integration assertion on #70638: if the HTTP/CLI caller is
    lost AFTER the claim was acquired, the gateway owner must produce at
    most one terminal ledger/artifact/delivery, clear only its own claim,
    and block retries while that ownership is live."""

    def test_second_fire_cannot_claim_while_first_ownership_live(self, tmp_path):
        import cron.jobs as jobs

        with jobs.use_cron_store(tmp_path):
            job = jobs.create_job(prompt="x", schedule="every 5m", name="owned")
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)

            # Caller died here — the claim outlives it. A retry (NAS/webhook
            # or manual) must be refused while the lease is fresh.
            retry = jobs.claim_job_for_fire(job["id"], return_job=True)
            assert retry is False or not isinstance(retry, dict)

            # The live owner still heartbeats and terminally marks — exactly
            # one terminal write, and only its own claim is cleared.
            owner = claimed["fire_claim"]["by"]
            assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is True
            assert jobs.mark_job_run(
                job["id"], True, expected_fire_owner=owner,
            ) is True
            refreshed = jobs.get_job(job["id"])
            assert refreshed["fire_claim"] is None
            assert refreshed["last_status"] == "ok"


class TestRunOneJobHonoursInterruptedFlag:
    """run_one_job() must not let a job's own completion overwrite a
    status the shutdown path already wrote for the same run."""

    def _make_job(self, job_id="job-1"):
        return {"id": job_id, "name": "test job", "prompt": "do work"}

    def test_success_path_skipped_when_interrupted(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        # The would-be "success" write must NOT happen -- the shutdown
        # path already wrote the authoritative interrupted status.
        mock_mark.assert_not_called()
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
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "a plausible final response", None),
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
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler.run_job", side_effect=RuntimeError("boom")), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is False
        mock_mark.assert_not_called()
