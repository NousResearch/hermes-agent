"""Tests for the executions-ledger half of conscious interruption (#60432).

``mark_running_jobs_interrupted`` already writes the precise interruption
cause into jobs.json (``mark_job_run``) at the moment the gateway kills a
job's tool subprocess. These tests pin the matching behavior for the
executions ledger: the same call must ALSO write a terminal ``failed``
record for that job's open own-process attempts, so the row is never left
to the next scheduler's ``recover_interrupted_executions`` sweep, which
can only mislabel it ``unknown`` ("whether side effects ran is unknown")
long after the fact — even though THIS process knew the exact cause and
merely exited before its agent thread unwound.

Two records of one run (jobs.json last_status + executions ledger) must
not disagree about what happened.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Clean slate for the module-level sets shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions


def _seed_running_fire(sched, job_id, owner="owner-1", profile_home=None):
    """Register one in-flight fire the way dispatch does, return its token."""
    if profile_home is None:
        profile_home = sched._get_hermes_home().resolve()
    token = object()
    with sched._running_lock:
        sched._running_job_ids.add(job_id)
        sched._running_fire_owners.setdefault(job_id, {})[token] = (
            owner,
            profile_home,
        )
    return token


class TestFailOpenExecutionsForJob:
    def test_closes_open_own_process_attempts_as_failed(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        rec = executions.create_execution("job-1", source="builtin")
        executions.mark_execution_running(rec["id"])

        closed = executions.fail_open_executions_for_job(
            "job-1", reason="gateway shutdown killed tool subprocess"
        )

        assert len(closed) == 1
        row = executions.list_executions(job_id="job-1")[0]
        assert row["status"] == "failed"
        assert "gateway shutdown" in row["error"]
        assert row["finished_at"] is not None

    def test_terminal_attempts_are_immutable(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        rec = executions.create_execution("job-1", source="builtin")
        executions.finish_execution(rec["id"], success=True)

        closed = executions.fail_open_executions_for_job(
            "job-1", reason="shutdown"
        )

        assert closed == []
        row = executions.list_executions(job_id="job-1")[0]
        assert row["status"] == "completed"

    def test_other_process_attempts_untouched(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        with patch.object(executions, "_PROCESS_ID", "some-other-process"):
            rec = executions.create_execution("job-1", source="builtin")
        # sanity: the row exists and is open
        assert executions.list_executions(job_id="job-1")[0]["status"] == "claimed"

        closed = executions.fail_open_executions_for_job(
            "job-1", reason="shutdown"
        )

        assert closed == []
        assert executions.list_executions(job_id="job-1")[0]["status"] == "claimed"

    def test_other_jobs_untouched(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        rec_a = executions.create_execution("job-a", source="builtin")
        rec_b = executions.create_execution("job-b", source="builtin")
        executions.mark_execution_running(rec_a["id"])
        executions.mark_execution_running(rec_b["id"])

        closed = executions.fail_open_executions_for_job("job-a", reason="x")

        assert [r["id"] for r in closed] == [rec_a["id"]]
        statuses = {
            r["job_id"]: r["status"] for r in executions.list_executions(limit=10)
        }
        assert statuses == {"job-a": "failed", "job-b": "running"}

    def test_racing_finish_execution_converges(self, monkeypatch, tmp_path):
        """Whichever terminal write lands first wins; the loser no-ops."""
        executions = _point_ledger(monkeypatch, tmp_path)
        rec = executions.create_execution("job-1", source="builtin")
        executions.mark_execution_running(rec["id"])

        first = executions.finish_execution(
            rec["id"], success=False, error="thread unwound: tool killed"
        )
        second = executions.fail_open_executions_for_job(
            "job-1", reason="shutdown sweep"
        )

        assert first is not None
        assert second == []
        row = executions.list_executions(job_id="job-1")[0]
        assert row["status"] == "failed"
        assert "thread unwound" in row["error"]


class TestMarkRunningJobsInterruptedClosesLedger:
    """Scheduler-level: interrupting a real claimed fire closes its ledger."""

    def _seed_real_fire(self, tmp_path, name):
        """Create a real job + fire claim in a tmp profile store, register
        the in-flight fire the way dispatch does. Returns (job_id, owner,
        profile_home)."""
        import cron.jobs as jobs
        import cron.scheduler as sched

        profile_home = tmp_path / f"profile-{name}"
        profile_home.mkdir()
        with jobs.use_cron_store(profile_home):
            created = jobs.create_job(
                prompt="ledger test", schedule="every 5m", name=name
            )
            claimed = jobs.claim_job_for_fire(
                created["id"], force=True, return_job=True
            )
            owner = claimed["fire_claim"]["by"]
        token = object()
        with sched._running_lock:
            sched._running_job_ids.add(created["id"])
            sched._running_fire_owners.setdefault(created["id"], {})[token] = (
                owner,
                profile_home,
            )
        return created["id"], owner, profile_home

    @contextmanager
    def _home(self, profile_home):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        tok = set_hermes_home_override(str(profile_home))
        try:
            yield
        finally:
            reset_hermes_home_override(tok)

    def test_interrupt_closes_ledger_for_inflight_job(self, tmp_path):
        import cron.executions as executions
        import cron.scheduler as sched

        job_id, _owner, home = self._seed_real_fire(tmp_path, "alpha")
        with self._home(home):
            rec = executions.create_execution(job_id, source="builtin")
            executions.mark_execution_running(rec["id"])

            marked = sched.mark_running_jobs_interrupted(
                "gateway shutdown (drain)"
            )

            assert marked == [job_id]
            row = executions.list_executions(job_id=job_id)[0]
            assert row["status"] == "failed"
            assert "gateway shutdown" in row["error"]
            assert row["finished_at"] is not None

    def test_interrupt_without_owner_leaves_ledger_open(self, tmp_path):
        """Legacy dispatch (no registered fire owner) skips the persisted
        writes by design — the ledger stays open for the recovery sweep,
        which remains the only authority for unattributed attempts."""
        import cron.executions as executions
        import cron.scheduler as sched

        home = tmp_path / "profile-legacy"
        home.mkdir()
        with self._home(home):
            rec = executions.create_execution("legacy-job", source="builtin")
        sched._running_job_ids.add("legacy-job")

        marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == ["legacy-job"]
        with self._home(home):
            row = executions.list_executions(job_id="legacy-job")[0]
            assert row["status"] == "claimed"

    def test_interrupt_routes_ledger_write_to_fire_profile(
        self, tmp_path
    ):
        """The ledger file is profile-local (resolved from HERMES_HOME at
        transaction time). The interruption write must land in the fire's
        OWN profile home, not the calling process's."""
        import cron.executions as executions
        import cron.scheduler as sched

        job_id, _owner, fire_home = self._seed_real_fire(tmp_path, "beta")
        other_home = tmp_path / "profile-caller"
        other_home.mkdir()
        # The open attempt lives in the CALLER's ledger, not fire_home.
        with self._home(other_home):
            rec = executions.create_execution(job_id, source="builtin")
            assert executions.list_executions(job_id=job_id)[0]["status"] == "claimed"

            marked = sched.mark_running_jobs_interrupted("shutdown")

            assert marked == [job_id]
            # Caller's ledger row untouched (open) — the write was routed
            # to fire_home, where no attempt exists for this job.
            assert executions.list_executions(job_id=job_id)[0]["status"] == "claimed"
        # fire_home ledger was created (schema init) by the routed write.
        assert (fire_home / "cron" / "executions.db").exists()

    def test_ledger_failure_does_not_block_marking(self, tmp_path):
        import cron.executions as executions
        import cron.scheduler as sched

        job_a, _oa, home_a = self._seed_real_fire(tmp_path, "gamma")
        job_b, _ob, home_b = self._seed_real_fire(tmp_path, "delta")
        with self._home(home_a):
            rec_a = executions.create_execution(job_a, source="builtin")
            executions.mark_execution_running(rec_a["id"])
        with self._home(home_b):
            rec_b = executions.create_execution(job_b, source="builtin")
            executions.mark_execution_running(rec_b["id"])

        real_fail_open = sched.fail_open_executions_for_job
        calls = []

        def flaky_fail_open(job_id, *, reason):
            calls.append(job_id)
            if job_id == job_a:
                raise RuntimeError("sqlite locked")
            return real_fail_open(job_id, reason=reason)

        with patch.object(
            sched, "fail_open_executions_for_job", side_effect=flaky_fail_open
        ):
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert sorted(marked) == [job_a, job_b]  # jobs.json writes still ran
        assert sorted(calls) == sorted([job_a, job_b])
        with self._home(home_a):
            assert executions.list_executions(job_id=job_a)[0]["status"] == "running"
        with self._home(home_b):
            assert executions.list_executions(job_id=job_b)[0]["status"] == "failed"

    def test_only_owners_scoping_preserved(self, tmp_path):
        """only_owners (dashboard webhook drain) must keep scoping BOTH
        writes — only the targeted fire's job is marked and closed."""
        import cron.executions as executions
        import cron.scheduler as sched

        job_a, owner_a, home_a = self._seed_real_fire(tmp_path, "epsilon")
        job_b, _owner_b, home_b = self._seed_real_fire(tmp_path, "zeta")
        with self._home(home_a):
            executions.create_execution(job_a, source="builtin")
        with self._home(home_b):
            executions.create_execution(job_b, source="builtin")

        marked = sched.mark_running_jobs_interrupted(
            "drain", only_owners={(job_a, owner_a)}
        )

        assert marked == [job_a]
        with self._home(home_a):
            assert executions.list_executions(job_id=job_a)[0]["status"] == "failed"
        with self._home(home_b):
            assert executions.list_executions(job_id=job_b)[0]["status"] == "claimed"
