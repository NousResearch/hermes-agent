"""A fire fence that cannot be ACQUIRED is not a verdict about the fire claim.

``_fire_job_lock`` fails closed on a bounded timeout (``_JOBS_LOCK_TIMEOUT_SECONDS``, 30s in
production) and something routinely holds a job's fence for longer than that: the run's own
delivery, which is a client-side network call (measured 50-57s on the Matrix lane while the
homeserver answers in 2-6ms). Both refusal sites used to collapse that into an ownership verdict:

* ``fire_claim_fence`` → ``_FireClaimLostDuringSideEffect`` → ``_record_fire_ownership_lost``,
  which stamps ``Interrupted by shutdown before terminal completion.`` on a run whose claim the
  same function just re-confirmed as ours;
* ``_finish_completed_run`` → ``mark_job_run`` returning False → ``Fire claim ownership lost
  before terminal completion.`` — after the delivery had already left the process.

The fence timing out samples NOTHING from the store, so it carries no ownership evidence: the run
must still fail closed (no delivery, no terminal status write) but record the real cause.
"""

import contextlib
import threading
import time

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json/executions don't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_running_state():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


def _fence_holder(job_id, held: threading.Event, release: threading.Event):
    """A sibling owner whose own long side effect holds this job's fire fence."""
    import cron.jobs as jobs

    while not release.is_set():
        with jobs._fire_job_lock(job_id) as acquired:
            if acquired:
                held.set()
                release.wait(timeout=10)
                return
        time.sleep(0.005)


def _claimed_job(name: str):
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name=name)
    assert claim_job_for_fire(job["id"]) is True
    job = get_job(job["id"])
    assert isinstance(job.get("fire_claim"), dict) and job["fire_claim"].get("by")
    return job


def test_busy_fence_is_distinguishable_from_a_stale_owner(temp_home, monkeypatch):
    """False from a fence means one of two different things; only one of them is about the claim."""
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.1)
    job = _claimed_job("fence-refusal-reasons")
    owner = job["fire_claim"]["by"]

    held, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_fence_holder, args=(job["id"], held, release), daemon=True)
    holder.start()
    try:
        assert held.wait(timeout=5), "sibling owner never took the fence"

        busy: list = []
        with jobs.fire_claim_fence(job["id"], expected_owner=owner, fence_busy=busy) as owns_claim:
            assert owns_claim is False
        assert busy == [True], "a refused fence acquire must report itself busy"

        busy = []
        assert jobs._under_fire_fence(
            job["id"], lambda: True, fence_busy=busy) is False
        assert busy == [True]

        busy = []
        assert jobs.mark_job_run(
            job["id"], True, expected_fire_owner=owner, fence_busy=busy) is False
        assert busy == [True]
    finally:
        release.set()
        holder.join(timeout=5)
    assert holder.is_alive() is False

    # Fence free, claim owned by someone else: a real verdict, and NOT reported as busy.
    busy = []
    with jobs.fire_claim_fence(job["id"], expected_owner="stale", fence_busy=busy) as owns_claim:
        assert owns_claim is False
    assert busy == [], "a stale owner is a verdict, not fence contention"

    busy = []
    assert jobs.mark_job_run(
        job["id"], True, expected_fire_owner="stale", fence_busy=busy) is False
    assert busy == []


def test_side_effect_fence_busy_is_not_recorded_as_ownership_loss(temp_home, monkeypatch):
    """A fence the run could not acquire must not stamp it as lost ownership (it never sampled)."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.executions import get_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
    job = _claimed_job("side-effect-fence-busy")
    owner = job["fire_claim"]["by"]
    delivered = []

    held, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_fence_holder, args=(job["id"], held, release), daemon=True)
    holder.start()
    assert held.wait(timeout=5), "sibling owner never took the fence"

    monkeypatch.setattr(sched, "run_job", lambda job, **kw: (True, "output text", "report", None))
    monkeypatch.setattr(sched, "_deliver_result", lambda job, content, **kw: delivered.append(content))
    try:
        assert sched.run_one_job(job) is True
    finally:
        release.set()
        holder.join(timeout=5)

    assert delivered == [], "the side effect was never authorized, so it must not have happened"
    execution = get_execution(job["execution_id"])
    assert execution["status"] == "failed"
    assert execution["error"] == sched._FIRE_FENCE_BUSY
    assert "ownership lost" not in (execution["error"] or "")
    assert execution["error"] != sched._OWNERSHIP_LOST_INTERRUPTED
    # The claim was never taken from this run, so its record must not carry a failure.
    record = jobs.get_job(job["id"])
    assert record.get("last_status") is None
    assert record.get("failure_streak") in (None, 0)
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is True


def test_terminal_fence_busy_after_delivery_is_not_recorded_as_ownership_loss(temp_home, monkeypatch):
    """The delivered run's bookkeeping must not be overwritten by a fence timeout."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.executions import create_execution, get_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
    job = _claimed_job("terminal-fence-busy")
    owner = job["fire_claim"]["by"]
    execution_id = create_execution(job["id"], source="builtin")["id"]

    held, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_fence_holder, args=(job["id"], held, release), daemon=True)
    holder.start()
    assert held.wait(timeout=5), "sibling owner never took the fence"

    delivery = sched._RunDelivery(job=job, success=True, error=None, should_deliver=False)
    try:
        assert sched._finish_completed_run(delivery, owner, execution_id) is True
    finally:
        release.set()
        holder.join(timeout=5)

    execution = get_execution(execution_id)
    assert execution["status"] == "failed"
    assert execution["error"] == sched._FIRE_FENCE_BUSY
    assert execution["error"] != "Fire claim ownership lost before terminal completion."
    assert execution["error"] != sched._OWNERSHIP_LOST_INTERRUPTED
    record = jobs.get_job(job["id"])
    assert record.get("last_status") is None, "no terminal write may happen without the fence"


def _drive_heartbeat(monkeypatch, *, degraded: bool, grace: float):
    """One run whose renewals all miss; returns the seconds until the run was cancelled."""
    import cron.scheduler as sched

    job = {"id": "degraded-heartbeat", "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "o"}}
    cancelled_after = []

    def heartbeat(*_args, **_kwargs):
        # The pre-flight validation runs on the run's own thread; only the heartbeat thread misses.
        return threading.current_thread().name != "cron-fire-claim-heartbeat"

    def run_body(_job, **kwargs):
        start = time.monotonic()
        assert kwargs["claim_lost"].wait(timeout=5), "the run was never cancelled"
        cancelled_after.append(time.monotonic() - start)
        return True

    monkeypatch.setattr(sched, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(sched, "jobs_lock_degraded", lambda: degraded)
    monkeypatch.setattr(sched, "_run_one_job_body", run_body)
    monkeypatch.setattr(sched, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(sched, "_FIRE_CLAIM_MISS_CONFIRM_SECONDS", 0.02)
    monkeypatch.setattr(sched, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", grace)
    assert sched.run_one_job(job) is True
    return cancelled_after[0]


def test_miss_sampled_without_the_cross_process_lock_is_re_sampled_not_latched(monkeypatch):
    """#fence-class: a compare-and-refresh that ran UNLOCKED cross-process is a sample, not a
    verdict — keep renewing until the grace window is exhausted, then latch."""
    grace = 0.4
    cancelled_after = _drive_heartbeat(monkeypatch, degraded=True, grace=grace)
    assert cancelled_after >= grace, (
        "a miss that was sampled without the cross-process jobs lock cancelled the live run "
        "before the grace window was exhausted")


def test_miss_with_the_cross_process_lock_still_latches_promptly(monkeypatch):
    """The degraded-lock allowance must not slow down a real, serialized miss."""
    grace = 0.4
    cancelled_after = _drive_heartbeat(monkeypatch, degraded=False, grace=grace)
    assert cancelled_after < grace
