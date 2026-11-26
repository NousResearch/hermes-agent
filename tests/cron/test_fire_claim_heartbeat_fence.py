"""Fire-claim heartbeat against a busy fire fence.

The per-job fire fence (``cron.jobs._fire_job_lock``) is shared by the heartbeat, the run's own
delivery (``fire_claim_fence`` is held across the network send), ``mark_job_run`` and
``claim_job_for_fire``. A heartbeat that cannot take the fence has learned nothing about
ownership; only a heartbeat that takes it and finds another owner has.

Incident 2026-11-26: a delivery holding the fence past the lock timeout made the heartbeat log
"fire claim ownership lost; interrupting stale run", and a run that had completed AND delivered
was recorded as "Interrupted by shutdown before terminal completion."

These drive the real heartbeat loop (``run_one_job``), the real fence and the real store under the
isolated ``HERMES_HOME``. Timing knobs are shrunk so a contention episode costs milliseconds.
"""
import contextlib
import threading
from unittest.mock import MagicMock, patch

_HEARTBEAT_THREAD_NAME = "cron-fire-claim-heartbeat"


def _stored_job(job_id):
    import cron.jobs as jobs

    stored = jobs.get_job(job_id)
    assert stored is not None
    return stored


def _claimed_recurring_job(name="patrol"):
    """A recurring job whose current fire is claimed by this process (as a dispatched run sees it)."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name=name)
    assert jobs.claim_job_for_fire(job["id"]) is True
    return _stored_job(job["id"])


def _shrink_timing(monkeypatch):
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.05)


class _FenceProbe:
    """Observes the heartbeat thread's attempts to take the fire fence. Each finished attempt
    appends True (fence taken, claim inspected) or False (timed out: nothing learned)."""

    def __init__(self, monkeypatch, *, gate=None):
        import cron.jobs as jobs

        self._real_fence = jobs._fire_job_lock
        self._gate = gate
        self.outcomes = []
        self._changed = threading.Condition()
        monkeypatch.setattr(jobs, "_fire_job_lock", self._fence)

    @contextlib.contextmanager
    def _fence(self, job_id):
        heartbeat = threading.current_thread().name == _HEARTBEAT_THREAD_NAME
        if heartbeat and self._gate is not None:
            assert self._gate.wait(5), "heartbeat reached the fence before the test released it"
        with self._real_fence(job_id) as acquired:
            yield acquired
        if heartbeat:
            with self._changed:
                self.outcomes.append(acquired)
                self._changed.notify_all()

    def wait_until(self, predicate, timeout=5.0):
        with self._changed:
            return self._changed.wait_for(lambda: predicate(self.outcomes), timeout)


@contextlib.contextmanager
def _fence_held_by_another_thread(job_id):
    """Another party holds the job's fire fence (as a slow delivery does) until the block exits."""
    import cron.jobs as jobs

    held = threading.Event()
    release = threading.Event()

    def _hold():
        with jobs._fire_job_lock(job_id):
            held.set()
            release.wait(30)

    holder = threading.Thread(target=_hold, name="fence-holder", daemon=True)
    holder.start()
    try:
        assert held.wait(5), "holder never took the fence"
        yield
    finally:
        release.set()
        holder.join(timeout=5)


def test_heartbeat_rides_out_a_fence_held_past_the_lock_timeout(monkeypatch):
    """A heartbeat that times out on a busy fence has verified nothing: inside the grace window it
    must keep trying and must not declare the run lost."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    claimed = _claimed_recurring_job()
    owner = claimed["fire_claim"]["by"]
    probe = _FenceProbe(monkeypatch)
    _shrink_timing(monkeypatch)
    seen = {}

    def run_body(job, *, fire_claim_lost, **_kwargs):
        with _fence_held_by_another_thread(job["id"]):
            assert probe.wait_until(lambda outcomes: outcomes.count(False) >= 2), (
                f"heartbeat stopped retrying the busy fence after {probe.outcomes}; "
                f"ownership declared lost: {fire_claim_lost.is_set()}")
            assert not fire_claim_lost.is_set(), "a fence timeout was taken as ownership loss"
            seen["stamp"] = _stored_job(job["id"])["fire_claim"]["at"]
        # Holder gone: the next heartbeat takes the fence and confirms ownership again.
        attempts_before = len(probe.outcomes)
        assert probe.wait_until(
            lambda outcomes: len(outcomes) > attempts_before and outcomes[-1] is True)
        return True

    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)

    assert scheduler.run_one_job(claimed) is True

    refreshed = _stored_job(claimed["id"])["fire_claim"]
    assert refreshed["by"] == owner
    assert refreshed["at"] != seen["stamp"]


def test_heartbeat_gives_up_only_after_grace_of_unconfirmed_attempts(monkeypatch):
    """A fence that never frees still ends the run, but only once the unconfirmed streak has
    outlasted the heartbeat grace, on the same terms as a heartbeat that keeps raising."""
    import cron.scheduler as scheduler

    claimed = _claimed_recurring_job()
    probe = _FenceProbe(monkeypatch)
    _shrink_timing(monkeypatch)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 2.0)

    def run_body(job, *, fire_claim_lost, **_kwargs):
        with _fence_held_by_another_thread(job["id"]):
            assert fire_claim_lost.wait(10), "run outlived its grace with ownership unverified"
        return True

    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)

    assert scheduler.run_one_job(claimed) is True

    # The fence never freed, so every attempt timed out; the streak, not a single miss, ended it.
    assert True not in probe.outcomes
    assert probe.outcomes.count(False) >= 2


def test_heartbeat_declares_loss_on_the_first_beat_that_sees_another_owner(monkeypatch):
    """Getting the fence and finding another owner is conclusive: that heartbeat is the last one."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    claimed = _claimed_recurring_job()
    taken_over = threading.Event()
    probe = _FenceProbe(monkeypatch, gate=taken_over)
    _shrink_timing(monkeypatch)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 2.0)

    def run_body(job, *, fire_claim_lost, **_kwargs):
        # Another scheduler reclaims the expired fire while this runner is still going.
        assert jobs.claim_job_for_fire(job["id"], claim_ttl_seconds=0) is True
        taken_over.set()
        assert fire_claim_lost.wait(10), "a heartbeat that saw the replacement owner kept the run"
        return True

    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)

    assert scheduler.run_one_job(claimed) is True

    # Exactly one beat: it took the fence, saw the other owner, and the loop ended there.
    assert probe.outcomes == [True]


def test_claim_and_mark_job_run_fail_closed_while_the_fence_is_held(monkeypatch):
    """The heartbeat's tolerance must not leak: callers that mutate the claim still treat an
    unavailable fence as a refusal."""
    import cron.jobs as jobs

    claimed = _claimed_recurring_job()
    owner = claimed["fire_claim"]["by"]
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    refused = {}
    done = threading.Event()

    def fenced_writes():
        refused["mark"] = jobs.mark_job_run(claimed["id"], True, expected_fire_owner=owner)
        refused["claim"] = jobs.claim_job_for_fire(claimed["id"], claim_ttl_seconds=0)
        # The boolean heartbeat every other caller uses stays fail-closed as well.
        refused["heartbeat"] = jobs.heartbeat_fire_claim(claimed["id"], expected_owner=owner)
        done.set()

    with jobs._fire_job_lock(claimed["id"]) as acquired:
        assert acquired is True
        worker = threading.Thread(target=fenced_writes, daemon=True)
        worker.start()
        assert done.wait(5), "fenced writes waited past the lock timeout"

    worker.join(timeout=5)
    assert refused == {"mark": False, "claim": False, "heartbeat": False}
    stored = _stored_job(claimed["id"])
    assert stored["fire_claim"] == claimed["fire_claim"]
    assert stored["last_run_at"] is None
    # The refusals were the fence's, not the store's: the same write goes through once it is free.
    assert jobs.mark_job_run(claimed["id"], True, expected_fire_owner=owner) is True


def test_slow_delivery_holding_the_fence_does_not_fail_a_completed_run(monkeypatch):
    """Incident replay: delivery holds the fire fence across a slow send, heartbeats time out on
    it, and a run that completed AND delivered is still recorded as a success."""
    import cron.scheduler as scheduler

    claimed = _claimed_recurring_job()
    probe = _FenceProbe(monkeypatch)
    _shrink_timing(monkeypatch)
    finish = MagicMock()
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **_kwargs: (True, "output", "response", None))
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    def slow_delivery(job, content, **_kwargs):
        # The send outlives several heartbeat attempts while it holds the fence.
        probe.wait_until(lambda outcomes: outcomes.count(False) >= 2)
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", slow_delivery)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(claimed) is True

    recorded = _stored_job(claimed["id"])
    assert recorded["last_status"] == "ok", recorded["last_error"]
    assert recorded["last_error"] is None
    assert finish.call_args.kwargs["success"] is True
    assert finish.call_args.kwargs["error"] is None
    # Scenario sanity: the send really did hold the fence across timed-out heartbeats.
    assert probe.outcomes.count(False) >= 2, "delivery did not actually cross a timed-out beat"
