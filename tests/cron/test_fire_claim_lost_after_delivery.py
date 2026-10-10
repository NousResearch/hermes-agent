"""#105861 / #113357: a fire-claim that only *reads* as lost must not overwrite a completed run's
terminal status — an ok, or a failure notice carrying its real error.

`_FireOwnership.lost()` samples the claim from the store once. When that sample misses after
the notice already reached the channel, the run must fall through to the owner-fenced terminal
write in `_finish_completed_run` (the authoritative claim check) instead of recording an error.

These drive the real store (``jobs.json`` under a temp HERMES_HOME) so the assertion is the
actual on-disk ``last_status`` the health watchdog reads, not a mock's call list.
"""

import threading

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


class _SampledHeartbeat:
    """The real ``heartbeat_fire_claim``, with exactly ONE armed sample missing."""

    def __init__(self, real, samples_before_miss: int):
        self._real = real
        self._samples_before_miss = samples_before_miss
        self._armed = False
        self._seen = 0
        self.missed = 0

    def __call__(self, job_id, *, expected_owner):
        if self._armed:
            self._seen += 1
            if self._seen > self._samples_before_miss:
                self._armed = False
                self.missed += 1
                return False
        return self._real(job_id, expected_owner=expected_owner)

    def arm(self):
        """Answer ``samples_before_miss`` samples truthfully from here on, then miss once."""
        self._seen = 0
        self._armed = True


def _claimed_job():
    """A real recurring job record holding a live fire claim (as a firing tick has)."""
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="105861")
    assert claim_job_for_fire(job["id"]) is True
    job = get_job(job["id"])
    assert isinstance(job.get("fire_claim"), dict) and job["fire_claim"].get("by")
    return job


def _drive(monkeypatch, *, run_result, samples_before_miss):
    """Run one job through run_one_job with the sampled-heartbeat harness.

    After ``run_job`` returns the claim is sampled once (the pre-delivery check), once again just
    before the delivery side effect, and once after it (the post-delivery check that the incidents
    tripped on). ``samples_before_miss=2`` therefore makes the third, post-delivery sample miss.
    """
    import cron.scheduler as sched

    job = _claimed_job()
    hb = _SampledHeartbeat(sched.heartbeat_fire_claim, samples_before_miss=samples_before_miss)
    delivered = []

    def fake_run_job(job, **kwargs):
        hb.arm()
        return run_result

    def fake_deliver(job, content, **kwargs):
        delivered.append(content)

    monkeypatch.setattr(sched, "heartbeat_fire_claim", hb)
    monkeypatch.setattr(sched, "run_job", fake_run_job)
    monkeypatch.setattr(sched, "_deliver_result", fake_deliver)
    return sched, job, hb, delivered


_REAL_ERROR = "the real error text the failure path recorded"


@pytest.mark.parametrize(
    "success, run_result, expected_status, expected_error",
    [
        (True, (True, "output text", "the report", None), "ok", None),
        (False, (False, "output text", "", _REAL_ERROR), "error", _REAL_ERROR),
    ],
    ids=["delivered-ok", "delivered-failure-notice"],
)
def test_delivered_run_keeps_its_terminal_status_when_claim_sample_misses_after_delivery(
    temp_home, monkeypatch, success, run_result, expected_status, expected_error,
):
    """Delivery completed, then the claim sample missed → the on-disk record is the delivered
    run's real outcome (ok, or the real error — never ``_OWNERSHIP_LOST_INTERRUPTED``), and the
    loss latch stays on the raw source: the caller's idle transport event is left unset."""
    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(monkeypatch, run_result=run_result, samples_before_miss=2)
    cancel = threading.Event()

    assert sched.run_one_job(job, cancel_event=cancel) is True

    assert len(delivered) == 1, "the notice must have left the process"
    if success:
        assert delivered == ["the report"]
    else:
        assert "failed" in delivered[0], "the failure notice, not the (empty) report, was delivered"
    assert hb.missed == 1, "exactly the post-delivery sample missed"
    assert cancel.is_set() is False, "the run's loss latch must not cancel the caller's transport"
    record = get_job(job["id"])
    assert record["last_status"] == expected_status
    assert record["last_error"] == expected_error
    assert record["last_error"] != sched._OWNERSHIP_LOST_INTERRUPTED
    assert record["failure_streak"] == (0 if success else 1)


def test_transport_cancel_during_delivery_stays_fail_closed(temp_home, monkeypatch):
    """An explicit transport cancel during delivery is not a sampled miss → still interrupted."""
    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(
        monkeypatch, run_result=(True, "output text", "the report", None),
        samples_before_miss=99)
    cancel = threading.Event()
    deliver_result = sched._deliver_result

    def deliver_then_cancel(job, content, **kwargs):
        outcome = deliver_result(job, content, **kwargs)
        cancel.set()
        return outcome

    monkeypatch.setattr(sched, "_deliver_result", deliver_then_cancel)

    assert sched.run_one_job(job, cancel_event=cancel) is True

    assert delivered == ["the report"], "the notice had already left the process"
    assert hb.missed == 0, "the sampled claim never missed — only the transport event fired"
    record = get_job(job["id"])
    assert record["last_status"] == "error"
    assert record["last_error"] == sched._OWNERSHIP_LOST_INTERRUPTED


class _HeartbeatThreadMisses:
    """The real ``heartbeat_fire_claim``; the first ``misses`` samples taken on the heartbeat
    thread return False (optionally re-owning the stored claim so the loss is genuine)."""

    def __init__(self, real, misses: int, *, steal: bool = False):
        self._real, self._left, self._steal = real, misses, steal
        self.missed = 0
        self._samples = 0
        self.processed = threading.Event()

    def __call__(self, job_id, *, expected_owner):
        if threading.current_thread().name == "cron-fire-claim-heartbeat":
            self._samples += 1
            if self._samples > 2:
                # Reaching the next poll proves the confirmation was accepted without latching.
                self.processed.set()
        if threading.current_thread().name == "cron-fire-claim-heartbeat" and self._left:
            self._left -= 1
            self.missed += 1
            if self._steal:
                from cron.jobs import _with_job, save_jobs

                def re_own(jobs, _i, job):
                    job["fire_claim"] = {**job["fire_claim"], "by": "replacement:deadbeef"}
                    save_jobs(jobs)

                _with_job(job_id, re_own)
            return False
        return self._real(job_id, expected_owner=expected_owner)


def _drive_heartbeat_thread(monkeypatch, *, misses, steal=False):
    """run_one_job with the REAL fire-claim heartbeat thread sampling every 10 ms; ``run_job``
    finishes after the heartbeat has processed its confirmation, not merely sampled it."""
    import cron.scheduler as sched

    job = _claimed_job()
    hb = _HeartbeatThreadMisses(sched.heartbeat_fire_claim, misses, steal=steal)
    delivered, run_cancel = [], []
    start_heartbeat = sched._start_heartbeat_thread

    def observed_heartbeat(loop_fn, name, fail_log):
        def run():
            try:
                loop_fn()
            finally:
                # A confirmed loss exits the loop only after setting the real cancel latch.
                hb.processed.set()

        return start_heartbeat(run, name, fail_log)

    def fake_run_job(job, **kwargs):
        # Deadlock guard only: store I/O and thread scheduling have no 100 ms upper bound.
        assert hb.processed.wait(timeout=30), "heartbeat never processed its confirmation"
        run_cancel.append(kwargs["cancel_event"].is_set())
        return True, "output text", "the report", None

    monkeypatch.setattr(sched, "_start_heartbeat_thread", observed_heartbeat)
    monkeypatch.setattr(sched, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(sched, "_FIRE_CLAIM_MISS_CONFIRM_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(sched, "heartbeat_fire_claim", hb)
    monkeypatch.setattr(sched, "run_job", fake_run_job)
    monkeypatch.setattr(sched, "_deliver_result", lambda job, content, **kw: delivered.append(content))
    return sched, job, hb, delivered, run_cancel


@pytest.mark.parametrize("misses, latched", [(1, False), (2, True)], ids=["one-sample", "latched"])
def test_heartbeat_miss_mid_run_keeps_completed_run(temp_home, monkeypatch, misses, latched):
    """#113357: the heartbeat thread's miss is a sample, not the verdict. One miss is re-sampled
    and never cancels the run; two consecutive misses latch, but a claim the store still validates
    at completion records the completed run — not ``_OWNERSHIP_LOST_INTERRUPTED``."""
    from cron.executions import get_execution
    from cron.jobs import get_job

    sched, job, hb, delivered, run_cancel = _drive_heartbeat_thread(monkeypatch, misses=misses)

    assert sched.run_one_job(job) is True

    assert hb.missed == misses
    assert run_cancel == [latched], "only a confirmed miss reaches the run's cancel event"
    assert delivered == ["the report"]
    record = get_job(job["id"])
    assert record["last_status"] == "ok"
    assert record["last_error"] is None
    assert get_execution(job["execution_id"])["status"] == "completed"


def test_confirmed_claim_loss_mid_run_still_yields(temp_home, monkeypatch):
    """A genuinely re-owned claim (two misses, stored ``by`` rewritten) still fences the run out:
    nothing delivered, no terminal write over the new owner, ledger records the discard."""
    from cron.executions import get_execution
    from cron.jobs import get_job

    sched, job, _hb, delivered, run_cancel = _drive_heartbeat_thread(
        monkeypatch, misses=2, steal=True)

    assert sched.run_one_job(job) is True

    assert run_cancel == [True]
    assert delivered == []
    record = get_job(job["id"])
    assert record["last_status"] is None
    assert record["fire_claim"]["by"] == "replacement:deadbeef"
    assert "discarded" in get_execution(job["execution_id"])["error"]


_UNKNOWN_DELIVERY_ERROR = (
    "timed out while gateway delivery was in progress; outcome is unknown and was not retried"
)


@pytest.mark.parametrize(
    "run_result, expected_status, expected_error, expected_streak",
    [
        ((True, "output text", "the report", None), "delivery_failed", None, 0),
        ((False, "output text", "", _REAL_ERROR), "error", _REAL_ERROR, 1),
    ],
    ids=["unknown-delivered-ok", "unknown-delivered-failure-notice"],
)
def test_unknown_fenced_delivery_keeps_terminal_status_when_claim_sample_misses(
    temp_home, monkeypatch, run_result, expected_status, expected_error, expected_streak,
):
    """#132078: the restart-safe worker's wait budget fences the send ``unknown`` while the
    gateway is still mid-send; the message lands anyway, then the runner tail dies and the
    post-delivery claim sample misses. The run must keep its real outcome — an ok run stays a
    delivery-only blemish (streak untouched), a failed run keeps its real error — instead of
    recording ``_OWNERSHIP_LOST_INTERRUPTED`` for a shutdown that never happened."""
    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(
        monkeypatch, run_result=run_result, samples_before_miss=2)

    def fence_unknown(delivery_job, content, **kwargs):
        delivered.append(content)
        delivery_job["_delivery_outcome_unknown"] = True
        return _UNKNOWN_DELIVERY_ERROR

    monkeypatch.setattr(sched, "_deliver_result", fence_unknown)

    assert sched.run_one_job(job) is True

    assert len(delivered) == 1, "the notice had left the worker process"
    assert hb.missed == 1, "exactly the post-delivery sample missed"
    record = get_job(job["id"])
    assert record["last_status"] == expected_status
    assert record["last_error"] == expected_error
    assert record["last_error"] != sched._OWNERSHIP_LOST_INTERRUPTED
    assert record["failure_streak"] == expected_streak
    assert record["last_delivery_error"] == _UNKNOWN_DELIVERY_ERROR


def test_worker_wait_timeout_stamps_the_unknown_delivery_outcome(monkeypatch, tmp_path):
    """The worker's queue branch marks a delivery the queue fenced ``unknown`` on the job, so the
    post-delivery ownership check can tell "uncertain" from "failed"; a later confirmed outcome
    re-derives the marker per attempt (#132078)."""
    from cron import scheduler_delivery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "cron.delivery_queue.enqueue_and_wait",
        lambda execution_id, job, content, **kw: _UNKNOWN_DELIVERY_ERROR)
    monkeypatch.setattr(
        "cron.delivery_queue.get_status",
        lambda execution_id: {"status": "unknown", "error": _UNKNOWN_DELIVERY_ERROR})
    monkeypatch.setattr("cron.jobs.get_job", lambda job_id: {})
    monkeypatch.setattr(
        scheduler_delivery, "_resolve_delivery_targets",
        lambda job, for_failure=False: [{"platform": "telegram", "chat_id": "123"}])
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-1")

    job = {"id": "job-1", "execution_id": "exec-1", "deliver": "telegram:123"}
    assert scheduler_delivery._deliver_result(
        job, "done", adapters=None, loop=None) == _UNKNOWN_DELIVERY_ERROR
    assert job.get("_delivery_outcome_unknown") is True

    monkeypatch.setattr(
        "cron.delivery_queue.enqueue_and_wait",
        lambda execution_id, job, content, **kw: None)
    monkeypatch.setattr(
        "cron.delivery_queue.get_status",
        lambda execution_id: {"status": "delivered", "error": None})
    assert scheduler_delivery._deliver_result(job, "done", adapters=None, loop=None) is None
    assert job.get("_delivery_outcome_unknown") is None
