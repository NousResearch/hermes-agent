"""#105861: a fire-claim that only *reads* as lost AFTER a run delivered must not overwrite
the delivered run's terminal status — an ok, or a failure notice carrying its real error.

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
        return None

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
