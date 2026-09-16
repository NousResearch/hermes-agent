"""#105861: a fire-claim that only *reads* as lost AFTER a run delivered must not overwrite
the delivered success with an error.

`_FireOwnership.lost()` samples the claim from the store once. In the reported incidents that
sample missed (a run's own delivery holds the per-job fence past `_JOBS_LOCK_TIMEOUT_SECONDS`,
and jobs.json is rewritten by two processes per run), the run wrapper then recorded
`last_status: error` — and the operator's health watchdog alerted on every tick for a job whose
output had already reached the channel.

Suggested fix #1 (the smallest one that closes it): when the run's delivery actually completed,
the sampled claim loss is bookkeeping, not a delivery failure — warn and fall through to the
owner-fenced terminal write in `_finish_completed_run`, which is the authoritative claim check
(it records `ok` only while the claim is still held, and records nothing when it really is gone).

These drive the real store (``jobs.json`` under a temp HERMES_HOME) so the assertion is the
actual on-disk ``last_status`` the health watchdog reads, not a mock's call list.
"""

import logging

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
    """The real ``heartbeat_fire_claim``, with exactly ONE armed sample missing.

    That single miss is the whole bug: the claim is sampled once, a miss flips the run's
    ownership flag permanently, and the *next* heartbeat (inside
    ``_record_fire_ownership_lost``) confirms the claim is still ours — which is how a
    delivered run ended up recorded as ``error``.
    """

    def __init__(self, real, skip_samples: int):
        self._real = real
        self._skip = skip_samples
        self._armed = False
        self._seen = 0
        self.missed = 0

    def __call__(self, job_id, *, expected_owner):
        if self._armed:
            self._seen += 1
            if self._seen > self._skip:
                self._armed = False
                self.missed += 1
                return False
        return self._real(job_id, expected_owner=expected_owner)

    def arm(self):
        """Miss on the ``skip_samples + 1``-th sample taken from here on."""
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


def _drive(monkeypatch, *, final_response, deliver, miss_at_sample):
    """Run one job through run_one_job with the sampled-heartbeat harness.

    The claim is sampled once after ``run_job`` returns (the pre-delivery check), once again
    just before the delivery side effect, and once after it (the post-delivery check that the
    incidents tripped on). ``miss_at_sample`` picks which of those samples misses.
    """
    import cron.scheduler as sched

    job = _claimed_job()
    hb = _SampledHeartbeat(sched.heartbeat_fire_claim, skip_samples=miss_at_sample)
    delivered = []

    def fake_run_job(job, **kwargs):
        hb.arm()
        return (True, "output text", final_response, None)

    def fake_deliver(job, content, **kwargs):
        delivered.append(content)
        return deliver

    monkeypatch.setattr(sched, "heartbeat_fire_claim", hb)
    monkeypatch.setattr(sched, "run_job", fake_run_job)
    monkeypatch.setattr(sched, "_deliver_result", fake_deliver)
    return sched, job, hb, delivered


def test_delivered_run_stays_ok_when_claim_sample_misses_after_delivery(
    temp_home, monkeypatch, caplog,
):
    """① Delivery completed, then the claim sample missed → last_status stays ok + warning.

    Sample 1 is the pre-delivery check, sample 2 the one guarding the side effect; the third
    (post-delivery) is the one the reported incidents hit.
    """
    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(
        monkeypatch, final_response="the report", deliver=None, miss_at_sample=2)

    with caplog.at_level(logging.WARNING):
        assert sched.run_one_job(job) is True

    assert delivered == ["the report"], "the notice must have left the process"
    assert hb.missed == 1, "exactly the post-delivery sample missed"
    record = get_job(job["id"])
    assert record["last_status"] == "ok"
    assert record["last_error"] is None
    assert record["failure_streak"] == 0
    warnings = " | ".join(r.getMessage().lower() for r in caplog.records)
    assert "ownership lost" in warnings and "deliver" in warnings


def test_claim_lost_during_delivery_still_records_error(temp_home, monkeypatch):
    """②a A delivery that FAILED, then a lost claim → error (the protection must not weaken)."""
    from cron.jobs import get_job

    sched, job, _hb, delivered = _drive(
        monkeypatch, final_response="the report", deliver="send failed: 502", miss_at_sample=2)

    assert sched.run_one_job(job) is True

    assert delivered == ["the report"]
    record = get_job(job["id"])
    assert record["last_status"] == "error"
    assert record["last_error"] == sched._OWNERSHIP_LOST_INTERRUPTED


def test_claim_lost_with_delivery_never_attempted_still_records_error(temp_home, monkeypatch):
    """②b A SILENT run delivers nothing, then a lost claim → error, not a delivered-ok.

    No delivery means no delivery phase to guard: sample 2 here is the post-delivery check.
    """
    import cron.scheduler as sched
    from cron.jobs import get_job

    sched, job, _hb, delivered = _drive(
        monkeypatch, final_response=sched.SILENT_MARKER, deliver=None, miss_at_sample=1)

    assert sched.run_one_job(job) is True

    assert delivered == [], "a SILENT run must not deliver"
    record = get_job(job["id"])
    assert record["last_status"] == "error"
    assert record["last_error"] == sched._OWNERSHIP_LOST_INTERRUPTED


def test_claim_lost_before_delivery_still_records_error(temp_home, monkeypatch):
    """③ A claim sampled as lost before the side effect → error, delivery untouched."""
    from cron.jobs import get_job

    sched, job, _hb, delivered = _drive(
        monkeypatch, final_response="the report", deliver=None, miss_at_sample=0)

    assert sched.run_one_job(job) is True

    assert delivered == [], "a run that lost its claim must not deliver"
    record = get_job(job["id"])
    assert record["last_status"] == "error"
    assert record["last_error"] == sched._OWNERSHIP_LOST_INTERRUPTED


def test_transport_cancel_during_delivery_stays_fail_closed(temp_home, monkeypatch):
    """④ The transport cancel is an explicit stop signal, not a sampled miss → still interrupted.

    ``fire_claim_lost`` ORs the sampled claim with the transport-level ``cancel_event``, so the
    delivered-success exemption must not swallow a drain that arrived while the notice was being
    sent: the run's provider relies on that cancellation being fail-closed.
    """
    import threading

    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(
        monkeypatch, final_response="the report", deliver=None, miss_at_sample=99)
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


def test_sampled_miss_leaves_an_idle_transport_event_alone(temp_home, monkeypatch):
    """⑤ A sampled miss with an (unset) transport cancel supplied → the run's own loss latch
    must not read as a transport cancellation, so the delivered run stays ok.

    ``_FireOwnership.lost()`` latches a sampled loss by calling ``set()`` on ``fire_claim_lost``.
    With a transport ``cancel_event`` in play that object is ``_CombinedCancelEvent(lost_ownership,
    cancel_event)``, and its ``set()`` sets every source — so the transport event read as cancelled
    and the delivered-run exemption was skipped, recording the same false interrupted error the
    issue reports. The latch has to carry the cancellation source's identity.
    """
    import threading

    from cron.jobs import get_job

    sched, job, hb, delivered = _drive(
        monkeypatch, final_response="the report", deliver=None, miss_at_sample=2)
    cancel = threading.Event()

    assert sched.run_one_job(job, cancel_event=cancel) is True

    assert delivered == ["the report"], "the notice had already left the process"
    assert hb.missed == 1, "exactly the post-delivery sample missed"
    assert cancel.is_set() is False, "the run's loss latch must not cancel the caller's transport"
    record = get_job(job["id"])
    assert record["last_status"] == "ok"
    assert record["last_error"] is None
    assert record["failure_streak"] == 0
