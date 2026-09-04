"""Regression coverage for false "Fire claim ownership lost" discards (#95307).

A finite cron job delivering to Bot Chat runs a full child-agent turn while
the runner holds the per-job fire fence. The fire-claim heartbeat contends
with that same fence and used to read the lock timeout as a definitive
ownership loss, so the completed result was discarded instead of delivered.

Contract under test:

* ``heartbeat_fire_claim`` is tri-state: ``True`` (owned + refreshed),
  ``False`` (definitive loss), ``None`` (fence busy — ownership UNKNOWN).
* A heartbeat beat that cannot acquire the fence never sets the
  ownership-lost event while the legitimate runner holds it.
* A definitive takeover (claim rewritten to another owner) is still detected.
* Startup validation distinguishes "unverifiable" from "lost".
"""

import contextlib
import threading
import time
import unittest.mock as mock

import cron.jobs as jobs
import cron.scheduler as scheduler


@contextlib.contextmanager
def _secret_scope_patched():
    """No-op the per-run secret scope install/teardown for handed-in jobs."""
    with mock.patch(
        "agent.secret_scope.set_secret_scope", return_value=None
    ), mock.patch(
        "agent.secret_scope.build_profile_secret_scope", return_value=None
    ), mock.patch("agent.secret_scope.reset_secret_scope"):
        yield


def _make_claimed_job(profile_home, name):
    """Create a one-shot job, claim its fire; returns (job, claimed, owner)."""
    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name=name)
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed = jobs.get_job(job["id"])
        owner = claimed["fire_claim"]["by"]
    return job, claimed, owner


def test_heartbeat_fire_claim_is_unknown_while_fence_is_held(tmp_path, monkeypatch):
    """Fence unavailable ⇒ None (unknown), never False (loss).

    The per-job fence fails closed when cross-process locking cannot be
    acquired within the budget — a cross-process holder (or an unavailable
    locking backend) must read as UNKNOWN ownership, not as a takeover
    (#95307). In-process holders block on the threading lock instead and are
    covered by the delayed-refresh guard below.
    """

    @contextlib.contextmanager
    def _busy_fence(_job_id):
        yield False

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    job, _claimed, owner = _make_claimed_job(profile_home, "tri-state-store")

    real_fire_lock = jobs._fire_job_lock
    monkeypatch.setattr(jobs, "_fire_job_lock", _busy_fence)
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is None

    # Sanity: uncontended calls still resolve through the real fence.
    monkeypatch.setattr(jobs, "_fire_job_lock", real_fire_lock)
    with jobs.use_cron_store(profile_home):
        assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is True
        assert (
            jobs.heartbeat_fire_claim(job["id"], expected_owner="someone-else")
            is False
        )


def test_run_completes_when_delivery_spans_heartbeat_beats(tmp_path, monkeypatch):
    """A fenced Bot-Chat-style delivery spanning beats must still deliver.

    Mirrors #95307: delivery holds the side-effect fence longer than several
    heartbeat intervals; every beat inside that window contends with OUR OWN
    fence, which is not a takeover. The run must finalize successfully.
    """
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    _job, claimed, _owner = _make_claimed_job(profile_home, "fenced-delivery")
    claimed["execution_id"] = "exec-fenced-delivery"
    claimed["deliver"] = "local"

    delivery_started = threading.Event()
    delivery_may_finish = threading.Event()

    def _slow_delivery(job, content, adapters=None, loop=None, **kwargs):
        delivery_started.set()
        # Hold the fire fence well past several heartbeat intervals.
        assert delivery_may_finish.wait(timeout=5)
        return None

    ledger_calls = []

    def _ledger(execution_id, success, error=None, **kwargs):
        ledger_calls.append({"success": success, "error": error})

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None, extra_prompt=None,
        cancel_event=None: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda job_id, output: "p")
    monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)
    monkeypatch.setattr(scheduler, "finish_execution", _ledger)
    mark_run = mock.MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "mark_job_run", mark_run)

    with jobs.use_cron_store(profile_home), _secret_scope_patched():

        def _finish_delivery():
            assert delivery_started.wait(timeout=5)
            time.sleep(0.3)  # > 4 heartbeat intervals inside the fence
            delivery_may_finish.set()

        finisher = threading.Thread(target=_finish_delivery, daemon=True)
        finisher.start()
        try:
            assert scheduler.run_one_job(claimed) is True
        finally:
            delivery_may_finish.set()
            finisher.join(timeout=5)

    failures = [call for call in ledger_calls if call["success"] is False]
    assert not failures, (
        "successful run was failed by false ownership loss: %s" % ledger_calls
    )
    assert any(call["success"] is True for call in ledger_calls), ledger_calls


def test_definitive_takeover_during_run_still_detected(tmp_path, monkeypatch):
    """A real owner change mid-run stays a loss — tri-state must not hide it."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    _job, claimed, _owner = _make_claimed_job(profile_home, "real-takeover")
    claimed["execution_id"] = "exec-real-takeover"
    claimed["deliver"] = "local"

    started = threading.Event()
    beat_after_start = threading.Event()
    real_heartbeat = jobs.heartbeat_fire_claim

    def _observing_heartbeat(job_id: str, *, expected_owner: str):
        status = real_heartbeat(job_id, expected_owner=expected_owner)
        if status is True and started.is_set():
            beat_after_start.set()
        return status

    def _run_job(job, defer_agent_teardown=None, extra_prompt=None, cancel_event=None):
        started.set()
        assert beat_after_start.wait(timeout=5)
        with jobs.use_cron_store(profile_home):
            stored = jobs.get_job(job["id"])
            stored["fire_claim"] = {
                "at": stored["fire_claim"]["at"],
                "by": "replacement-owner",
            }
            jobs.save_jobs([stored])
        return True, "out", "stale response", None

    ledger_calls = []

    def _ledger(execution_id, success, error=None, **kwargs):
        ledger_calls.append({"success": success, "error": error})

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observing_heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _run_job)
    monkeypatch.setattr(
        scheduler, "save_job_output", mock.MagicMock(return_value="p")
    )
    monkeypatch.setattr(
        scheduler, "_deliver_result", mock.MagicMock(return_value=None)
    )
    monkeypatch.setattr(scheduler, "finish_execution", _ledger)
    monkeypatch.setattr(
        scheduler, "mark_job_run", mock.MagicMock(return_value=True)
    )

    with _secret_scope_patched():
        assert scheduler.run_one_job(claimed) is True

    assert any(
        call["success"] is False and "ownership lost" in str(call["error"]).lower()
        for call in ledger_calls
    ), f"a definitive takeover was not detected: {ledger_calls}"


def test_initial_validation_busy_reports_unverifiable_not_lost(
    tmp_path, monkeypatch
):
    """Busy-at-startup closes the row as 'could not be validated', not 'lost'."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    _job, claimed, _owner = _make_claimed_job(profile_home, "busy-start")
    claimed["execution_id"] = "exec-busy-start"

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: None
    )
    body = mock.MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "_run_one_job_body", body)
    finish = mock.MagicMock()
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    assert scheduler.run_one_job(claimed) is True
    body.assert_not_called()
    finish.assert_called_once_with(
        "exec-busy-start",
        success=False,
        error=(
            "Fire claim ownership could not be validated "
            "before execution started."
        ),
    )


def test_initial_validation_definitive_loss_still_reported(tmp_path, monkeypatch):
    """Definitive loss at startup keeps its distinct 'ownership lost' record."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    _job, claimed, _owner = _make_claimed_job(profile_home, "lost-start")
    claimed["execution_id"] = "exec-lost-start"

    monkeypatch.setattr(
        scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: False
    )
    body = mock.MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "_run_one_job_body", body)
    finish = mock.MagicMock()
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    assert scheduler.run_one_job(claimed) is True
    body.assert_not_called()
    finish.assert_called_once_with(
        "exec-lost-start",
        success=False,
        error="Fire claim ownership lost before execution started.",
    )


def test_terminal_bookkeeping_treats_busy_as_not_lost(tmp_path, monkeypatch):
    """Post-delivery probe with a busy fence completes instead of discarding."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    _job, claimed, _owner = _make_claimed_job(profile_home, "busy-terminal")
    claimed["execution_id"] = "exec-busy-terminal"
    claimed["deliver"] = "local"

    calls = {"n": 0}

    def _busy_after_run_heartbeat(job_id: str, *, expected_owner: str):
        calls["n"] += 1
        # Call #1 = startup validation (owned). Later calls = the probes the
        # body makes around delivery/terminal writes; report them as busy so
        # only the tri-state-aware paths can proceed.
        if calls["n"] == 1:
            return True
        return None

    ledger_calls = []

    def _ledger(execution_id, success, error=None, **kwargs):
        ledger_calls.append({"success": success, "error": error})

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _busy_after_run_heartbeat)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None, extra_prompt=None,
        cancel_event=None: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda job_id, output: "p")
    monkeypatch.setattr(
        scheduler, "_deliver_result", mock.MagicMock(return_value=None)
    )
    monkeypatch.setattr(scheduler, "finish_execution", _ledger)
    monkeypatch.setattr(
        scheduler, "mark_job_run", mock.MagicMock(return_value=True)
    )

    with jobs.use_cron_store(profile_home), _secret_scope_patched():
        assert scheduler.run_one_job(claimed) is True

    assert any(call["success"] is True for call in ledger_calls), ledger_calls
