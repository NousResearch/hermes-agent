"""Fire-claim heartbeat must not treat a busy per-job fire fence as ownership loss.

The fence is held across delivery (a Telegram send with retries can run 30–60s).
The heartbeat runs on a background thread. Before this fix it acquired that same
fence, timed out at 30s (_JOBS_LOCK_TIMEOUT_SECONDS), and returned False — which
the heartbeat loop treated as a verified takeover and interrupted a run that had
already finished and delivered.

These tests fail on unpatched main: the first returns False (or blocks ~30s) while
the fence is held; the second stamps a successful slow delivery as interrupted.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_heartbeat_succeeds_immediately_while_fire_fence_held(temp_home):
    """Root cause: heartbeat must refresh the claim without waiting on the fire fence."""
    from cron.jobs import (
        claim_job_for_fire,
        create_job,
        fire_claim_fence,
        get_job,
        heartbeat_fire_claim,
    )

    job = create_job(prompt="x", schedule="every 5m", name="heartbeat-fence")
    job_id = job["id"]
    assert claim_job_for_fire(job_id) is True
    claimed = get_job(job_id)
    assert claimed is not None
    owner = claimed["fire_claim"]["by"]

    fence_entered = threading.Event()
    release = threading.Event()
    result = {}

    def _hold():
        with fire_claim_fence(job_id, expected_owner=owner) as owns:
            assert owns is True
            fence_entered.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert fence_entered.wait(timeout=5)

    start = time.monotonic()
    result["ok"] = heartbeat_fire_claim(job_id, expected_owner=owner)
    result["waited"] = time.monotonic() - start
    release.set()
    holder.join(timeout=5)

    assert result["ok"] is True, "heartbeat treated a busy fence as ownership loss"
    assert result["waited"] < 2.0, f"heartbeat blocked on the fire fence ({result['waited']:.2f}s)"
    assert heartbeat_fire_claim(job_id, expected_owner="other-owner") is False


def _claimed_job(tmp_path, name):
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name=name)
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed = jobs.get_job(job["id"])
    assert isinstance(claimed, dict)
    return jobs, profile_home, claimed


def _patch_run_scaffold(monkeypatch, scheduler, *, run_job):
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a, **_kw: {})
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda *_a, **_kw: False)
    monkeypatch.setattr(scheduler, "run_job", run_job)


def test_slow_delivery_holding_fence_is_not_recorded_as_ownership_loss(tmp_path, monkeypatch):
    """A delivery that outlives the fence wait must still be recorded successful."""
    import cron.scheduler as scheduler

    jobs, profile_home, claimed = _claimed_job(tmp_path, "slow-delivery")

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 30.0)

    delivered = threading.Event()
    body_saw_loss: list[bool] = []

    def _slow_delivery(*_args, **_kwargs):
        delivered.set()
        time.sleep(0.5)
        return None

    real_body = scheduler._run_one_job_body

    def _observed_body(job, **kwargs):
        result = real_body(job, **kwargs)
        lost = kwargs.get("fire_claim_lost")
        body_saw_loss.append(bool(lost is not None and lost.is_set()))
        return result

    _patch_run_scaffold(
        monkeypatch, scheduler,
        run_job=lambda *_a, **_kw: (True, "output", "response", None))
    monkeypatch.setattr(scheduler, "_run_one_job_body", _observed_body)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "output.md")
    monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)
    mark_run = MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "mark_job_run", mark_run)
    monkeypatch.setattr(scheduler, "finish_execution", MagicMock())

    with jobs.use_cron_store(profile_home), \
         patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(claimed) is True

    assert delivered.is_set(), "delivery never ran"
    assert body_saw_loss == [False], "a fire fence timeout was treated as ownership loss"
    args, _kwargs = mark_run.call_args
    assert args[1] is True, "the run was not recorded as successful"
    assert args[2] is None, f"run was stamped with an error: {args[2]!r}"


def test_fence_held_elsewhere_skips_side_effects_without_ownership_lost_stamp(
    tmp_path, monkeypatch,
):
    """Side effect that cannot acquire the fence is skipped; not an ownership loss."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="fence-busy")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed = jobs.get_job(job["id"])
        assert isinstance(claimed, dict)
        owner = claimed["fire_claim"]["by"]

        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)

        holder_ready = threading.Event()
        release_holder = threading.Event()

        def _hold_fence() -> None:
            with jobs.use_cron_store(profile_home):
                with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owns:
                    assert owns is True, "the holder must own the fence for this test"
                    holder_ready.set()
                    release_holder.wait(timeout=10)

        holder = threading.Thread(target=_hold_fence, daemon=True, name="fence-holder")

        def _start_holder_then_finish(*_a, **_kw):
            holder.start()
            assert holder_ready.wait(timeout=5)
            return True, "output", "response", None

        save_output = MagicMock(return_value="output.md")
        deliver = MagicMock(return_value=None)
        mark_run = MagicMock(return_value=True)
        finish = MagicMock()
        _patch_run_scaffold(monkeypatch, scheduler, run_job=_start_holder_then_finish)
        monkeypatch.setattr(scheduler, "save_job_output", save_output)
        monkeypatch.setattr(scheduler, "_deliver_result", deliver)
        monkeypatch.setattr(scheduler, "mark_job_run", mark_run)
        monkeypatch.setattr(scheduler, "finish_execution", finish)

        try:
            with patch("agent.secret_scope.set_secret_scope", return_value=None), \
                 patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
                 patch("agent.secret_scope.reset_secret_scope"):
                assert scheduler.run_one_job(claimed) is True
        finally:
            release_holder.set()
            holder.join(timeout=5)

    save_output.assert_not_called()
    deliver.assert_not_called()
    finish.assert_called_once()
    _args, fkwargs = finish.call_args
    error_text = str(fkwargs.get("error") or "")
    assert "fence" in error_text.lower(), f"honest cause not recorded: {error_text!r}"
    interrupted = "Interrupted by shutdown" in error_text
    assert not interrupted, f"busy fence stamped as shutdown interrupt: {error_text!r}"
