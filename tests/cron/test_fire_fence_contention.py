"""The per-job fire fence is a serialization primitive, not an ownership signal.

A fence acquire timeout means a same-job holder (typically the in-flight save/delivery) kept
us out — ownership is *unverified*, not lost. Treating it as loss stamps a successfully
delivered run with "Interrupted by shutdown before terminal completion." and, on the sibling
path, discards a run whose side effects were never attempted.

Reproduction of the 2026-09-12 shape: a Telegram send held the side-effect fence for 33s while
the fire-claim heartbeat waited 30s (``_JOBS_LOCK_TIMEOUT_SECONDS``) for the same fence, timed
out, and reported ownership loss on a run that had delivered (message_id=43288).
"""

import threading
import time
from unittest.mock import MagicMock, patch


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
    monkeypatch.setattr(scheduler, "run_job", run_job)


def test_fence_contention_during_slow_delivery_is_not_ownership_loss(tmp_path, monkeypatch):
    """A delivery that outlives the fence wait must not be recorded as an ownership loss."""
    import cron.scheduler as scheduler

    jobs, profile_home, claimed = _claimed_job(tmp_path, "slow-delivery")

    # Compressed stand-ins for the production constants: the delivery outlives the fence wait
    # several times over, while the grace window (the real 180s) stays comfortably open.
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
    """A side effect that cannot acquire the fence is skipped and reported as such.

    Ownership is unverified there, so the run must not be recorded as a lost/interrupted run
    (the fail-closed policy for the side effect itself is unchanged: nothing is written or sent).
    """
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
            # ContextVars do not cross a bare thread boundary: re-enter the profile store so the
            # holder resolves the same cron store as the run under test.
            with jobs.use_cron_store(profile_home):
                with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owns:
                    assert owns is True, "the holder must own the fence for this test"
                    holder_ready.set()
                    release_holder.wait(timeout=10)

        holder = threading.Thread(target=_hold_fence, daemon=True, name="fence-holder")

        def _start_holder_then_finish(*_a, **_kw):
            # The run starts with the fence free (initial claim validation succeeds); the holder
            # takes it once the run is under way, like a concurrent delivery.
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
    assert "ownership lost" not in error_text.lower(), error_text
    assert "interrupted" not in error_text.lower(), error_text
