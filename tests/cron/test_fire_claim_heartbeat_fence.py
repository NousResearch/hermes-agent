"""Fire-claim heartbeat must not treat a busy per-job fire fence as ownership loss.

The fence is held across delivery. Heartbeat tries it with timeout 0: busy returns
None (no write) instead of blocking 30s and returning False. False remains verified
takeover only.
"""
import contextlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_heartbeat_is_unverified_not_loss_while_fire_fence_held(temp_home):
    """Busy fence → None immediately, no write; wrong owner still False after release."""
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
    original_at = claimed["fire_claim"]["at"]

    fence_entered = threading.Event()
    release = threading.Event()

    def _hold():
        with fire_claim_fence(job_id, expected_owner=owner) as owns:
            assert owns is True
            fence_entered.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert fence_entered.wait(timeout=5)

    start = time.monotonic()
    result = heartbeat_fire_claim(job_id, expected_owner=owner)
    waited = time.monotonic() - start
    still = get_job(job_id)
    release.set()
    holder.join(timeout=5)

    assert result is None, "busy fence must be unverified, not ownership loss"
    assert waited < 2.0, f"heartbeat blocked on the fire fence ({waited:.2f}s)"
    assert still is not None and still["fire_claim"]["at"] == original_at
    assert heartbeat_fire_claim(job_id, expected_owner=owner) is True
    assert heartbeat_fire_claim(job_id, expected_owner="other-owner") is False


def test_heartbeat_does_not_restore_replaced_owner_while_fence_held(temp_home):
    """Takeover persisted + fence held: heartbeat of the old owner must not rewrite the claim."""
    from cron.jobs import (
        claim_job_for_fire,
        create_job,
        fire_claim_fence,
        get_job,
        heartbeat_fire_claim,
        load_jobs,
        save_jobs,
    )

    job = create_job(prompt="x", schedule="every 5m", name="no-clobber")
    job_id = job["id"]
    assert claim_job_for_fire(job_id) is True
    owner_a = get_job(job_id)["fire_claim"]["by"]
    records = load_jobs()
    records[0]["fire_claim"] = {"at": records[0]["fire_claim"]["at"], "by": "replacement-owner"}
    save_jobs(records)

    fence_entered = threading.Event()
    release = threading.Event()

    def _hold():
        with fire_claim_fence(job_id, expected_owner="replacement-owner") as owns:
            assert owns is True
            fence_entered.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert fence_entered.wait(timeout=5)
    result = heartbeat_fire_claim(job_id, expected_owner=owner_a)
    persisted = get_job(job_id)["fire_claim"]["by"]
    release.set()
    holder.join(timeout=5)

    assert result is None
    assert persisted == "replacement-owner"


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


def _enter_secrets(stack):
    stack.enter_context(patch("agent.secret_scope.set_secret_scope", return_value=None))
    stack.enter_context(patch("agent.secret_scope.build_profile_secret_scope", return_value=None))
    stack.enter_context(patch("agent.secret_scope.reset_secret_scope"))


def test_slow_delivery_holding_fence_is_not_recorded_as_ownership_loss(tmp_path, monkeypatch):
    """A delivery that outlives a heartbeat beat must still be recorded successful."""
    import cron.scheduler as scheduler

    jobs, profile_home, claimed = _claimed_job(tmp_path, "slow-delivery")

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 30.0)

    delivered = threading.Event()

    def _slow_delivery(*_args, **_kwargs):
        delivered.set()
        time.sleep(0.5)
        return None

    _patch_run_scaffold(
        monkeypatch, scheduler,
        run_job=lambda *_a, **_kw: (True, "output", "response", None))
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "output.md")
    monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)

    with jobs.use_cron_store(profile_home), contextlib.ExitStack() as stack:
        _enter_secrets(stack)
        assert scheduler.run_one_job(claimed) is True
        persisted = jobs.get_job(claimed["id"])

    assert delivered.is_set(), "delivery never ran"
    assert persisted is not None
    assert persisted.get("last_status") == "ok", (
        f"slow delivery stamped {persisted.get('last_status')!r} "
        f"error={persisted.get('last_error')!r}")
    assert persisted.get("last_error") is None


def test_fence_held_elsewhere_skips_side_effects_without_ownership_lost_stamp(
    tmp_path, monkeypatch,
):
    """Side effect that cannot acquire the fence is skipped; not an ownership loss.

    Uses real mark_job_run (not a True-stub) so fence-busy on the terminal write is
    not misclassified as owner mismatch.
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
        original_claim = dict(claimed["fire_claim"])

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
        _patch_run_scaffold(monkeypatch, scheduler, run_job=_start_holder_then_finish)
        monkeypatch.setattr(scheduler, "save_job_output", save_output)
        monkeypatch.setattr(scheduler, "_deliver_result", deliver)

        try:
            with contextlib.ExitStack() as stack:
                _enter_secrets(stack)
                assert scheduler.run_one_job(claimed) is True
        finally:
            release_holder.set()
            holder.join(timeout=5)

        persisted = jobs.get_job(job["id"])

    save_output.assert_not_called()
    deliver.assert_not_called()
    assert persisted is not None
    assert persisted.get("last_status") == "error"
    assert "fence" in str(persisted.get("last_error") or "").lower()
    assert "ownership lost" not in str(persisted.get("last_error") or "").lower()
    assert persisted.get("fire_claim", {}).get("by") == original_claim["by"]


def test_delivery_fence_busy_after_output_is_not_recorded_success(tmp_path, monkeypatch):
    """Output fence True, delivery fence None: re-raise busy; do not book success."""
    import cron.scheduler as scheduler

    jobs, profile_home, claimed = _claimed_job(tmp_path, "delivery-busy")
    calls = {"n": 0}

    @contextlib.contextmanager
    def _seq_fence(job_id, *, expected_owner):
        calls["n"] += 1
        yield True if calls["n"] == 1 else None

    monkeypatch.setattr(scheduler, "fire_claim_fence", _seq_fence)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    _patch_run_scaffold(
        monkeypatch, scheduler,
        run_job=lambda *_a, **_kw: (True, "output", "response", None))
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "output.md")
    deliver = MagicMock(return_value=None)
    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    with jobs.use_cron_store(profile_home), contextlib.ExitStack() as stack:
        _enter_secrets(stack)
        assert scheduler.run_one_job(claimed) is True
        persisted = jobs.get_job(claimed["id"])

    deliver.assert_not_called()
    assert persisted is not None
    assert persisted.get("last_status") == "error"
    assert "fence" in str(persisted.get("last_error") or "").lower()
    assert "ownership lost" not in str(persisted.get("last_error") or "").lower()
