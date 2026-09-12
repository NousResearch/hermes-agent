"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import threading
import time
from pathlib import Path

import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # cron.jobs caches no home at import; get_hermes_home() reads the env live.
    yield tmp_path


def test_claim_succeeds_once_then_blocks(temp_home):
    """First claim for a fire wins; a second claim for the same fire loses, and
    next_run_at is advanced (a re-delivery for the old time can't re-fire)."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="t")
    jid = job["id"]
    before = get_job(jid)["next_run_at"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid) is False
    assert get_job(jid)["next_run_at"] != before


def test_claim_oneshot_cannot_be_double_claimed(temp_home):
    """A one-shot can't be double-claimed (the fresh claim blocks the retry)."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="in 30m", name="o")
    assert claim_job_for_fire(job["id"]) is True
    assert claim_job_for_fire(job["id"]) is False


def test_claim_unknown_job_returns_false(temp_home):
    from cron.jobs import claim_job_for_fire

    assert claim_job_for_fire("nope-does-not-exist") is False


def test_claim_paused_job_returns_false(temp_home):
    """A paused job can't be claimed."""
    from cron.jobs import create_job, claim_job_for_fire, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="p")
    pause_job(job["id"])
    assert claim_job_for_fire(job["id"]) is False


def test_forced_claim_atomically_resumes_paused_job(temp_home):
    """Explicit manual fire may resume a paused job without exposing a due
    intermediate state to the ticker."""
    from cron.jobs import create_job, claim_job_for_fire, get_job, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="manual")
    pause_job(job["id"])

    assert claim_job_for_fire(job["id"], force=True) is True
    claimed = get_job(job["id"])
    assert claimed["enabled"] is True
    assert claimed["state"] == "scheduled"
    assert claimed["paused_at"] is None
    assert claimed["paused_reason"] is None
    assert claimed["fire_claim"] is not None


def test_stale_claim_is_reclaimable(temp_home, monkeypatch):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    # With a 0s TTL, the existing claim is always considered stale.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_mark_job_run_clears_claim(temp_home):
    """After a recurring job completes, its claim is cleared so the next fire
    can be claimed again."""
    from cron.jobs import create_job, claim_job_for_fire, mark_job_run, get_job

    job = create_job(prompt="x", schedule="every 5m", name="c")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    assert get_job(jid).get("fire_claim") is not None

    mark_job_run(jid, success=True)
    assert get_job(jid).get("fire_claim") is None
    # …and the re-armed recurring job is claimable again.
    assert claim_job_for_fire(jid) is True


def test_fire_claim_heartbeat_refreshes_only_expected_owner(temp_home, monkeypatch):
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="heartbeat")
    assert jobs.claim_job_for_fire(job["id"]) is True
    claimed = jobs.get_job(job["id"])["fire_claim"]
    claimed_at = datetime.fromisoformat(claimed["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: claimed_at + timedelta(seconds=30),
    )

    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner=claimed["by"],
    ) is True
    refreshed = jobs.get_job(job["id"])["fire_claim"]
    assert refreshed["at"] != claimed["at"]
    assert refreshed["by"] == claimed["by"]
    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner="replacement-owner",
    ) is False


def test_reclaimed_fire_uses_new_owner_token(temp_home, monkeypatch):
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="reclaim")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    original_at = datetime.fromisoformat(original["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: original_at + timedelta(seconds=301),
    )

    assert jobs.claim_job_for_fire(job["id"]) is True
    replacement = dict(jobs.get_job(job["id"])["fire_claim"])
    assert replacement["by"] != original["by"]
    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner=original["by"],
    ) is False
    assert jobs.get_job(job["id"])["fire_claim"] == replacement


def test_stale_fire_owner_cannot_mark_replacement_run(temp_home):
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="fenced")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    records = jobs.load_jobs()
    records[0]["fire_claim"] = {"at": original["at"], "by": "replacement"}
    jobs.save_jobs(records)

    assert jobs.mark_job_run(
        job["id"],
        success=True,
        expected_fire_owner=original["by"],
    ) is False
    persisted = jobs.get_job(job["id"])
    assert persisted["fire_claim"]["by"] == "replacement"
    assert persisted.get("last_run_at") is None


def test_fire_claim_fence_serializes_terminal_revocation(temp_home):
    """A side effect authorized by owner linearizes before terminal revocation."""
    from cron.jobs import (
        claim_job_for_fire,
        create_job,
        fire_claim_fence,
        mark_job_run,
    )

    job = create_job(prompt="x", schedule="every 5m", name="fenced-side-effect")
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    terminal_done = threading.Event()

    def finish_run():
        mark_job_run(job["id"], True, expected_fire_owner=owner)
        terminal_done.set()

    with fire_claim_fence(job["id"], expected_owner=owner) as owns_claim:
        assert owns_claim is True
        thread = threading.Thread(target=finish_run)
        thread.start()
        time.sleep(0.05)
        assert terminal_done.is_set() is False

    thread.join(timeout=1)
    assert terminal_done.is_set() is True


def test_fire_claim_fence_rejects_stale_owner(temp_home):
    from cron.jobs import claim_job_for_fire, create_job, fire_claim_fence

    job = create_job(prompt="x", schedule="every 5m", name="stale-fence")
    claim_job_for_fire(job["id"])

    with fire_claim_fence(job["id"], expected_owner="stale") as owns_claim:
        assert owns_claim is False


def test_same_process_fire_fence_refuses_second_claim_after_timeout(temp_home, monkeypatch):
    """A wedged local holder must not indefinitely block another claimant."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="local-fence-timeout")
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.1)
    completed = threading.Event()
    result = {}

    def second_claimant():
        result["claimed"] = jobs.claim_job_for_fire(job["id"])
        completed.set()

    with jobs._fire_job_lock(job["id"]) as acquired:
        assert acquired is True
        thread = threading.Thread(target=second_claimant)
        thread.start()
        assert completed.wait(timeout=2), "same-process claimant waited past the fire-fence timeout"
        assert result["claimed"] is False

    thread.join(timeout=2)
    assert thread.is_alive() is False
    assert jobs.claim_job_for_fire(job["id"]) is True


def test_same_thread_fire_fence_reentrancy_preserves_ownership(temp_home):
    """Nested same-thread callers retain the existing fire fence."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="local-fence-reentrant")
    completed = threading.Event()
    result = {}

    def reentrant_claimant():
        with jobs._fire_job_lock(job["id"]) as outer_acquired:
            result["outer"] = outer_acquired
            with jobs._fire_job_lock(job["id"]) as inner_acquired:
                result["inner"] = inner_acquired
        completed.set()

    thread = threading.Thread(target=reentrant_claimant, daemon=True)
    thread.start()
    assert completed.wait(timeout=2), "same-thread nested fire fence did not return"
    assert result == {"outer": True, "inner": True}
    thread.join(timeout=2)
    assert thread.is_alive() is False


def test_manual_claim_does_not_stamp_a_future_occurrence(temp_home):
    """An off-tick run-now must not consume the NEXT scheduled slot.

    Outside a scheduler tick ``next_run_at`` is the occurrence that has NOT happened
    yet, so stamping it as a completed occurrence makes ``_job_is_due`` skip that slot
    when it arrives — silently, with no error and no dispatch record. ``manual=True``
    is the caller's declaration that this is an off-tick fire.
    """
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="m")
    pending = get_job(job["id"])["next_run_at"]

    claimed = claim_job_for_fire(job["id"], manual=True, return_job=True)
    assert isinstance(claimed, dict)
    assert claimed["_scheduled_instant"] is None, (
        f"manual fire stamped the future occurrence {pending}")


def test_manual_claim_still_refuses_a_paused_job(temp_home):
    """``manual=True`` suppresses only the occurrence stamp — unlike ``force=True`` it
    must not resume a paused job, which the run-now tool relies on to refuse it."""
    from cron.jobs import create_job, claim_job_for_fire, get_job, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="mp")
    pause_job(job["id"])

    assert claim_job_for_fire(job["id"], manual=True) is False
    assert get_job(job["id"]).get("paused_at") is not None


def test_fresh_claim_from_a_dead_same_host_owner_is_reclaimable(temp_home):
    """A claim younger than the TTL whose owner pid (same host) has exited is stale at once: a
    ``hermes cron run`` killed mid-flight must not block the next manual run for the whole TTL
    with "already being fired". A live owner's fresh claim still blocks."""
    import os
    import socket
    import subprocess
    import sys

    from cron.jobs import claim_job_for_fire, create_job, load_jobs, save_jobs

    jid = create_job(prompt="x", schedule="every 5m", name="s")["id"]
    assert claim_job_for_fire(jid) is True

    # Live same-host owner (this process) → still blocked.
    jobs = load_jobs()
    job = next(j for j in jobs if j["id"] == jid)
    job["fire_claim"]["by"] = f"{socket.gethostname()}:{os.getpid()}:tok"
    save_jobs(jobs)
    assert claim_job_for_fire(jid) is False

    # Owner that has provably exited → reclaimable despite the fresh timestamp.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    jobs = load_jobs()
    job = next(j for j in jobs if j["id"] == jid)
    job["fire_claim"]["by"] = f"{socket.gethostname()}:{child.pid}:tok"
    save_jobs(jobs)
    assert claim_job_for_fire(jid) is True


def _hold_jobs_flock(path: Path, release: threading.Event, held: threading.Event):
    """Hold an exclusive flock on *path* from a separate fd until released.

    flock locks are per-open-file-description, so a second open() in the SAME
    process contends exactly like another process would.
    """
    fd = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        held.set()
        release.wait(timeout=30)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()


def test_heartbeat_fire_claim_missing_job_returns_false(temp_home):
    import cron.jobs as jobs

    assert jobs.heartbeat_fire_claim("nope-does-not-exist", expected_owner="anyone") is False


@pytest.mark.skipif(fcntl is None, reason="flock semantics are POSIX-only")
def test_heartbeat_does_not_pin_fire_fence_while_jobs_lock_contended(temp_home, monkeypatch):
    """Correct-owner heartbeat must not hold fire_fence across a .jobs.lock wait.

    Contended path: another thread holds ``.jobs.lock`` while the owner calls
    ``heartbeat_fire_claim``; concurrently ``mark_job_run`` needs the fire fence.

    PRE-FIX: heartbeat wraps ``_with_job`` in ``_under_fire_fence``, so the
    flock wait keeps fire_fence held and ``mark_job_run`` fails closed (False).
    POST-FIX: heartbeat only CAS-refreshes via ``_with_job``; mark can take the
    fence and succeed (or at least is not fence-blocked by the heartbeat).
    """
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="fence-hostage")
    jid = job["id"]
    assert jobs.claim_job_for_fire(jid) is True
    claimed = jobs.get_job(jid)["fire_claim"]
    owner = claimed["by"]
    claimed_at = datetime.fromisoformat(claimed["at"])
    monkeypatch.setattr(jobs, "_hermes_now", lambda: claimed_at + timedelta(seconds=30))
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.8)

    lock_path = jobs._jobs_lock_file()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    release = threading.Event()
    held = threading.Event()
    heartbeat_waiting_on_jobs_lock = threading.Event()
    results = {}
    real_acquire = jobs._acquire_flock
    real_refresh = jobs._refresh_claim

    def wrapped_acquire(lock_fd, timeout):
        name = Path(getattr(lock_fd, "name", "") or "").name
        if name == ".jobs.lock":
            heartbeat_waiting_on_jobs_lock.set()
        return real_acquire(lock_fd, timeout)

    def wrapped_refresh(job_list, claim, expected_owner):
        ok = real_refresh(job_list, claim, expected_owner)
        if ok:
            results["refreshed_at"] = claim.get("at")
            results["refreshed_by"] = claim.get("by")
        return ok

    monkeypatch.setattr(jobs, "_acquire_flock", wrapped_acquire)
    monkeypatch.setattr(jobs, "_refresh_claim", wrapped_refresh)

    holder = threading.Thread(
        target=_hold_jobs_flock, args=(lock_path, release, held), daemon=True,
    )
    holder.start()
    assert held.wait(timeout=5), "test holder failed to take .jobs.lock"

    def run_heartbeat():
        results["heartbeat"] = jobs.heartbeat_fire_claim(jid, expected_owner=owner)

    def run_mark():
        results["mark"] = jobs.mark_job_run(jid, True, expected_fire_owner=owner)

    try:
        hb_thread = threading.Thread(target=run_heartbeat)
        hb_thread.start()
        assert heartbeat_waiting_on_jobs_lock.wait(timeout=5), (
            "heartbeat never reached the .jobs.lock wait inside _with_job/save_jobs"
        )

        mark_thread = threading.Thread(target=run_mark)
        mark_thread.start()
        mark_thread.join(timeout=8)
        hb_thread.join(timeout=8)
        assert mark_thread.is_alive() is False
        assert hb_thread.is_alive() is False
    finally:
        release.set()
        holder.join(timeout=5)

    assert results.get("heartbeat") is True
    assert results.get("refreshed_by") == owner
    assert results.get("refreshed_at") != claimed["at"]
    assert results.get("mark") is True, (
        "mark_job_run failed closed while heartbeat held fire_fence across the "
        ".jobs.lock wait (heartbeat must not wrap _with_job in _under_fire_fence)"
    )
