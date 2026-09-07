"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import threading
import time

import pytest


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


@pytest.mark.parametrize("null_next_run", [False, True], ids=["non-null", "null"])
@pytest.mark.parametrize("success", [True, False], ids=["success", "failure"])
def test_preserve_paused_claim_and_terminal_mark_keep_exact_next_run(
    temp_home, null_next_run, success
):
    """A one-off paused recurring run must not re-arm or perturb its paused schedule."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="paused-preserve")
    jobs.pause_job(job["id"])
    if null_next_run:
        records = jobs.load_jobs()
        records[0]["next_run_at"] = None
        jobs.save_jobs(records)
    before = jobs.get_job(job["id"])
    assert before is not None
    expected_next_run = before.get("next_run_at")

    claimed = jobs.claim_job_for_fire(
        job["id"], force=True, preserve_paused=True, return_job=True
    )
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    persisted_claim = jobs.get_job(job["id"])
    assert persisted_claim is not None
    assert persisted_claim["next_run_at"] == expected_next_run

    assert jobs.mark_job_run(
        job["id"], success=success, expected_fire_owner=owner
    ) is True
    after = jobs.get_job(job["id"])
    assert after is not None
    assert after["next_run_at"] == expected_next_run
    assert after["state"] == "paused"
    assert after["enabled"] is False
    assert after["last_status"] == ("ok" if success else "error")
    assert after["repeat"]["completed"] == 1


def test_preserve_paused_one_shot_still_terminalizes_after_manual_run(temp_home):
    """Pause preservation must not keep a consumed finite one-shot runnable."""
    import cron.jobs as jobs

    job = jobs.create_job(
        prompt="x",
        schedule="in 30m",
        name="paused-once",
        repeat=1,
    )
    jobs.pause_job(job["id"])

    claimed = jobs.claim_job_for_fire(
        job["id"], force=True, preserve_paused=True, return_job=True
    )
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    assert jobs.mark_job_run(
        job["id"], success=True, expected_fire_owner=owner
    ) is True

    after = jobs.get_job(job["id"])
    assert after is not None
    assert after["state"] == "completed"
    assert after["enabled"] is False
    assert after["next_run_at"] is None
    assert after["fire_claim"] is None
    assert after["repeat"]["completed"] == 1


def test_finite_paused_recurring_manual_run_preserves_pause_at_repeat_limit(temp_home):
    """A manual occurrence must not retire a paused recurring schedule."""
    import cron.jobs as jobs

    job = jobs.create_job(
        prompt="x", schedule="every 5m", name="paused-finite", repeat=1
    )
    jobs.pause_job(job["id"])
    before = jobs.get_job(job["id"])
    assert before is not None

    claimed = jobs.claim_job_for_fire(
        job["id"], force=True, preserve_paused=True, return_job=True
    )
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    assert jobs.mark_job_run(
        job["id"], success=True, expected_fire_owner=owner
    ) is True

    after = jobs.get_job(job["id"])
    assert after is not None
    for key in ("enabled", "state", "paused_at", "next_run_at"):
        assert after[key] == before[key]
    assert after["repeat"] == {"times": 1, "completed": 1}
    assert jobs.claim_job_for_fire(
        job["id"], force=True, preserve_paused=True, return_job=True
    ) is False


def test_release_fire_claim_is_owner_fenced_and_state_neutral(temp_home):
    """Abort may clear only its own fire claim without recording an occurrence."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="release")
    jobs.pause_job(job["id"])
    assert jobs.claim_job_for_fire(job["id"], force=True, preserve_paused=True) is True
    claimed = jobs.get_job(job["id"])
    assert claimed is not None
    owner = claimed["fire_claim"]["by"]

    assert jobs.release_fire_claim(job["id"], expected_owner="other-owner") is False
    still_claimed = jobs.get_job(job["id"])
    assert still_claimed is not None
    assert still_claimed["fire_claim"]["by"] == owner
    assert jobs.release_fire_claim(job["id"], expected_owner=owner) is True
    released = jobs.get_job(job["id"])
    assert released is not None
    assert released["fire_claim"] is None
    assert released["state"] == "paused"
    assert released["enabled"] is False
    assert released["repeat"]["completed"] == 0


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
