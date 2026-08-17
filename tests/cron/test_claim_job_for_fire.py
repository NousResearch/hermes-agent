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

    job = create_job(prompt="x", schedule="30m", name="o")
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
