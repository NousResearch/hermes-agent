"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pytest


def _claim_token_in_process(home: str, job_id: str):
    """Claim from a fresh interpreter against one shared profile store."""
    from cron.jobs import claim_job_for_fire_token, use_cron_store

    with use_cron_store(home):
        return claim_job_for_fire_token(job_id)


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


def test_owned_fire_claim_heartbeat_is_token_fenced(temp_home):
    """Only the exact execution token may keep a live fire claim fresh."""
    from cron.jobs import (
        claim_job_for_fire_token,
        create_job,
        get_job,
        heartbeat_fire_claim,
    )

    job = create_job(prompt="x", schedule="every 5m", name="owned-heartbeat")
    first = claim_job_for_fire_token(job["id"])

    assert first
    assert get_job(job["id"])["fire_claim"]["id"] == first
    assert heartbeat_fire_claim(job["id"], expected_claim_id=first) is True

    second = claim_job_for_fire_token(job["id"], claim_ttl_seconds=0)
    assert second and second != first
    assert heartbeat_fire_claim(job["id"], expected_claim_id=first) is False
    assert get_job(job["id"])["fire_claim"]["id"] == second


def test_stale_fire_owner_cannot_finalize_new_owner_state(temp_home):
    """A reclaimed runner's late completion must not mutate shared job state."""
    from cron.jobs import claim_job_for_fire_token, create_job, get_job, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="owned-finalize")
    first = claim_job_for_fire_token(job["id"])
    second = claim_job_for_fire_token(job["id"], claim_ttl_seconds=0)
    assert first and second and first != second

    assert mark_job_run(
        job["id"], success=True, expected_fire_claim_id=first
    ) is False
    after_stale = get_job(job["id"])
    assert after_stale["fire_claim"]["id"] == second
    assert after_stale.get("last_run_at") is None

    assert mark_job_run(
        job["id"], success=True, expected_fire_claim_id=second
    ) is True
    assert get_job(job["id"])["fire_claim"] is None


def test_only_current_owner_can_release_fire_claim(temp_home):
    """Cleanup from a stale failed dispatch cannot clear a replacement owner."""
    from cron.jobs import (
        claim_job_for_fire_token,
        create_job,
        get_job,
        release_fire_claim,
    )

    job = create_job(prompt="x", schedule="every 5m", name="owned-release")
    first = claim_job_for_fire_token(job["id"])
    second = claim_job_for_fire_token(job["id"], claim_ttl_seconds=0)
    assert first and second and first != second

    assert release_fire_claim(job["id"], expected_claim_id=first) is False
    assert get_job(job["id"])["fire_claim"]["id"] == second
    assert release_fire_claim(job["id"], expected_claim_id=second) is True
    assert get_job(job["id"])["fire_claim"] is None


def test_cross_process_fire_claim_has_exactly_one_winner(temp_home):
    """Concurrent gateway processes cannot both own the same fire."""
    from cron.jobs import create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="process-race")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        tokens = list(
            pool.map(
                _claim_token_in_process,
                [str(temp_home)] * 4,
                [job["id"]] * 4,
            )
        )

    winners = [token for token in tokens if token]
    assert len(winners) == 1
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["fire_claim"]["id"] == winners[0]
