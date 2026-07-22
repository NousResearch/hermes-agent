"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
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


def test_stale_claim_is_reclaimable(temp_home, monkeypatch):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    # With a 0s TTL, the existing claim is always considered stale.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_stale_claim_with_unknown_legacy_owner_fails_closed(temp_home):
    """Migrated executions without host identity remain protected."""
    import sqlite3
    import cron.executions as executions
    from cron.executions import create_execution
    from cron.jobs import create_job, claim_job_for_fire, get_job, save_jobs

    job = create_job(prompt="x", schedule="every 5m", name="legacy owner")
    execution = create_execution(job["id"], source="builtin")
    stored_jobs = get_job(job["id"])
    assert stored_jobs is not None
    stored_jobs["fire_claim"] = {
        "at": "2000-01-01T00:00:00+00:00",
        "execution_id": execution["id"],
    }
    save_jobs([stored_jobs])
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute("UPDATE executions SET host_id='' WHERE id=?", (execution["id"],))

    assert claim_job_for_fire(job["id"], claim_ttl_seconds=0) is False


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
    assert claim_job_for_fire(jid) is True


def test_mark_job_run_preserves_replacement_claim(temp_home):
    """A stale completion cannot clear a newer execution's fire claim."""
    from cron.jobs import create_job, get_job, mark_job_run, save_jobs

    job = create_job(prompt="x", schedule="every 5m", name="replacement")
    jid = job["id"]
    jobs = [dict(job, fire_claim={"at": "now", "execution_id": "owner-b"})]
    save_jobs(jobs)

    mark_job_run(jid, success=False, error="stale", execution_id="owner-a")

    assert get_job(jid)["fire_claim"]["execution_id"] == "owner-b"


def test_mark_job_run_preserves_ownerless_claim_for_execution_bound_completion(temp_home):
    """An execution owner cannot clear an unowned legacy replacement claim."""
    from cron.jobs import create_job, get_job, mark_job_run, save_jobs

    job = create_job(prompt="x", schedule="every 5m", name="legacy replacement")
    jid = job["id"]
    save_jobs([dict(job, fire_claim={"at": "now"})])

    mark_job_run(jid, success=False, error="stale", execution_id="owner-a")

    stored = get_job(jid)
    assert stored is not None
    assert stored.get("fire_claim") is not None


def test_release_claim_requires_matching_execution_owner(temp_home):
    from cron.jobs import create_job, claim_job_for_fire, get_job, release_fire_claim

    job = create_job(prompt="x", schedule="every 5m", name="owned")
    jid = job["id"]
    assert claim_job_for_fire(jid, execution_id="owner-a") is True
    assert release_fire_claim(jid, execution_id="owner-b") is False
    assert get_job(jid).get("fire_claim") is not None
    assert release_fire_claim(jid, execution_id="owner-a") is True
    assert get_job(jid).get("fire_claim") is None


def test_release_legacy_claim_without_owner_uses_compatibility_path(temp_home):
    from cron.jobs import create_job, claim_job_for_fire, get_job, release_fire_claim

    job = create_job(prompt="x", schedule="every 5m", name="legacy")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    assert release_fire_claim(jid) is True
    assert get_job(jid).get("fire_claim") is None


def test_ownerless_release_preserves_execution_owned_claim(temp_home):
    """A stale ownerless cleanup cannot release a live execution claim."""
    from cron.jobs import create_job, claim_job_for_fire, get_job, release_fire_claim

    job = create_job(prompt="x", schedule="every 5m", name="owned legacy cleanup")
    jid = job["id"]
    assert claim_job_for_fire(jid, execution_id="owner-a") is True
    assert release_fire_claim(jid) is False
    stored = get_job(jid)
    assert stored is not None
    assert stored.get("fire_claim") is not None
    assert stored["fire_claim"]["execution_id"] == "owner-a"
