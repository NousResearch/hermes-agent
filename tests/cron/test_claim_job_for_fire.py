"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives built-in, manual, and external scheduler fires one
shared at-most-once boundary. Across N processes or gateway replicas, exactly
one caller owns a given fire until completion or safe recovery.

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated cron store so jobs.json doesn't touch the real store."""
    import cron.scheduler as scheduler
    from cron.jobs import _local_fire_claim_machine_fingerprint

    _local_fire_claim_machine_fingerprint.cache_clear()
    with scheduler._running_lock:
        scheduler._running_job_ids.clear()
        scheduler._running_job_claim_tokens.clear()
        scheduler._running_job_claim_snapshots.clear()
        scheduler._running_job_started_tokens.clear()
        scheduler._interrupted_job_ids.clear()
        scheduler._shutdown_started = False
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    try:
        yield tmp_path
    finally:
        _local_fire_claim_machine_fingerprint.cache_clear()
        with scheduler._running_lock:
            scheduler._running_job_ids.clear()
            scheduler._running_job_claim_tokens.clear()
            scheduler._running_job_claim_snapshots.clear()
            scheduler._running_job_started_tokens.clear()
            scheduler._interrupted_job_ids.clear()
            scheduler._shutdown_started = False


def _rewrite_claim(job_id, **changes):
    from cron.jobs import load_jobs, save_jobs

    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job["fire_claim"].update(changes)
    save_jobs(jobs)


def _rewrite_job(job_id, **changes):
    from cron.jobs import load_jobs, save_jobs

    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job.update(changes)
    save_jobs(jobs)


def _delete_claim_fields(job_id, *fields):
    from cron.jobs import load_jobs, save_jobs

    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            for field in fields:
                job["fire_claim"].pop(field, None)
    save_jobs(jobs)


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


def test_claim_persists_unambiguous_local_holder_provenance(temp_home):
    """Default claims identify the exact local machine and process separately."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="provenance")
    assert claim_job_for_fire(job["id"]) is True

    stored = get_job(job["id"])
    assert stored is not None
    holder = stored["fire_claim"]["holder"]
    assert holder["kind"] == "local-process-v1"
    assert holder["machine"]
    assert holder["pid"] == os.getpid()


def test_machine_fingerprint_ignores_optional_dependency_availability(temp_home):
    from cron.jobs import _local_fire_claim_machine_fingerprint

    _local_fire_claim_machine_fingerprint.cache_clear()
    parent_fingerprint = _local_fire_claim_machine_fingerprint()
    fake_dependencies = temp_home / "fake-dependencies"
    fake_dependencies.mkdir()
    (fake_dependencies / "psutil.py").write_text(
        "def boot_time():\n    return 123.456\n",
        encoding="utf-8",
    )
    probe = (
        "from cron.jobs import _local_fire_claim_machine_fingerprint; "
        "print(_local_fire_claim_machine_fingerprint())"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=os.getcwd(),
        env={
            **os.environ,
            "HERMES_HOME": str(temp_home),
            "PYTHONPATH": str(fake_dependencies),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert result.stdout.strip() == parent_fingerprint


def test_machine_fingerprint_ignores_random_node_when_proc_identity_exists(
    temp_home, monkeypatch
):
    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod.uuid, "getnode", lambda: 0x010000000001)
    jobs_mod._local_fire_claim_machine_fingerprint.cache_clear()
    first = jobs_mod._local_fire_claim_machine_fingerprint()

    monkeypatch.setattr(jobs_mod.uuid, "getnode", lambda: 0x030000000002)
    jobs_mod._local_fire_claim_machine_fingerprint.cache_clear()
    second = jobs_mod._local_fire_claim_machine_fingerprint()

    assert first
    assert second == first


def test_machine_fingerprint_rejects_random_node_as_only_identity(
    temp_home, monkeypatch
):
    import cron.jobs as jobs_mod

    def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(jobs_mod.Path, "read_text", unavailable)
    monkeypatch.setattr(jobs_mod.uuid, "getnode", lambda: 0x010000000001)
    jobs_mod._local_fire_claim_machine_fingerprint.cache_clear()

    assert jobs_mod._local_fire_claim_machine_fingerprint() == ""


def test_machine_fingerprint_rejects_partial_proc_identity(temp_home, monkeypatch):
    """A boot ID without PID-namespace identity is not safe for PID liveness."""
    import cron.jobs as jobs_mod

    def partial_proc_identity(path, *_args, **_kwargs):
        if str(path).endswith("/boot_id"):
            return "shared-host-boot-id"
        raise OSError("pid namespace identity unavailable")

    monkeypatch.setattr(jobs_mod.Path, "read_text", partial_proc_identity)
    monkeypatch.setattr(jobs_mod.uuid, "getnode", lambda: 0x010000000001)
    jobs_mod._local_fire_claim_machine_fingerprint.cache_clear()

    assert jobs_mod._local_fire_claim_machine_fingerprint() == ""


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


def test_stale_claim_of_dead_holder_is_reclaimable(temp_home):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import (
        _local_fire_claim_machine_fingerprint,
        claim_job_for_fire,
        create_job,
    )
    from gateway.status import _pid_exists

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = exited.pid
    assert exited.wait(timeout=10) == 0
    assert _pid_exists(dead_pid) is False
    _rewrite_claim(
        jid,
        by=f"{socket.gethostname()}:{dead_pid}",
        holder={
            "kind": "local-process-v1",
            "machine": _local_fire_claim_machine_fingerprint(),
            "pid": dead_pid,
        },
    )
    # With a 0s TTL and a dead holder, the claim is stale and reclaimable.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_live_holder_keeps_claim_past_freshness_ttl(temp_home):
    """A long-running job must not lose its claim merely because its TTL elapsed."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="live")
    jid = job["id"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is False


def test_same_hostname_remote_holder_uses_ttl_only(temp_home):
    """Remote provenance wins even when hostname and PID collide locally."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="remote-collision")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    _rewrite_claim(
        jid,
        by=f"{socket.gethostname()}:{os.getpid()}",
        holder={
            "kind": "local-process-v1",
            "machine": "remote-machine",
            "pid": os.getpid(),
        },
    )

    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_legacy_ambiguous_holder_uses_ttl_only(temp_home):
    """Pre-provenance host:pid claims remain readable but are never probed."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="legacy")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    _delete_claim_fields(jid, "holder")

    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_live_holder_blocks_competing_process_after_ttl(temp_home):
    """A ticker process cannot reclaim a still-running manual fire."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="race")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True

    probe = (
        "from cron.jobs import claim_job_for_fire; "
        f"raise SystemExit(1 if claim_job_for_fire({jid!r}, claim_ttl_seconds=0) else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=os.getcwd(),
        env={**os.environ, "HERMES_HOME": str(temp_home)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_builtin_tick_does_not_dispatch_past_live_manual_claim(temp_home, monkeypatch):
    """The built-in ticker must arbitrate through the same claim as manual run."""
    from cron.jobs import claim_job_for_fire, create_job
    from cron.scheduler import tick

    job = create_job(prompt="x", schedule="every 5m", name="manual-vs-tick")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    _rewrite_claim(
        jid,
        at=(datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat(),
    )
    _rewrite_job(
        jid,
        next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )

    ran = []
    monkeypatch.setattr(
        "cron.scheduler.run_one_job",
        lambda due_job, **kwargs: ran.append(due_job["id"]) or True,
    )

    assert tick(verbose=False, sync=True) == 0
    assert ran == []


def test_finite_oneshot_fire_claim_prevents_stale_cleanup(temp_home):
    """A manual/external one-shot fire must keep its record until completion."""
    from cron.jobs import (
        claim_dispatch,
        claim_job_for_fire,
        create_job,
        get_due_jobs,
        get_job,
    )

    job = create_job(prompt="x", schedule="30m", name="claimed-once", repeat=1)
    jid = job["id"]
    _rewrite_job(
        jid,
        next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    assert claim_job_for_fire(jid) is True
    assert claim_dispatch(jid) is True

    assert get_due_jobs() == []
    stored = get_job(jid)
    assert stored is not None
    assert stored["repeat"]["completed"] == 1
    assert stored["fire_claim"]["token"]


def test_tick_aborts_claim_when_shutdown_starts_during_claim(temp_home, monkeypatch):
    """A shutdown racing claim acquisition must prevent post-shutdown dispatch."""
    import cron.scheduler as scheduler
    from cron.jobs import claim_job_for_fire_snapshot, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="shutdown-race")
    second = create_job(
        prompt="y",
        schedule="30m",
        name="after-shutdown",
        repeat=1,
    )
    jid = job["id"]
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    _rewrite_job(
        jid,
        next_run_at=due_at,
    )
    _rewrite_job(second["id"], next_run_at=due_at)
    original_claim = claim_job_for_fire_snapshot
    ran = []
    claim_calls = []
    shutdown_started = False

    def claim_during_shutdown(job_id, **kwargs):
        nonlocal shutdown_started
        claim_calls.append(job_id)
        if not shutdown_started:
            shutdown_started = True
            scheduler.mark_running_jobs_interrupted("shutdown during claim")
        return original_claim(job_id, **kwargs)

    monkeypatch.setattr(
        scheduler,
        "claim_job_for_fire_snapshot",
        claim_during_shutdown,
    )
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda claimed_job, **kwargs: ran.append(claimed_job["id"]) or True,
    )

    assert scheduler.tick(verbose=False, sync=True) == 0
    assert ran == []
    assert claim_calls == [jid]
    stored = get_job(jid)
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored["next_run_at"] == due_at
    second_stored = get_job(second["id"])
    assert second_stored is not None
    assert second_stored.get("fire_claim") is None
    assert second_stored.get("run_claim") is None
    assert second_stored["next_run_at"] == due_at


def test_tick_never_enters_body_when_shutdown_starts_after_registration(
    temp_home,
    monkeypatch,
):
    """A registered-but-unstarted pool job is released before side effects."""
    import cron.scheduler as scheduler
    from cron.jobs import create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="registered-tick")
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    _rewrite_job(job["id"], next_run_at=due_at)
    original_register = scheduler._register_running_claim
    body_entered = []

    def register_then_shutdown(job_id, claim_token, claimed_job=None):
        result = original_register(job_id, claim_token, claimed_job)
        if result == "published":
            scheduler.mark_running_jobs_interrupted("shutdown after registration")
        return result

    monkeypatch.setattr(scheduler, "_register_running_claim", register_then_shutdown)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda claimed_job, **kwargs: body_entered.append(claimed_job["id"]) or True,
    )

    assert scheduler.tick(verbose=False, sync=True) == 0
    assert body_entered == []
    stored = get_job(job["id"])
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored["next_run_at"] == due_at


def test_manual_run_never_enters_body_when_shutdown_starts_after_registration(
    temp_home,
    monkeypatch,
):
    """Manual/provider wrapper has the same registration-to-body barrier."""
    import cron.scheduler as scheduler
    from cron.jobs import claim_job_for_fire_snapshot, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="registered-manual")
    original = get_job(job["id"])
    assert original is not None
    before = original["next_run_at"]
    claimed = claim_job_for_fire_snapshot(job["id"])
    assert claimed is not None
    original_register = scheduler._register_running_claim
    body_entered = []

    def register_then_shutdown(job_id, claim_token, claimed_job=None):
        result = original_register(job_id, claim_token, claimed_job)
        if result == "published":
            scheduler.mark_running_jobs_interrupted("shutdown after registration")
        return result

    monkeypatch.setattr(scheduler, "_register_running_claim", register_then_shutdown)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda claimed_job, **kwargs: body_entered.append(claimed_job["id"]) or True,
    )

    assert scheduler.run_claimed_job(claimed, verbose=False) is False
    assert body_entered == []
    stored = get_job(job["id"])
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored["next_run_at"] == before


def test_builtin_tick_executes_with_persisted_claim_token(temp_home, monkeypatch):
    from cron.jobs import create_job, fire_claim_token, get_job, mark_job_run
    from cron.scheduler import tick

    job = create_job(prompt="x", schedule="every 5m", name="tick-token")
    _rewrite_job(
        job["id"],
        next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    seen_tokens = []

    def run_claimed(claimed_job, **kwargs):
        token = fire_claim_token(claimed_job)
        assert token is not None
        seen_tokens.append(token)
        assert mark_job_run(
            claimed_job["id"],
            success=True,
            fire_claim_token=token,
        )
        return True

    monkeypatch.setattr("cron.scheduler.run_one_job", run_claimed)

    assert tick(verbose=False, sync=True) == 1
    assert len(seen_tokens) == 1
    completed = get_job(job["id"])
    assert completed is not None
    assert completed.get("fire_claim") is None


def test_manual_run_executes_with_persisted_claim_token(temp_home, monkeypatch):
    from cron.jobs import create_job, fire_claim_token, mark_job_run
    from tools.cronjob_tools import _execute_job_now

    job = create_job(prompt="x", schedule="every 5m", name="manual-token")
    seen_tokens = []

    def run_claimed(claimed_job, **kwargs):
        token = fire_claim_token(claimed_job)
        assert token is not None
        seen_tokens.append(token)
        assert mark_job_run(
            claimed_job["id"],
            success=True,
            fire_claim_token=token,
        )
        return True

    monkeypatch.setattr("cron.scheduler.run_one_job", run_claimed)

    result = _execute_job_now(job)
    assert result["claimed"] is True
    assert result["success"] is True
    assert len(seen_tokens) == 1


def test_builtin_tick_does_not_reread_after_atomic_claim(temp_home, monkeypatch):
    """A won claim returns its token snapshot without a fallible second read."""
    import cron.scheduler as scheduler
    from cron.jobs import create_job, fire_claim_token, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="atomic-tick-snapshot")
    _rewrite_job(
        job["id"],
        next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )

    def forbidden_reread(_job_id):
        raise OSError("post-claim read failed")

    def run_claimed(claimed_job, **_kwargs):
        token = fire_claim_token(claimed_job)
        assert token is not None
        assert mark_job_run(
            claimed_job["id"],
            success=True,
            fire_claim_token=token,
        )
        return True

    monkeypatch.setattr(scheduler, "get_job", forbidden_reread, raising=False)
    monkeypatch.setattr(scheduler, "run_one_job", run_claimed)

    assert scheduler.tick(verbose=False, sync=True) == 1


def test_manual_run_does_not_reread_before_execution(temp_home, monkeypatch):
    """Manual execution receives the atomic claim snapshot before store rereads."""
    import tools.cronjob_tools as cron_tools
    from cron.jobs import create_job, fire_claim_token, get_job, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="atomic-manual-snapshot")
    ran = False

    def get_only_after_run(job_id):
        if not ran:
            raise OSError("post-claim read failed")
        return get_job(job_id)

    def run_claimed(claimed_job, **_kwargs):
        nonlocal ran
        import cron.scheduler as scheduler

        token = fire_claim_token(claimed_job)
        assert token is not None
        assert scheduler.get_running_job_ids() == frozenset({claimed_job["id"]})
        assert scheduler._running_job_claim_tokens[claimed_job["id"]] == token
        ran = True
        assert mark_job_run(
            claimed_job["id"],
            success=True,
            fire_claim_token=token,
        )
        return True

    monkeypatch.setattr(cron_tools, "get_job", get_only_after_run)
    monkeypatch.setattr("cron.scheduler.run_one_job", run_claimed)

    result = cron_tools._execute_job_now(job)
    assert result["claimed"] is True
    assert result["success"] is True
    assert ran is True


def test_tick_releases_its_claim_when_submission_is_rejected(temp_home, monkeypatch):
    from cron.jobs import create_job, get_job
    from cron.scheduler import tick

    class RejectingPool:
        def submit(self, callback):
            raise RuntimeError("cannot schedule new futures after shutdown")

    job = create_job(prompt="x", schedule="every 5m", name="reject-submit")
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    _rewrite_job(
        job["id"],
        next_run_at=due_at,
    )
    monkeypatch.setattr("cron.scheduler._get_parallel_pool", lambda workers: RejectingPool())
    monkeypatch.setattr(
        "cron.scheduler._interpreter_shutting_down",
        lambda error=None: error is not None,
    )

    assert tick(verbose=False, sync=True) == 0
    stored = get_job(job["id"])
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored["next_run_at"] == due_at


def test_rejected_oneshot_submission_clears_run_and_fire_claims(temp_home, monkeypatch):
    """A one-shot rejected before execution remains immediately due."""
    from cron.jobs import create_job, get_due_jobs, get_job
    from cron.scheduler import tick

    class RejectingPool:
        def submit(self, callback):
            raise RuntimeError("cannot schedule new futures after shutdown")

    job = create_job(prompt="x", schedule="30m", name="reject-once", repeat=1)
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    _rewrite_job(job["id"], next_run_at=due_at)
    monkeypatch.setattr("cron.scheduler._get_parallel_pool", lambda workers: RejectingPool())
    monkeypatch.setattr(
        "cron.scheduler._interpreter_shutting_down",
        lambda error=None: error is not None,
    )

    assert tick(verbose=False, sync=True) == 0
    stored = get_job(job["id"])
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored.get("run_claim") is None
    assert stored["next_run_at"] == due_at
    assert [due["id"] for due in get_due_jobs()] == [job["id"]]


def test_unstarted_release_cannot_clear_replacement_run_claim(temp_home):
    """Run-claim cleanup is owner-CAS just like fire-claim cleanup."""
    from cron.jobs import (
        claim_job_for_fire_snapshot,
        create_job,
        fire_claim_token,
        get_due_jobs,
        get_job,
        release_fire_claim,
    )

    job = create_job(prompt="x", schedule="30m", name="replacement-run", repeat=1)
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    _rewrite_job(job["id"], next_run_at=due_at)
    due = get_due_jobs()
    assert len(due) == 1
    original_owner = due[0]["run_claim"]["by"]
    claimed = claim_job_for_fire_snapshot(job["id"])
    assert claimed is not None
    token = fire_claim_token(claimed)
    assert token is not None
    _rewrite_job(
        job["id"],
        run_claim={"by": "replacement-owner", "at": datetime.now(timezone.utc).isoformat()},
    )

    assert release_fire_claim(
        job["id"],
        token,
        restore_next_run_at=due_at,
        expected_next_run_at=claimed["next_run_at"],
        expected_run_claim_owner=original_owner,
    )
    stored = get_job(job["id"])
    assert stored is not None
    assert stored.get("fire_claim") is None
    assert stored["run_claim"]["by"] == "replacement-owner"


def test_live_holder_cannot_keep_claim_forever(temp_home):
    """A reused live PID must not wedge a crashed job indefinitely."""
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire,
        create_job,
    )

    job = create_job(prompt="x", schedule="every 5m", name="bounded")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    _rewrite_claim(
        jid,
        at=(
            datetime.now(timezone.utc)
            - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS + 1)
        ).isoformat(),
    )

    assert (
        claim_job_for_fire(
            jid,
            claim_ttl_seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS * 2,
        )
        is True
    )


def test_live_holder_is_reclaimable_at_absolute_cap(temp_home, monkeypatch):
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire,
        create_job,
    )

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job(prompt="x", schedule="every 5m", name="exact-cap")
    assert claim_job_for_fire(job["id"]) is True
    _rewrite_claim(
        job["id"],
        at=(now - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS)).isoformat(),
    )

    assert claim_job_for_fire(job["id"]) is True


def test_tick_replaces_local_registry_entry_at_absolute_cap(temp_home, monkeypatch):
    """The local fast guard must not override the persisted claim's absolute cap."""
    import cron.scheduler as scheduler
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire_snapshot,
        create_job,
        fire_claim_token,
        mark_job_run,
    )

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job(prompt="x", schedule="every 5m", name="registry-cap")
    first = claim_job_for_fire_snapshot(job["id"])
    assert first is not None
    first_token = fire_claim_token(first)
    assert first_token is not None
    _rewrite_claim(
        job["id"],
        at=(now - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS)).isoformat(),
    )
    _rewrite_job(job["id"], next_run_at=(now - timedelta(seconds=1)).isoformat())
    with scheduler._running_lock:
        scheduler._running_job_ids.add(job["id"])
        scheduler._running_job_claim_tokens[job["id"]] = first_token

    replacement_tokens = []

    def run_replacement(claimed_job, **_kwargs):
        replacement_token = fire_claim_token(claimed_job)
        assert replacement_token is not None
        assert replacement_token != first_token
        replacement_tokens.append(replacement_token)
        assert scheduler._is_interrupted(job["id"], first_token) is True
        assert mark_job_run(
            job["id"],
            success=True,
            fire_claim_token=replacement_token,
        )
        return True

    monkeypatch.setattr(scheduler, "run_one_job", run_replacement)

    assert scheduler.tick(verbose=False, sync=True) == 1
    assert len(replacement_tokens) == 1
    assert scheduler.get_running_job_ids() == frozenset()


def test_shutdown_cas_loss_keeps_old_flag_while_replacement_completes(temp_home):
    """A replacement completes normally while a killed old token stays suppressed."""
    import cron.scheduler as scheduler
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire_snapshot,
        create_job,
        fire_claim_token,
        get_job,
        mark_job_run,
    )

    job = create_job(prompt="x", schedule="every 5m", name="shutdown-replacement")
    old = claim_job_for_fire_snapshot(job["id"])
    assert old is not None
    old_token = fire_claim_token(old)
    assert old_token is not None
    _rewrite_claim(
        job["id"],
        at=(
            datetime.now(timezone.utc)
            - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS + 1)
        ).isoformat(),
    )
    replacement = claim_job_for_fire_snapshot(job["id"])
    assert replacement is not None
    replacement_token = fire_claim_token(replacement)
    assert replacement_token is not None
    assert replacement_token != old_token
    with scheduler._running_lock:
        scheduler._running_job_ids.add(job["id"])
        scheduler._running_job_claim_tokens[job["id"]] = old_token
        scheduler._running_job_started_tokens.add((job["id"], old_token))

    assert scheduler.mark_running_jobs_interrupted("gateway drain") == []
    assert scheduler._is_interrupted(job["id"], old_token) is True
    assert scheduler._is_interrupted(job["id"], replacement_token) is False
    assert mark_job_run(
        job["id"],
        success=True,
        fire_claim_token=replacement_token,
    )
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_status"] == "ok"
    assert stored.get("fire_claim") is None
    assert scheduler._consume_interrupted_flag(job["id"], replacement_token) is False
    assert scheduler._consume_interrupted_flag(job["id"], old_token) is True


def test_stale_runner_cannot_deliver_after_replacement_claim(temp_home, monkeypatch):
    """Completion ownership is finalized before any user-visible delivery."""
    import cron.scheduler as scheduler
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire_snapshot,
        create_job,
        fire_claim_token,
        get_job,
    )

    job = create_job(prompt="x", schedule="every 5m", name="stale-delivery")
    old = claim_job_for_fire_snapshot(job["id"])
    assert old is not None
    old_token = fire_claim_token(old)
    assert old_token is not None
    replacement_token = None
    delivered = []

    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda claimed_job, **_kwargs: (True, "output", "OLD RESPONSE", None),
    )

    def replace_before_completion(job_id, output):
        nonlocal replacement_token
        _rewrite_claim(
            job_id,
            at=(
                datetime.now(timezone.utc)
                - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS + 1)
            ).isoformat(),
        )
        replacement = claim_job_for_fire_snapshot(job_id)
        assert replacement is not None
        replacement_token = fire_claim_token(replacement)
        assert replacement_token is not None
        assert replacement_token != old_token
        return temp_home / "old-output.md"

    monkeypatch.setattr(scheduler, "save_job_output", replace_before_completion)
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda claimed_job, content, **_kwargs: delivered.append(content),
    )

    assert scheduler.run_one_job(old, verbose=False) is True
    assert delivered == []
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["fire_claim"]["token"] == replacement_token


def test_stale_delivery_result_cannot_overwrite_newer_completion(temp_home):
    """Post-delivery bookkeeping is completion-token CAS."""
    from cron.jobs import (
        claim_job_for_fire_snapshot,
        create_job,
        fire_claim_token,
        get_job,
        mark_job_run,
        update_job_run_post_completion,
    )

    job = create_job(prompt="x", schedule="every 5m", name="delivery-cas")
    first = claim_job_for_fire_snapshot(job["id"])
    assert first is not None
    first_fire_token = fire_claim_token(first)
    assert first_fire_token is not None
    assert mark_job_run(
        job["id"],
        success=True,
        fire_claim_token=first_fire_token,
        completion_token="completion-one",
    )
    second = claim_job_for_fire_snapshot(job["id"])
    assert second is not None
    second_fire_token = fire_claim_token(second)
    assert second_fire_token is not None
    assert mark_job_run(
        job["id"],
        success=True,
        fire_claim_token=second_fire_token,
        completion_token="completion-two",
    )

    assert update_job_run_post_completion(
        job["id"],
        "completion-one",
        "stale delivery failure",
    ) is False
    assert update_job_run_post_completion(
        job["id"],
        "completion-two",
        "current delivery failure",
    ) is True
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_delivery_error"] == "current delivery failure"


def test_stale_completion_cannot_clear_replacement_claim(temp_home):
    """A capped-out old run cannot erase the newer run's ownership token."""
    from cron.jobs import (
        FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS,
        claim_job_for_fire,
        create_job,
        get_job,
        mark_job_run,
    )

    job = create_job(prompt="x", schedule="every 5m", name="claim-token")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    first = get_job(jid)
    assert first is not None
    first_token = first["fire_claim"]["token"]
    _rewrite_claim(
        jid,
        at=(
            datetime.now(timezone.utc)
            - timedelta(seconds=FIRE_CLAIM_MAX_LIVE_HOLDER_SECONDS + 1)
        ).isoformat(),
    )
    assert claim_job_for_fire(jid) is True
    replacement = get_job(jid)
    assert replacement is not None
    replacement_token = replacement["fire_claim"]["token"]
    assert replacement_token != first_token

    mark_job_run(jid, success=True, fire_claim_token=first_token)
    after_stale_completion = get_job(jid)
    assert after_stale_completion is not None
    assert after_stale_completion["fire_claim"]["token"] == replacement_token
    assert after_stale_completion.get("last_run_at") is None

    mark_job_run(jid, success=True, fire_claim_token=replacement_token)
    completed = get_job(jid)
    assert completed is not None
    assert completed.get("fire_claim") is None


def test_pid_shaped_explicit_machine_id_uses_ttl_only(temp_home, monkeypatch):
    """An explicit machine ID is opaque even when it resembles host:pid."""
    from cron.jobs import create_job, claim_job_for_fire

    monkeypatch.setenv("HERMES_MACHINE_ID", f"{socket.gethostname()}:{os.getpid()}")
    job = create_job(prompt="x", schedule="every 5m", name="explicit-id")
    jid = job["id"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_mark_job_run_clears_claim(temp_home):
    """After a recurring job completes, its claim is cleared so the next fire
    can be claimed again."""
    from cron.jobs import create_job, claim_job_for_fire, mark_job_run, get_job

    job = create_job(prompt="x", schedule="every 5m", name="c")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    claimed = get_job(jid)
    assert claimed is not None
    token = claimed["fire_claim"]["token"]

    mark_job_run(jid, success=True, fire_claim_token=token)
    assert get_job(jid).get("fire_claim") is None
    # …and the re-armed recurring job is claimable again.
    assert claim_job_for_fire(jid) is True
