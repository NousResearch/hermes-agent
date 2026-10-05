"""A store that cannot be written must degrade the cron tick, not stop it.

``_get_due_jobs_locked`` repairs store-side problems mid-scan (a half-paused record
self-disables, completed one-shots are swept, records are normalized) and then persists those
repairs with a ``save_jobs(...)`` on the way out. That save used to be unguarded, so any
store-write failure — ENOSPC on a full disk, a read-only mount, a permissions problem — raised
out of ``get_due_jobs()`` and aborted the tick, and every job on that profile stopped firing
until a write succeeded. Observed live on a full /opt/data: repeated ``Cron tick error ...
[Errno 28] No space left on device`` from this exact call site, with every job then reporting
"missed its scheduled time".

The scan is only half the tick, though: ``tick()`` then calls ``advance_next_runs()`` to take
the recurring occurrence off the schedule before dispatch, and THAT persist raised under the
same conditions — so the tick still aborted before ``_submit_with_guard`` and the profile
stopped firing anyway. The repairs/advance are already applied in memory — that is what the
scan and the same-process dedupe key off — so the persist is a side effect the next tick can
retry.

Both halves are pinned here: a failing persist neither propagates nor loses the dispatch, and
at-most-once survives it, because the authoritative gate is the run's own durable fire claim —
``claim_job_for_fire`` persists the advance and clears ``pending_slot`` BEFORE any side effect
and fails closed while the store is unwritable. So a contained failure dispatches the
occurrence (it is not silently dropped) but can never let it EXECUTE unpersisted: it fires
exactly once, after the store accepts writes again. A writable store still persists as before.
"""

import errno
import logging

import pytest
from datetime import datetime, timezone

from cron import jobs as cronjobs
from cron.jobs import advance_next_runs, get_due_jobs, load_jobs, save_jobs

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir and pin the clock."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _due_job(jid="due-job"):
    """A recurring job whose next run is the pinned instant — on the schedule's grid, so the scan
    dispatches it rather than re-anchoring it."""
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "cron", "expr": "* * * * *", "display": "every minute"},
        "next_run_at": FIXED_NOW.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "active",
        "repeat": None,
        "deliver": "local",
    }


def _half_paused_job(jid="half-paused"):
    """enabled=true with pause markers: the scan repairs it in place and sets needs_save, which is
    what makes the scan persist on its way out — the call that used to abort the tick."""
    job = _due_job(jid)
    job["paused_at"] = FIXED_NOW.isoformat()
    job["paused_reason"] = "test"
    return job


@pytest.fixture()
def full_disk(monkeypatch):
    """Make the store unwritable the way a full disk does, without ever touching the real store."""

    def _raise(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cronjobs, "save_jobs", _raise)
    return _raise


@pytest.fixture()
def claim_persist_fails(monkeypatch):
    """Fail ONLY the fire claim's persist, leaving the tick's own writes intact.

    The claim is the one store write distinguishable by its payload: it is the sole pass that
    stamps ``fire_claim`` on the record. Failing on that discriminator reproduces the reviewer's
    sequence — scan and advance persist fine, dispatch happens, and the claim raises — rather
    than the blunter whole-store failure the other cases inject.
    """
    real_save = cronjobs.save_jobs
    attempts: list = []

    def _claim_only(jobs, *args, **kwargs):
        if any(isinstance(j.get("fire_claim"), dict) for j in jobs):
            attempts.append(True)
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_save(jobs, *args, **kwargs)

    monkeypatch.setattr(cronjobs, "save_jobs", _claim_only)
    return attempts


def _open_receipts(tmp_path):
    """Every execution row the ledger still holds in a non-terminal state."""
    import cron.executions as executions

    rows = executions.list_executions(limit=100)
    return [r for r in rows if r["status"] in ("claimed", "running")]


def _record_dispatch(monkeypatch, sched, ran=None):
    """Stub the tick's executor boundary so a full tick runs cheaply, recording what it dispatched.

    ``create_execution`` is the first side-effecting step ``_submit_with_guard`` takes after its
    in-flight guard, so a recorded call proves the tick reached dispatch; ``run_one_job`` records
    what actually executed (the fire claim must be won first — it is never stubbed).
    """
    submitted: list = []
    executed: list = ran if ran is not None else []

    def _create_execution(job_id, **_kwargs):
        submitted.append(job_id)
        return {"id": f"exec-{job_id}"}

    monkeypatch.setattr(sched, "create_execution", _create_execution)
    monkeypatch.setattr(sched, "run_one_job", lambda job, **_kw: executed.append(job["id"]) or True)
    return submitted


class TestDueScanSaveFailure:
    def test_due_jobs_are_returned_when_the_store_cannot_be_saved(self, cron_store, full_disk):
        save_jobs([_due_job(), _half_paused_job()])

        assert [d["id"] for d in get_due_jobs()] == ["due-job"]

    def test_the_failed_persist_is_logged(self, cron_store, full_disk, caplog):
        save_jobs([_due_job(), _half_paused_job()])

        with caplog.at_level(logging.WARNING):
            get_due_jobs()

        assert "could not be persisted" in caplog.text
        assert "No space left on device" in caplog.text

    def test_a_writable_store_still_persists_the_repair(self, cron_store):
        """No regression: when the store is writable the repair still lands."""
        save_jobs([_due_job(), _half_paused_job()])

        assert [d["id"] for d in get_due_jobs()] == ["due-job"]
        repaired = {j["id"]: j for j in load_jobs()}
        assert repaired["half-paused"]["enabled"] is False
        assert repaired["half-paused"]["state"] == "paused"

    def test_the_repair_is_retried_by_the_next_scan(self, cron_store, full_disk, monkeypatch):
        """The failed persist is deferred, not lost."""
        save_jobs([_due_job(), _half_paused_job()])
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]
        # Still unsaved after the failure.
        assert {j["id"]: j["enabled"] for j in load_jobs()}["half-paused"] is True

        # Disk freed: the same repair is still pending in the store and now lands.
        monkeypatch.setattr(cronjobs, "save_jobs", save_jobs)
        get_due_jobs()
        assert {j["id"]: j["enabled"] for j in load_jobs()}["half-paused"] is False

    def test_repeated_failures_stay_contained(self, cron_store, full_disk, monkeypatch):
        """A store that stays unwritable across ticks keeps dispatching, one entry per job per scan,
        and the repair lands as soon as writing works again."""
        save_jobs([_due_job(), _half_paused_job()])

        for _ in range(3):
            assert [d["id"] for d in get_due_jobs()] == ["due-job"]

        monkeypatch.setattr(cronjobs, "save_jobs", save_jobs)
        get_due_jobs()
        assert {j["id"]: j["enabled"] for j in load_jobs()}["half-paused"] is False

    def test_only_the_persist_is_contained(self, cron_store):
        """The fix narrows one call: it does not blanket-swallow scan errors."""
        save_jobs([_due_job()])
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]  # no repair pending, no save

    def test_a_non_store_error_from_the_save_still_surfaces(self, cron_store, monkeypatch):
        save_jobs([_due_job(), _half_paused_job()])

        def _boom(*_args, **_kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(cronjobs, "save_jobs", _boom)
        with pytest.raises(RuntimeError):
            get_due_jobs()


class TestScheduleAdvanceSaveFailure:
    """``advance_next_runs`` persists the recurring advance the tick makes before dispatch."""

    def test_the_advance_is_contained_and_logged(self, cron_store, full_disk, caplog):
        save_jobs([_due_job()])

        with caplog.at_level(logging.WARNING):
            advanced = advance_next_runs(["due-job"])

        assert advanced == 1  # the in-memory advance stands; the tick is not aborted
        assert "No space left on device" in caplog.text
        assert "errno=%s" % errno.ENOSPC in caplog.text

    def test_a_writable_store_still_persists_the_advance(self, cron_store):
        save_jobs([_due_job()])

        assert advance_next_runs(["due-job"]) == 1
        assert load_jobs()[0]["next_run_at"] != FIXED_NOW.isoformat()

    def test_nothing_advanced_means_no_save_attempted(self, cron_store, full_disk):
        save_jobs([_due_job()])

        assert advance_next_runs([]) == 0
        assert advance_next_runs(["unknown-job"]) == 0

    def test_a_non_store_error_from_the_advance_still_surfaces(self, cron_store, monkeypatch):
        save_jobs([_due_job()])

        def _boom(*_args, **_kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(cronjobs, "save_jobs", _boom)
        with pytest.raises(RuntimeError):
            advance_next_runs(["due-job"])


class TestFullTickWithUnwritableStore:
    """The whole tick, not just the scan: a store that cannot be written degrades to dispatch
    rather than stopping the profile — without ever letting an occurrence run unclaimed."""

    def test_the_due_job_reaches_dispatch_when_the_store_cannot_be_written(
            self, cron_store, full_disk, monkeypatch):
        """The reviewer's repro: the tick now survives the advance's failed persist and submits."""
        import cron.scheduler as sched

        save_jobs([_due_job(), _half_paused_job()])
        submitted = _record_dispatch(monkeypatch, sched)
        sched._running_job_ids.clear()

        sched.tick(verbose=False, sync=True)  # must not raise

        assert submitted == ["due-job"]
        sched._shutdown_parallel_pool()

    def test_the_occurrence_never_executes_unclaimed_and_fires_once_after_recovery(
            self, cron_store, full_disk, monkeypatch):
        """At-most-once, unweakened: while the store rejects writes the fire claim fails closed, so
        repeated ticks neither execute the occurrence nor double-fire it; once writes land it runs
        exactly once."""
        import cron.scheduler as sched

        save_jobs([_due_job(), _half_paused_job()])
        ran: list = []
        submitted = _record_dispatch(monkeypatch, sched, ran=ran)
        sched._running_job_ids.clear()

        for _ in range(3):
            sched.tick(verbose=False, sync=True)

        assert ran == []                      # dispatched (no silent drop) but never executed
        assert submitted.count("due-job") >= 1

        # Store writable again: exactly one execution, and only one, for that occurrence.
        monkeypatch.setattr(cronjobs, "save_jobs", save_jobs)
        sched.tick(verbose=False, sync=True)
        assert ran == ["due-job"]

        sched.tick(verbose=False, sync=True)
        assert ran == ["due-job"]             # the occurrence was consumed; no replay

        sched._shutdown_parallel_pool()


class TestFireClaimFailureClosesTheReceipt:
    """A claim that cannot be written must still close the execution receipt it was dispatched with.

    ``_submit_with_guard`` creates the ledger row BEFORE the worker claims the occurrence, and
    ``run_one_job`` is what normally owns the terminal writes. A claim that RAISES used to exit
    between those two points, so the row stayed ``claimed`` forever — a permanently in-flight
    attempt for a run that never happened. These drive the real ledger through a full tick.

    On ``main`` these pass trivially (the tick aborts before the row is created); at ``5742a7cd19``
    the open-row assertion is what fails.
    """

    @pytest.fixture(autouse=True)
    def ledger(self, monkeypatch, tmp_path):
        """Point the ledger at a temp file so a full tick can persist real receipts."""
        import cron.executions as executions

        monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")

    def test_a_raising_fire_claim_leaves_no_open_receipt(
            self, cron_store, claim_persist_fails, monkeypatch, tmp_path):
        """The reviewer's regression: the tick's claim raises, and the receipt is still terminal."""
        import cron.executions as executions
        import cron.scheduler as sched

        save_jobs([_due_job(), _half_paused_job()])
        ran: list = []
        monkeypatch.setattr(sched, "run_one_job", lambda job, **_kw: ran.append(job["id"]) or True)
        sched._running_job_ids.clear()

        sched.tick(verbose=False, sync=True)   # must not raise

        assert claim_persist_fails, "the fire claim's persist never failed — the repro is not genuine"
        assert ran == []                       # fail-closed: the occurrence was NOT executed
        assert _open_receipts(tmp_path) == []  # and nothing is left non-terminal

        rows = executions.list_executions(limit=100)
        assert [r["status"] for r in rows] == ["failed"]
        assert "Fire claim" in (rows[0]["error"] or "")
        sched._shutdown_parallel_pool()

    def test_a_non_store_claim_error_still_surfaces_but_settles_the_receipt(
            self, cron_store, monkeypatch, tmp_path):
        """Only ``OSError`` is contained: anything else propagates — after the row is settled."""
        import cron.executions as executions
        import cron.scheduler as sched

        save_jobs([_due_job()])
        receipt = executions.create_execution("due-job", source="builtin")

        def _boom(*_args, **_kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(sched, "claim_job_for_fire", _boom)
        monkeypatch.setattr(sched, "run_one_job", lambda job, **_kw: pytest.fail("must not run"))
        sched._running_job_ids.clear()

        with pytest.raises(RuntimeError):
            sched._process_due_job(
                {"id": "due-job", "execution_id": receipt["id"]}, None, None, False)

        assert _open_receipts(tmp_path) == []
        settled = executions.get_execution(receipt["id"])
        assert settled is not None and settled["status"] == "failed"
