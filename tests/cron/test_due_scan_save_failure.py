"""The due scan must still hand back its due jobs when the store cannot be written.

``_get_due_jobs_locked`` repairs store-side problems mid-scan (a half-paused record self-disables,
completed one-shots are swept, records are normalized) and then persists those repairs with a
``save_jobs(...)`` on the way out. That save used to be unguarded, so any store-write failure —
ENOSPC on a full disk, a read-only mount, a permissions problem — raised out of ``get_due_jobs()``
and aborted the tick, and every job on that profile stopped firing until a write succeeded.
Observed live on a full /opt/data: repeated ``Cron tick error ... [Errno 28] No space left on
device`` from this exact call site, with every job then reporting "missed its scheduled time".

The repairs are already applied in memory — that is what the scan keys off — so the persist is a
side effect the next tick can retry. The scan still returns its due jobs; tick() skips recurring
dispatch (no durable advance) without raising or mutating the store.
"""

import errno
import json
import logging

import pytest
from datetime import datetime, timezone

from cron import jobs as cronjobs
from cron.jobs import get_due_jobs, load_jobs, save_jobs

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


def _enospc(*_args, **_kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.fixture()
def full_disk(monkeypatch):
    """Make the store unwritable the way a full disk does, without ever touching the real store."""
    monkeypatch.setattr(cronjobs, "save_jobs", _enospc)


def test_due_jobs_are_returned_when_the_store_cannot_be_saved(cron_store, full_disk, monkeypatch, caplog):
    save_jobs([_due_job(), _half_paused_job()])
    monkeypatch.setattr(cronjobs, "_last_store_warning", {})

    with caplog.at_level(logging.WARNING, logger="cron.jobs"):
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]

        with cronjobs.use_cron_store(cron_store / "other-profile"):  # a second profile's store
            cronjobs.warn_store_unwritable(OSError(errno.ENOSPC, "No space left on device"), "x", "scan")

    # Rate-limited per store: one WARNING per outage per profile, not one per 60s scan, and a
    # sibling profile on the same errno is not silenced; each names its store.
    warnings = [r.getMessage() for r in caplog.records if r.name == "cron.jobs" and r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "due-scan repairs not persisted" in warnings[0]
    assert str(cron_store / "cron") in warnings[0] and str(cron_store / "other-profile") in warnings[1]


@pytest.mark.parametrize("with_once,junk", [(True, False), (False, False), (False, True)],
                         ids=["one-shot-due", "recurring-only", "load-repair"])
def test_tick_on_unwritable_store_returns_cleanly_without_dispatch(cron_store, monkeypatch, caplog, with_once, junk):
    """Real tick(): the advance cannot be persisted, so the recurring job is NOT run (at-most-once);
    a due one-shot still reaches its fire claim, which fails closed with a ``failed`` execution row.
    The tick neither raises nor alters the store, still reaps MCP orphans (also when every due job
    was skipped), and a WARNING names the unwritable store."""
    from cron import executions, scheduler

    once = dict(_due_job("once"), schedule={"kind": "once", "run_at": FIXED_NOW.isoformat(), "display": "once"},
                repeat={"times": 1, "completed": 0})
    save_jobs([_due_job(), _half_paused_job()] + ([once] if with_once else []))
    before = load_jobs()
    if junk:  # load_jobs() drops a non-object entry and persists that repair: the save must not abort the tick
        raw = json.loads(cronjobs.JOBS_FILE.read_text())
        cronjobs.JOBS_FILE.write_text(json.dumps(dict(raw, jobs=raw["jobs"] + [42])))
    ran, sweeps = [], []
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_store / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda job, **k: ran.append(job["id"]) or True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: sweeps.append(1))
    monkeypatch.setattr(cronjobs, "_last_store_warning", {})
    monkeypatch.setattr(cronjobs, "_stage_jobs_payload", _enospc)
    with caplog.at_level(logging.WARNING):
        assert scheduler.tick(verbose=False, sync=True) == 0

    assert ran == [] and sweeps == [1]
    assert f"Cron store {cron_store / 'cron'} is unwritable" in caplog.text
    assert "skipped 1 recurring job(s)" in caplog.text  # not silenced by the scan's earlier warning
    assert executions.latest_execution("due-job") is None  # skipped before its fire claim
    assert load_jobs() == before
    row = executions.latest_execution("once")
    if with_once:
        assert row["status"] == "failed" and "Cron store unwritable" in row["error"]
    else:
        assert row is None


@pytest.mark.parametrize("site", ["claim_job_for_fire", "note_cron_execution"])
def test_dispatch_failure_after_receipt_never_leaves_it_claimed(cron_store, monkeypatch, site):
    """A non-OSError fire-claim failure, or a failure after create_execution in _submit_with_guard,
    must settle the receipt: a ``claimed`` row never resolves and reads forever as in flight."""
    from cron import executions, scheduler

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    save_jobs([_due_job()])
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_store / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda job, **k: True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler, site, boom)
    try:
        scheduler.tick(verbose=False, sync=True)
    except RuntimeError:
        pass  # a claim failure may still surface; the receipt is what must be terminal
    row = executions.latest_execution("due-job")
    assert row is not None and row["status"] == "failed"
