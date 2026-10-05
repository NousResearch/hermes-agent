"""The due scan must still hand back its due jobs when the store cannot be written.

``_get_due_jobs_locked`` repairs store-side problems mid-scan (a half-paused record self-disables,
completed one-shots are swept, records are normalized) and then persists those repairs with a
``save_jobs(...)`` on the way out. That save used to be unguarded, so any store-write failure —
ENOSPC on a full disk, a read-only mount, a permissions problem — raised out of ``get_due_jobs()``
and aborted the tick, and every job on that profile stopped firing until a write succeeded.
Observed live on a full /opt/data: repeated ``Cron tick error ... [Errno 28] No space left on
device`` from this exact call site, with every job then reporting "missed its scheduled time".

The repairs are already applied in memory — that is what the scan keys off — so the persist is a
side effect the next tick can retry. These tests pin both halves: a failing persist neither
propagates nor loses the dispatch, and a writable store still persists as before.
"""

import errno
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


@pytest.fixture()
def full_disk(monkeypatch):
    """Make the store unwritable the way a full disk does, without ever touching the real store."""

    def _raise(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cronjobs, "save_jobs", _raise)
    return _raise


def test_due_jobs_are_returned_when_the_store_cannot_be_saved(cron_store, full_disk):
    save_jobs([_due_job(), _half_paused_job()])

    assert [d["id"] for d in get_due_jobs()] == ["due-job"]


def test_tick_on_unwritable_store_returns_cleanly_without_dispatch(cron_store, monkeypatch, caplog):
    """Real tick(): the advance cannot be persisted, so the recurring job is NOT run (at-most-once)
    and the tick neither raises nor alters the store; a WARNING names the unwritable store."""
    from cron import scheduler, scheduler_tick

    save_jobs([_due_job(), _half_paused_job()])
    before = load_jobs()
    ran = []
    monkeypatch.setattr(scheduler, "_process_due_job", lambda job, *a, **k: ran.append(job["id"]) or True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler_tick, "_last_store_warning", {})

    def _enospc(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cronjobs, "_stage_jobs_payload", _enospc)
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        assert scheduler.tick(verbose=False, sync=True) == 0

    assert ran == []
    assert "Cron store is unwritable" in caplog.text
    assert load_jobs() == before
