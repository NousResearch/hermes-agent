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