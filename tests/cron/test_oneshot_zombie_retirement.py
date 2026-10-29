"""Regression test for #126787 — spent one-shot must retire before the
completed-occurrence fast path.

A one-shot whose budget is spent (``repeat.completed >= times``) and whose
slot already carries a ``status='completed'`` ledger row used to hit the
completed-occurrence check FIRST, which returned without persisting or
retiring (``recurring`` is False, so ``new_next`` was None). The record sat
in jobs.json as ``state='scheduled'`` forever — retention only sweeps
``state=='completed'`` — and every tick repeated the ledger query: the
permanent zombie the issue describes.

The fix hoists the two one-shot retirement guards above that fast path, so
the spent record is removed on the next tick instead of being skipped past
its own retirement.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as jobs_mod
from cron.executions import _process_start_time
from cron.jobs import ONESHOT_GRACE_SECONDS, _hermes_now


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _seed_completed_ledger_row(job_id: str, slot: str) -> None:
    """Insert a status='completed' executions row for *job_id*'s slot so
    ``completed_occurrence`` proves the occurrence already ran."""
    from cron import executions
    from cron.occurrences import scheduled_instant

    with executions._transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at, status,
                claimed_at, finished_at, scheduled_instant)
               VALUES ('seed-completed', ?, 'test', 'test', ?, ?, 'completed',
                       ?, ?, ?)""",
            (str(job_id), os.getpid(), _process_start_time(os.getpid()),
             _hermes_now().isoformat(), _hermes_now().isoformat(),
             scheduled_instant(slot)),
        )


def _spent_oneshot_zombie() -> dict:
    """A one-shot that looks due, whose budget is spent, and whose slot has a
    completed ledger row — the zombie shape from the issue."""
    slot = _hermes_now() - timedelta(seconds=5)  # overdue, within dispatch grace
    job = jobs_mod.create_job(prompt="x", schedule="in 30m", name="zombie-oneshot")
    jobs_list = jobs_mod.load_jobs()
    for j in jobs_list:
        if j["id"] == job["id"]:
            j["schedule"] = {"kind": "once", "run_at": slot.isoformat()}
            j["next_run_at"] = slot.isoformat()
            j["enabled"] = True
            j["state"] = "scheduled"
            j["repeat"] = {"times": 1, "completed": 1}
    jobs_mod.save_jobs(jobs_list)
    _seed_completed_ledger_row(job["id"], slot.isoformat())
    return job


class TestSpentOneshotRetiresBeforeCompletedOccurrence:
    @pytest.mark.parametrize("completed", [1, 3])
    def test_future_oneshot_with_inherited_spent_budget_survives_until_due(
        self, cron_store, monkeypatch, completed,
    ):
        """A real recurring→once edit can retain its earlier run counter."""
        from datetime import datetime
        from tools.cronjob_tools import _update_run_fields

        job = jobs_mod.create_job(prompt="x", schedule="every 5m", name="converted")
        for _ in range(completed):
            assert jobs_mod.advance_next_run(job["id"])
            jobs_mod.mark_job_run(job["id"], success=True)
        current = jobs_mod.get_job(job["id"])
        assert current is not None
        updates = {}
        arguments = {
            "enabled_toolsets": None, "attach_to_session": None, "workdir": None,
            "no_agent": None, "repeat": None, "schedule": "in 30m",
        }
        assert _update_run_fields(current, arguments, updates) is None
        converted = jobs_mod.update_job(job["id"], updates)
        assert converted is not None
        assert converted["repeat"] == {"times": 1, "completed": completed}
        slot = datetime.fromisoformat(converted["next_run_at"])
        before = jobs_mod.load_jobs()

        monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: slot - timedelta(seconds=1))
        assert jobs_mod.get_due_jobs() == []
        assert jobs_mod.load_jobs() == before
        assert jobs_mod.get_job(job["id"]) is not None

        # Due-time budget enforcement remains intact; only early retirement
        # is prohibited, not retirement when the converted slot arrives.
        monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: slot)
        assert jobs_mod.get_due_jobs() == []
        assert jobs_mod.get_job(job["id"]) is None

    def test_zombie_record_is_removed_not_looped_forever(self, cron_store):
        job = _spent_oneshot_zombie()
        jid = job["id"]

        due = jobs_mod.get_due_jobs()

        # The tick must retire the spent record, not skip past its retirement.
        assert jid not in [d["id"] for d in due]
        assert jid not in [j["id"] for j in jobs_mod.load_jobs()]

    def test_future_oneshot_is_never_retired_early(self, cron_store):
        """The hoisted guards must stay inert for a healthy scheduled one-shot:
        a future slot is always within the grace window and the budget is not
        spent, so retirement must not fire early."""
        from datetime import timedelta as _td

        job = jobs_mod.create_job(prompt="x", schedule="in 30m", name="healthy")
        jobs_list = jobs_mod.load_jobs()
        for j in jobs_list:
            if j["id"] == job["id"]:
                j["next_run_at"] = (_hermes_now() + _td(hours=1)).isoformat()
                j["state"] = "scheduled"
        jobs_mod.save_jobs(jobs_list)

        due = jobs_mod.get_due_jobs()

        assert job["id"] not in [d["id"] for d in due]  # not due yet
        surviving = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]]
        assert surviving, "healthy scheduled one-shot must survive the tick"
