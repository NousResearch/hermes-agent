"""Due-scan grace gate for one-shot cron jobs.

create_job / update_job / resume_job all reject a one-shot whose run time is
more than ONESHOT_GRACE_SECONDS in the past ("will never fire"), and
``_recoverable_oneshot_run_at`` never recovers such a schedule — but the
due-scan (``_get_due_jobs_locked``) used to dispatch any one-shot whose
*persisted* ``next_run_at`` was in the past, even hours later (gateway down
past the window, host asleep, hand-edited jobs.json). That fired a
wall-clock one-shot late, contradicting the "will never fire" contract.

These tests pin the due-scan grace gate:
  - beyond grace  -> not due; record retired with a diagnostic
  - within grace  -> still due (legitimate catch-up)
  - beyond grace + live claim -> not due, but the record is kept (a run may be
    in flight in another process; its mark_job_run must still land)
  - beyond grace + dead claim (owner exited, or TTL expired) -> retired with
    a diagnostic: nothing can land that mark_job_run, so keeping the record
    would wedge the job forever (#147102)
  - re-triggered  -> due again (the Run button still works)
"""

import socket
import subprocess
import sys

import pytest
from datetime import datetime, timedelta, timezone, UTC

from cron.jobs import (
    get_due_jobs,
    load_jobs,
    save_jobs,
    trigger_job,
)

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir and pin the clock."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _oneshot(jid, run_at_dt, *, completed=0, run_claim=None, fire_claim=None):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": completed},
        "deliver": "local",
        "run_claim": run_claim,
        "fire_claim": fire_claim,
    }


class TestOneShotGraceDueScan:
    def test_stale_beyond_grace_not_due_and_retired_with_diagnostic(self, cron_store):
        stale = _oneshot("stale", FIXED_NOW - timedelta(hours=3))
        save_jobs([stale])

        due = get_due_jobs()

        assert [d["id"] for d in due] == []
        # Record is retired (removed) so it stops being scanned — with a
        # diagnostic left behind so the miss is operator-visible.
        assert [j["id"] for j in load_jobs()] == []
        out_dir = cron_store / "cron" / "output" / "stale"
        diag = list(out_dir.glob("*.md")) if out_dir.exists() else []
        assert diag, "expected a diagnostic file for the retired stale one-shot"

    def test_within_grace_still_due(self, cron_store):
        recent = _oneshot("recent", FIXED_NOW - timedelta(seconds=60))
        save_jobs([recent])

        due = get_due_jobs()
        assert [d["id"] for d in due] == ["recent"]
        # Still stored (dispatch proceeds normally).
        assert [j["id"] for j in load_jobs()] == ["recent"]

    def test_live_claim_beyond_grace_skipped_but_record_kept(self, cron_store):
        # A live run_claim (fresh, owner not provably dead) means a run may
        # still be in flight in another process — the due-scan must not fire
        # OR retire the record, so mark_job_run can still land.
        claimed = _oneshot(
            "claimed",
            FIXED_NOW - timedelta(hours=3),
            completed=1,
            run_claim={"at": (FIXED_NOW - timedelta(seconds=60)).isoformat(), "by": "other-host:4242"},
        )
        save_jobs([claimed])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        # Record preserved (not retired) despite being beyond grace.
        assert [j["id"] for j in load_jobs()] == ["claimed"]

    def test_dead_owner_claim_beyond_grace_is_retired_with_diagnostic(self, cron_store):
        # #147102: a run_claim whose owner pid is provably dead can never land
        # its mark_job_run — treating its presence as liveness wedged the job
        # forever (never fired, never retired, never diagnosed). It must be
        # retired with the missed-run diagnostic like a missing claim.
        proc = subprocess.run(
            [sys.executable, "-c", "import os; print(os.getpid())"],
            capture_output=True, text=True, check=True)
        dead_claim = _oneshot(
            "dead-owner",
            FIXED_NOW - timedelta(hours=3),
            run_claim={
                "at": (FIXED_NOW - timedelta(seconds=60)).isoformat(),
                "by": f"{socket.gethostname()}:{int(proc.stdout.strip())}",
            },
        )
        save_jobs([dead_claim])

        due = get_due_jobs()

        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == []
        out_dir = cron_store / "cron" / "output" / "dead-owner"
        diag = list(out_dir.glob("*.md")) if out_dir.exists() else []
        assert diag, "expected a missed-run diagnostic for the retired dead-claim one-shot"

    def test_expired_claim_beyond_grace_is_retired_with_diagnostic(self, cron_store):
        # The other dead-claim shape: a claim older than the run-claim TTL. A
        # live one-shot run keeps its claim fresh via the scheduler heartbeat,
        # so an expired claim means the owning tick died — same retirement.
        expired = _oneshot(
            "expired-claim",
            FIXED_NOW - timedelta(hours=3),
            run_claim={"at": (FIXED_NOW - timedelta(hours=3)).isoformat(), "by": "other-host:4242"},
        )
        save_jobs([expired])

        due = get_due_jobs()

        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == []
        out_dir = cron_store / "cron" / "output" / "expired-claim"
        diag = list(out_dir.glob("*.md")) if out_dir.exists() else []
        assert diag, "expected a missed-run diagnostic for the retired expired-claim one-shot"

    def test_retriggered_stale_oneshot_is_due(self, cron_store):
        # A user can still explicitly re-run a stale one-shot: trigger_job
        # sets next_run_at=now, which is inside the grace window -> due.
        stale = _oneshot("stale", FIXED_NOW - timedelta(hours=3))
        save_jobs([stale])
        trigger_job("stale")

        due = get_due_jobs()
        assert [d["id"] for d in due] == ["stale"]

    def test_recurring_jobs_unaffected(self, cron_store):
        # The grace gate is once-kind only; a stale recurring job keeps its
        # existing execute-now catch-up behavior.
        interval = {
            "id": "int",
            "name": "int",
            "prompt": "x",
            "schedule": {"kind": "interval", "minutes": 60},
            "next_run_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "last_run_at": None,
            "enabled": True,
            "state": "scheduled",
            "repeat": {"times": None, "completed": 0},
            "deliver": "local",
        }
        save_jobs([interval])
        due = get_due_jobs()
        # Recurring stale handling still executes once now (existing behavior).
        assert [d["id"] for d in due] == ["int"]
