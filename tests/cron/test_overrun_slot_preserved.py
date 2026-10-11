"""A job whose run outlives its own period must not lose the slots it overruns.

The tick takes the next occurrence off the schedule (advance + ``pending_slot`` stamp) BEFORE
dispatch, so a crash mid-run cannot re-fire it (at-most-once). The in-flight guard then refuses to
dispatch that occurrence while the PREVIOUS run is still running — correct, and the window
``test_missed_window_catchup`` covers for a process that DIED inside it.

What was not correct: when the overrunning run finally completed, ``mark_job_run`` re-anchored
``next_run_at`` from *now* — landing PAST the occurrence it had just refused — and dropped the
``pending_slot`` stamp that recorded it. The slot then vanished with no run, no output and no
execution row: an invisible missed fire, invisible to ``_restore_unclaimed_slot`` (its input was
gone) and to the missed-fire watchdog (``last_run_at`` looked like it covered the schedule).

Drives the REAL ``tick()`` against a throwaway HERMES_HOME with a ``no_agent`` script job that
appends one line per fire, with the previous run held in the process's own in-flight set exactly as
a live run is.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def overrun_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    (home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)
    # Hermetic PATH: a sandbox that launches pytest from a shell carrying the real install tree's
    # launcher bin (~/.hermes/installs/.../venv/bin) makes the script run stat a real-home path,
    # which tests/home_io_guard.py fails as a TEST BUG before the scheduler is even reached.
    real_home = str(Path.home() / ".hermes")
    monkeypatch.setenv("PATH", os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep) if p and not p.startswith(real_home)))

    import cron.executions as E
    import cron.jobs as J
    import cron.scheduler as S

    monkeypatch.setattr(J, "HERMES_DIR", home)
    monkeypatch.setattr(J, "CRON_DIR", home / "cron")
    monkeypatch.setattr(J, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(J, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(E, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(S, "_hermes_home", home)
    S._running_job_ids.clear()
    S._running_since.clear()
    S._running_futures.clear()

    counter = home / "fires.txt"
    # A .py script on THIS interpreter: an unpinned script run resolves the install store's Python
    # and tests/home_io_guard.py fails that as a test bug (real-home I/O), not a scheduler one.
    script = home / "scripts" / "fire.py"
    script.write_text(f"open({str(counter)!r}, 'a').write('fired\\n')\n", encoding="utf-8")
    job = J.create_job(prompt=None, schedule="every 1h", name="overrun", script="fire.py",
                       interpreter=sys.executable, no_agent=True, deliver="local")
    # The occurrence under test is already due: its run STARTED, and it is the run that will outlive
    # the period (the tick below is the one that lands inside it).
    slot = (J._hermes_now() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    stored = J.load_jobs()
    next(r for r in stored if r["id"] == job["id"])["next_run_at"] = slot
    J.save_jobs(stored)

    def fires() -> int:
        return counter.read_text(encoding="utf-8").count("\n") if counter.exists() else 0

    yield {"job_id": job["id"], "slot": slot, "fires": fires, "S": S, "J": J, "E": E}
    S._shutdown_parallel_pool()


class TestOverrunSlotPreserved:
    def test_overrunning_run_must_not_lose_the_slot_it_refused(self, overrun_env):
        S, J, E = overrun_env["S"], overrun_env["J"], overrun_env["E"]
        job_id, slot = overrun_env["job_id"], overrun_env["slot"]

        # The previous run is still in flight — held in the process's own in-flight set, which is
        # exactly what makes _submit_with_guard refuse the next dispatch.
        assert S.try_register_running_job(job_id) is True
        S.tick(verbose=False, sync=True)

        mid = J.get_job(job_id)
        assert overrun_env["fires"]() == 0, "dispatch must be refused while the run is in flight"
        assert isinstance(mid.get("pending_slot"), dict), \
            "the refused occurrence must be recorded as taken-off-the-schedule"
        assert J._ensure_aware(datetime.fromisoformat(mid["next_run_at"])) > J._hermes_now(), \
            "the tick advances past the refused occurrence (at-most-once)"

        # The overrunning run finally completes; the worker releases its dedupe key before
        # mark_job_run lands, exactly as production does.
        S.release_running_job(job_id)
        assert J.mark_job_run(job_id, success=True) is True

        after = J.get_job(job_id)
        assert J._ensure_aware(datetime.fromisoformat(after["next_run_at"])) <= J._hermes_now(), \
            ("the refused occurrence is still owed: next_run_at must come back to it, not skip it "
             "to the following slot")
        assert "pending_slot" not in after, "the stamp is consumed, never left to age"

        # It fires exactly once, keeping the occurrence's own identity, and is never replayed.
        S.tick(verbose=False, sync=True)
        _row = E.latest_execution(job_id)
        assert overrun_env["fires"]() == 1, (
            f"the refused occurrence must fire once (row={_row})")
        from cron.occurrences import scheduled_instant

        row = E.latest_execution(job_id)
        assert row["status"] == "completed", row.get("error")
        assert row["scheduled_instant"] == scheduled_instant(slot), \
            "the restored slot keeps its identity"
        S.tick(verbose=False, sync=True)
        assert overrun_env["fires"]() == 1, "one occurrence, one fire"
        assert "pending_slot" not in J.get_job(job_id)

    def test_occurrence_the_ledger_already_completed_is_not_rearmed(self, overrun_env):
        """At-most-once for the restored instant: a completed row for the slot means someone ran it."""
        S, J, E = overrun_env["S"], overrun_env["J"], overrun_env["E"]
        job_id, slot = overrun_env["job_id"], overrun_env["slot"]
        from cron.occurrences import scheduled_instant

        assert S.try_register_running_job(job_id) is True
        S.tick(verbose=False, sync=True)   # refuses the occurrence, stamps pending_slot

        # Another process ran that very occurrence and completed it.
        ex = E.create_execution(job_id, source="builtin", scheduled_instant=scheduled_instant(slot))
        assert E.finish_execution(ex["id"], success=True) is not None

        S.release_running_job(job_id)
        assert J.mark_job_run(job_id, success=True) is True

        after = J.get_job(job_id)
        assert "pending_slot" not in after
        assert J._ensure_aware(datetime.fromisoformat(after["next_run_at"])) > J._hermes_now(), \
            "an occurrence the ledger already completed must not be re-armed"
        S.tick(verbose=False, sync=True)
        assert overrun_env["fires"]() == 0, "no second run for one occurrence"