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
  - beyond grace + stale claim -> exhausted record retired after restart
  - within grace + stale claim + exhausted -> retired as wedged, never re-fired
  - beyond grace + live claim -> record kept so mark_job_run can land
  - re-triggered  -> due again (the Run button still works)
"""

import pytest
from datetime import datetime, timedelta, timezone

from cron.jobs import (
    get_due_jobs,
    load_jobs,
    save_jobs,
    trigger_job,
)

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


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

    @pytest.mark.parametrize("claim_field", ["run_claim", "fire_claim"])
    def test_stale_claim_after_restart_does_not_block_exhausted_retirement(
        self, cron_store, monkeypatch, claim_field,
    ):
        monkeypatch.setattr("cron.jobs._job_running_in_this_process", lambda _job_id: False)
        claim = {"at": (FIXED_NOW - timedelta(hours=3)).isoformat(), "by": "other"}
        claimed = _oneshot(
            "claimed",
            FIXED_NOW - timedelta(hours=3),
            completed=1,
            **{claim_field: claim},
        )
        save_jobs([claimed])

        assert get_due_jobs() == []
        assert load_jobs() == []

    @pytest.mark.parametrize("claim_field", ["run_claim", "fire_claim"])
    def test_exhausted_oneshot_with_stale_claim_within_grace_retires_instead_of_refiring(
        self, cron_store, monkeypatch, claim_field,
    ):
        """Stale crash residue + completed >= times, still INSIDE the one-shot
        grace window: the grace gate alone would call this schedule fireable,
        so the exhausted check must run first and retire it as a wedged run,
        not dispatch it again and not report it as a missed schedule."""
        monkeypatch.setattr("cron.jobs._job_running_in_this_process", lambda _job_id: False)
        missed = []
        monkeypatch.setattr("cron.jobs.record_cron_missed", lambda job: missed.append(job["id"]))
        run_at = FIXED_NOW - timedelta(seconds=60)  # inside ONESHOT_GRACE_SECONDS
        claim = {"at": (FIXED_NOW - timedelta(hours=3)).isoformat(), "by": "dead-tick"}
        save_jobs([_oneshot("wedged", run_at, completed=1, **{claim_field: claim})])

        assert get_due_jobs() == []
        assert load_jobs() == []
        assert missed == []
        diag = list((cron_store / "cron" / "output" / "wedged").glob("*.md"))
        assert len(diag) == 1
        text = diag[0].read_text(encoding="utf-8")
        assert "outside grace window" not in text
        assert text.startswith("# Cron job removed without producing output")

    @pytest.mark.parametrize(
        ("claim_field", "claimed_at", "locally_running"),
        [
            pytest.param("run_claim", FIXED_NOW, False, id="run-lease"),
            pytest.param("fire_claim", FIXED_NOW, False, id="fire-lease"),
            pytest.param(
                "run_claim", FIXED_NOW - timedelta(hours=3), True, id="local-running-set"
            ),
        ],
    )
    def test_live_owner_keeps_exhausted_oneshot_for_its_finishing_run(
        self, cron_store, monkeypatch, claim_field, claimed_at, locally_running,
    ):
        monkeypatch.setattr(
            "cron.jobs._job_running_in_this_process", lambda _job_id: locally_running
        )
        claim = {"at": claimed_at.isoformat(), "by": "other"}
        claimed = _oneshot(
            "claimed",
            FIXED_NOW - timedelta(hours=3),
            completed=1,
            **{claim_field: claim},
        )
        save_jobs([claimed])

        assert get_due_jobs() == []
        assert [job["id"] for job in load_jobs()] == ["claimed"]

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
