"""Stopping a live cron run (cancel / remove) and the per-job missed-run policy.

Real store, real execution ledger, real cancel markers under a temp HERMES_HOME.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from cron import jobs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        yield tmp_path


def _park(job_id: str, seconds_ago: float) -> None:
    stored = jobs.load_jobs()
    row = next(r for r in stored if r["id"] == job_id)
    row["next_run_at"] = (jobs._hermes_now() - timedelta(seconds=seconds_ago)).isoformat()
    jobs.save_jobs(stored)


def _ledger(job_id: str) -> list[dict]:
    from cron.executions import list_executions

    return list_executions(job_id=job_id)


def _live_run(job_id: str) -> str:
    from cron.executions import create_execution, mark_execution_running

    record = create_execution(job_id, source="test")
    mark_execution_running(record["id"])
    return record["id"]


# --- missed-run policy ----------------------------------------------------------------------


def test_catch_up_off_skips_and_books_one_missed_row_with_the_count(home):
    job = jobs.create_job(prompt="brief", schedule="every 1h", deliver="local", catch_up=False)
    _park(job["id"], 4 * 3600 + 60)  # slots at -4h1m, -3h1m, -2h1m, -1h1m, -1m

    assert jobs.get_due_jobs() == []
    rows = _ledger(job["id"])
    assert [r["status"] for r in rows] == ["missed"]
    assert rows[0]["error"].startswith("missed \u00d75 (skipped")
    assert rows[0]["started_at"] is None


def test_job_catch_up_beats_the_global_setting(home):
    (home / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    job = jobs.create_job(prompt="scan", schedule="every 1h", deliver="local", catch_up=True)
    _park(job["id"], 3 * 3600 + 60)

    assert [j["id"] for j in jobs.get_due_jobs()] == [job["id"]]
    # Four slots passed; one runs now, three were collapsed into it.
    assert [(r["status"], r["error"].split(" (")[0]) for r in _ledger(job["id"])] == [
        ("missed", "missed \u00d73")]


def test_global_setting_stays_the_default_for_jobs_without_their_own(home):
    (home / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    job = jobs.create_job(prompt="brief", schedule="every 1h", deliver="local")
    _park(job["id"], 2 * 3600)

    assert jobs.get_due_jobs() == []
    assert [r["status"] for r in _ledger(job["id"])] == ["missed"]


def test_explicit_grace_skips_past_it_unless_the_job_opts_into_catch_up(home):
    capped = jobs.create_job(prompt="brief", schedule="* * * * *", deliver="local", grace_minutes=5)
    opted = jobs.create_job(prompt="scan", schedule="* * * * *", deliver="local",
                            grace_minutes=5, catch_up=True)
    slot = (jobs._hermes_now() - timedelta(minutes=20)).replace(second=0, microsecond=0)
    stored = jobs.load_jobs()
    for row in stored:
        row["next_run_at"] = slot.isoformat()
    jobs.save_jobs(stored)

    assert [j["id"] for j in jobs.get_due_jobs()] == [opted["id"]]
    assert [r["status"] for r in _ledger(capped["id"])] == ["missed"]


def test_missed_slots_count_in_the_jobs_pinned_zone(home):
    """A 09:00 New York weekday job under a Los Angeles profile, overdue Thu 09:00 → Fri 15:00 NY:
    two slots (Thu, Fri), not a phantom Thu 09:00 LA occurrence in between."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from cron.missed_runs import count_missed_slots

    (home / "config.yaml").write_text("timezone: America/Los_Angeles\n", encoding="utf-8")
    ny = ZoneInfo("America/New_York")
    schedule = jobs.parse_schedule("0 9 * * 1-5")
    first = datetime(2026, 10, 22, 9, 0, tzinfo=ny).isoformat()
    now = datetime(2026, 10, 23, 15, 0, tzinfo=ny)

    assert count_missed_slots(schedule, first, now, timezone="America/New_York") == 2


def test_a_profile_distribution_carries_the_jobs_missed_run_policy(home):
    from cron.job_definition import merge_job_definition

    local = jobs.create_job(prompt="brief", schedule="0 7 * * *", deliver="local", catch_up=True)
    merged = merge_job_definition(dict(local), {"prompt": "brief", "schedule": local["schedule"],
                                                "catch_up": False})
    assert merged["catch_up"] is False


def test_one_shot_past_grace_skips_by_default_and_fires_once_when_opted_in(home):
    run_at = (jobs._hermes_now() + timedelta(hours=1)).isoformat()
    skipped = jobs.create_job(prompt="stale reminder", schedule=run_at, deliver="local")
    kept = jobs.create_job(prompt="late reminder", schedule=run_at, deliver="local", catch_up=True)
    for job in (skipped, kept):
        _park(job["id"], 3600)

    assert [j["id"] for j in jobs.get_due_jobs()] == [kept["id"]]
    assert jobs.get_job(skipped["id"]) is None
    assert [r["status"] for r in _ledger(skipped["id"])] == ["missed"]
    assert _ledger(kept["id"]) == []


def test_catch_up_rejects_non_boolean(home):
    with pytest.raises(ValueError, match="catch_up"):
        jobs.create_job(prompt="x", schedule="every 1h", deliver="local", catch_up="sometimes")


# --- ledger schema --------------------------------------------------------------------------


def test_a_pre_existing_ledger_is_widened_without_losing_rows(home):
    db = home / "cron" / "executions.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE executions (
             id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
             process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT)""")
    conn.execute(
        "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at) "
        "VALUES ('old', 'j1', 'builtin', 'p', 1, 'completed', '2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    from cron.executions import record_missed_occurrences

    record_missed_occurrences("j1", count=2, first_instant=None, detail="test")
    assert sorted(r["status"] for r in _ledger("j1")) == ["completed", "missed"]


# --- stopping a live run --------------------------------------------------------------------


def test_removing_a_job_stops_its_live_run(home):
    from cron.cancellation import CancelRequest

    job = jobs.create_job(prompt="long", schedule="every 1h", deliver="local")
    execution_id = _live_run(job["id"])

    assert jobs.remove_job(job["id"])
    request = CancelRequest(execution_id)
    assert request.is_set()
    assert request.reason == "job removed while running"


def test_a_cancelled_run_books_cancelled_without_counting_as_a_failure(home):
    from cron import scheduler
    from cron.cancellation import request_cancel

    from cron.executions import create_execution

    job = jobs.create_job(prompt="long", schedule="every 1h", deliver="local")
    execution_id = create_execution(job["id"], source="test")["id"]  # the body starts it
    request_cancel(execution_id, reason="operator asked")
    delivered: list = []

    def run_job(job, *, cancel_event=None, **_kw):
        assert cancel_event is not None and cancel_event.is_set()
        raise RuntimeError(f"Cron job was stopped: {scheduler.cancel_reason(cancel_event)}")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scheduler, "run_job", run_job)
        mp.setattr(scheduler, "_deliver_crash_failure",
                   lambda *a, **k: delivered.append(a) or (None, "delivered"))
        assert scheduler._run_one_job_body(dict(job, execution_id=execution_id)) is True

    row = _ledger(job["id"])[0]
    assert (row["status"], row["error"]) == ("cancelled", "operator asked")
    stored = jobs.get_job(job["id"])
    assert stored["last_status"] == "cancelled"
    assert not stored.get("failure_streak")
    assert delivered == []  # the operator asked for it: no failure alert
