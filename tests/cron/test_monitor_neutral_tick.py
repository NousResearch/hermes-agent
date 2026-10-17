"""A suppressed (unchanged) monitor tick is NEUTRAL for the job's failure state.

The monitor gate reports an unchanged tick as a successful run (it costs no agent turn and
delivers nothing), but that success must not read as "the job's work just succeeded": doing so
resolved the job's open failure incidents and reset ``failure_streak`` on EVERY unchanged tick,
which made ``cron.failure_repeat_alert_hours`` and ``cron.failure_nudge_threshold`` dead letters
for monitor-gated jobs — every real failure arrived as the first failure of a fresh streak, i.e.
one notice per failing run instead of one plus a reminder cadence (#140178).

The state a suppressed tick touches is the store record plus the incident ledger, so these tests
drive the tick's real body (``cron.scheduler._run_one_job_body``: execute → save → compose →
deliver → mark) rather than a stubbed bookkeeping tail.
"""

from __future__ import annotations

import pytest

import cron.scheduler as sched
from cron.jobs import create_job, get_job, update_job

FAILURE = "CronPromptInjectionBlocked: attached skill content tripped the scanner"
MONITOR_OUTPUT = "state A"


@pytest.fixture
def cron_home(tmp_path, monkeypatch, cron_owner):
    """Isolate HERMES_HOME: the cron store (jobs.json + executions.db) and the scripts dir.

    ``cron_owner`` (tests/cron/conftest.py) keeps the production owner execution seam in-process,
    so driving the tick's body exercises the real gate + bookkeeping instead of a live gateway.
    """
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _monitor_job(cron_home, body: str = f"echo '{MONITOR_OUTPUT}'\n"):
    (cron_home / "scripts" / "mon.sh").write_text(body, encoding="utf-8")
    return create_job(
        "Report what changed", "every 5m", name="watch the queue", model="test-model",
        monitor_script="mon.sh", deliver="local",
    )


def _state_after_a_failed_run(job, *, streak: int) -> str:
    """Stamp the record the way a really failed run leaves it (plus its open, alerted incident).

    A monitor job also gets the baseline the suppressed tick compares against, hashed from the
    output the gate itself observes — so the tick is a genuine ``no_change``.
    """
    from cron.incidents import set_incident_state, upsert_incident
    from cron.monitor import _run_monitor_source, hash_monitor_output, job_has_monitor

    updates: dict = {
        "failure_streak": streak,
        "last_status": "error",
        "last_error": FAILURE,
        "last_failure": {"at": "2026-10-17T00:00:00+00:00", "detail": FAILURE},
    }
    if job_has_monitor(job):
        ok, output = _run_monitor_source(job)
        assert ok, output
        updates["monitor_state"] = {
            "last_output_hash": hash_monitor_output(output),
            "last_changed_at": "2026-10-17T00:00:00+00:00",
        }
    update_job(job["id"], updates)
    incident_id, _is_new = upsert_incident(job["id"], FAILURE)
    assert set_incident_state(incident_id, "alerted")
    return incident_id


def test_unchanged_monitor_tick_leaves_the_failure_state_alone(cron_home):
    """The reported bug: a no_change tick recovered the job on every unchanged source poll."""
    from cron.incidents import get_incident

    job = _monitor_job(cron_home)
    incident_id = _state_after_a_failed_run(job, streak=2)

    assert sched._run_one_job_body(get_job(job["id"])) is True

    after = get_job(job["id"])
    assert after["failure_streak"] == 2, "an unchanged tick is not a success: the streak must stand"
    assert (after["last_status"], after["last_error"]) == ("error", FAILURE), (
        "the tick ran no agent work, so it must leave the last real run's verdict on the record"
    )
    assert get_incident(incident_id)["state"] == "alerted", (
        "an unchanged source says nothing about the job's own work: it cannot resolve its incident"
    )


def test_unchanged_monitor_tick_still_advances_the_schedule(cron_home):
    """Neutrality is about the job's HEALTH only: the tick still owns and books its slot."""
    job = _monitor_job(cron_home)
    _state_after_a_failed_run(job, streak=1)
    before = get_job(job["id"])

    assert sched._run_one_job_body(dict(before)) is True

    after = get_job(job["id"])
    assert after["last_run_at"] != before.get("last_run_at")
    assert after["repeat"]["completed"] == before["repeat"]["completed"] + 1
    assert after["fire_claim"] is None


def test_a_real_run_still_recovers_the_job(cron_home):
    """The other half of the contract: work that DID succeed still resolves and resets."""
    from cron.incidents import get_incident
    from cron.jobs import mark_job_run

    job = create_job("Report the backlog", "every 5m", model="test-model", deliver="local")
    incident_id = _state_after_a_failed_run(job, streak=2)

    tick = get_job(job["id"])
    sched._compose_run_delivery(
        tick, success=True, error=None, final_response="backlog is clear", output_file=None)
    assert get_incident(incident_id)["state"] == "resolved"

    assert mark_job_run(job["id"], True, None)
    after = get_job(job["id"])
    assert (after["failure_streak"], after["last_status"]) == (0, "ok")


def test_the_gate_flags_a_suppressed_tick_and_misses_the_changed_one(cron_home):
    """The seam the bookkeeping tail reads: set for no_change, absent for a real change."""
    from cron.monitor import suppressed_tick

    job = _monitor_job(cron_home)
    # No stored hash yet: the first tick's output IS a change, so the agent run is not suppressed.
    first = get_job(job["id"])
    early, _extra_prompt, context = sched._apply_monitor_gate(first, first["id"], "watch", None)
    assert early is None and context
    assert suppressed_tick(first) is False

    _state_after_a_failed_run(job, streak=1)
    unchanged = get_job(job["id"])
    early, _extra_prompt, _context = sched._apply_monitor_gate(unchanged, unchanged["id"], "watch", None)
    assert early is not None and early[2] == sched.SILENT_MARKER
    assert suppressed_tick(unchanged) is True
