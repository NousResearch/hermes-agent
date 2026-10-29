"""A manual run is an EXTRA fire: it never consumes the job's scheduled occurrence or repeat budget.

Each test drives the real bookkeeping sequence a fire goes through (claim_job_for_fire ->
claim_dispatch -> mark_job_run), the same calls the cronjob tool, the dashboard and the
ticker make around the agent run.
"""
from datetime import datetime, timedelta

import pytest

from cron.jobs import (
    advance_next_runs,
    claim_dispatch,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    heartbeat_fire_claim,
    load_jobs,
    mark_job_run,
    save_jobs,
    trigger_job,
)


@pytest.fixture()
def clock(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    state = {"now": datetime.fromisoformat("2026-10-28T09:00:00-07:00")}
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: state["now"])
    return state


def _fire(job_id, **claim_kwargs):
    claimed = claim_job_for_fire(job_id, return_job=True, **claim_kwargs)
    assert isinstance(claimed, dict), "fire was not claimed"
    assert claim_dispatch(job_id)
    assert mark_job_run(job_id, success=True, expected_fire_owner=claimed["fire_claim"]["by"])


def test_manual_run_keeps_the_scheduled_oneshot_and_it_still_fires(clock):
    job = create_job("remind me", "in 30m", name="reminder")
    slot = job["next_run_at"]

    _fire(job["id"], manual=True)  # cronjob(action="run") / hermes cron run

    after = get_job(job["id"])
    assert (after["state"], after["enabled"], after["next_run_at"]) == ("scheduled", True, slot)
    assert after["repeat"] == {"times": 1, "completed": 0}
    assert after["last_status"] == "ok"

    clock["now"] += timedelta(minutes=30, seconds=5)
    assert [d["id"] for d in get_due_jobs()] == [job["id"]]
    _fire(job["id"])  # the scheduled tick

    done = get_job(job["id"])
    assert (done["state"], done["next_run_at"]) == ("completed", None)
    assert done["repeat"] == {"times": 1, "completed": 1}


def test_gateway_forwarded_trigger_restores_the_slot_at_claim_time(clock):
    """trigger_job overwrites next_run_at with "now"; the claim puts the slot back in the same
    save, so a crash before mark_job_run cannot lose the scheduled occurrence."""
    job = create_job("remind me", "in 30m", name="reminder")
    slot = job["next_run_at"]
    trigger_job(job["id"])
    assert [d["id"] for d in get_due_jobs()] == [job["id"]]

    claimed = claim_job_for_fire(job["id"], return_job=True)  # the ticker's claim
    assert isinstance(claimed, dict)
    assert get_job(job["id"])["next_run_at"] == slot  # durable before the run starts

    assert claim_dispatch(job["id"])
    assert mark_job_run(job["id"], success=True, expected_fire_owner=claimed["fire_claim"]["by"])
    after = get_job(job["id"])
    assert (after["state"], after["next_run_at"]) == ("scheduled", slot)
    assert after["repeat"]["completed"] == 0
    assert "manual_run_restore_at" not in after


def test_second_trigger_before_the_claim_keeps_the_original_slot(clock):
    """Two run-now requests queued before the ticker claims: the saved slot is the job's real
    scheduled time, not the first trigger's "now"."""
    job = create_job("remind me", "in 30m", name="reminder")
    slot = job["next_run_at"]
    trigger_job(job["id"])
    clock["now"] += timedelta(seconds=20)
    trigger_job(job["id"])

    _fire(job["id"])  # the ticker's claim of the queued manual run

    after = get_job(job["id"])
    assert (after["state"], after["next_run_at"]) == ("scheduled", slot)
    assert after["repeat"] == {"times": 1, "completed": 0}


def test_forwarded_recurring_run_survives_the_tick_advance(clock):
    """The gateway-forwarded run goes through the tick: get_due_jobs, then advance_next_runs
    for the whole due set, THEN the worker's plain claim. The queued manual intent must survive
    that advance, so the run spends no repeat budget and the schedule does not shift."""
    job = create_job("digest", "every 60m", name="digest", repeat=1)
    slot = job["next_run_at"]
    clock["now"] += timedelta(minutes=20)
    trigger_job(job["id"])

    due = get_due_jobs()
    assert [d["id"] for d in due] == [job["id"]]
    advance_next_runs([d["id"] for d in due])
    _fire(job["id"])  # the worker: claim_job_for_fire with neither manual nor force

    after = get_job(job["id"])
    assert (after["state"], after["next_run_at"]) == ("scheduled", slot)
    assert after["repeat"] == {"times": 1, "completed": 0}


def test_manual_run_does_not_spend_a_recurring_repeat_budget_or_shift_the_schedule(clock):
    job = create_job("digest", "every 60m", name="digest", repeat=2)
    slot = job["next_run_at"]
    clock["now"] += timedelta(minutes=20)

    _fire(job["id"], manual=True)

    after = get_job(job["id"])
    assert after["repeat"] == {"times": 2, "completed": 0}
    assert after["next_run_at"] == slot


def test_crash_mid_manual_run_leaves_the_scheduled_oneshot_due(clock):
    job = create_job("remind me", "in 30m", name="reminder")
    assert isinstance(claim_job_for_fire(job["id"], manual=True, return_job=True), dict)
    assert claim_dispatch(job["id"])
    # The runner dies here: no mark_job_run, and its claim's owner is gone.
    record = load_jobs()[0]
    record["fire_claim"]["by"] = "dead-host:0"
    record["fire_claim"]["at"] = (clock["now"] - timedelta(hours=2)).isoformat()
    save_jobs([record])

    clock["now"] += timedelta(minutes=30, seconds=5)
    assert [d["id"] for d in get_due_jobs()] == [job["id"]]
    assert get_job(job["id"])["repeat"]["completed"] == 0


def test_run_at_or_after_the_due_slot_is_the_scheduled_fire(clock):
    """Control: a manual run once the one-shot is already due consumes it (no double fire)."""
    job = create_job("remind me", "in 2m", name="reminder")
    clock["now"] += timedelta(minutes=2, seconds=10)

    _fire(job["id"], manual=True)

    done = get_job(job["id"])
    assert (done["state"], done["repeat"]["completed"]) == ("completed", 1)


def test_trigger_landing_during_a_manual_run_survives_its_completion(clock):
    """A run-now queued while a manual run is in flight is its own extra fire: the in-flight run's
    completion must not clear it, and firing it must not consume the one-shot."""
    job = create_job("remind me", "in 30m", name="reminder")
    slot = job["next_run_at"]
    claimed = claim_job_for_fire(job["id"], manual=True, return_job=True)
    assert isinstance(claimed, dict) and claim_dispatch(job["id"])

    clock["now"] += timedelta(seconds=10)
    trigger_job(job["id"])  # gateway-forwarded run-now while the first run is executing
    clock["now"] += timedelta(seconds=10)
    assert heartbeat_fire_claim(job["id"], expected_owner=claimed["fire_claim"]["by"])  # still running
    assert mark_job_run(job["id"], success=True, expected_fire_owner=claimed["fire_claim"]["by"])

    assert [d["id"] for d in get_due_jobs()] == [job["id"]]  # the queued run-now is still due
    _fire(job["id"])  # the ticker picks it up
    after = get_job(job["id"])
    assert (after["state"], after["next_run_at"]) == ("scheduled", slot)
    assert after["repeat"] == {"times": 1, "completed": 0}



def test_quota_hold_on_a_manual_run_delays_a_queued_run_now_without_spending_the_schedule(
    clock, monkeypatch,
):
    """A manual run hits a provider quota window while another run-now is queued: the hold delays
    that queued run-now to the window boundary, where it still fires as an extra run. It spends no
    repeat run and the scheduled slot after the window stays put."""
    monkeypatch.setattr("cron.quota_hold._hermes_now", lambda: clock["now"])
    job = create_job("digest", "every 60m", name="digest", repeat=3)
    slot = job["next_run_at"]  # 10:00
    clock["now"] += timedelta(minutes=20)
    claimed = claim_job_for_fire(job["id"], manual=True, return_job=True)
    assert isinstance(claimed, dict) and claim_dispatch(job["id"])
    clock["now"] += timedelta(seconds=10)
    trigger_job(job["id"])  # queued while the manual run is in flight
    clock["now"] += timedelta(seconds=10)
    assert mark_job_run(job["id"], success=False, error="quota", quota_hold_seconds=600,
                        expected_fire_owner=claimed["fire_claim"]["by"])

    held = get_job(job["id"])
    assert held["next_run_at"] > clock["now"].isoformat()  # parked through the closed window
    assert get_due_jobs() == []
    clock["now"] = datetime.fromisoformat(held["next_run_at"]) + timedelta(seconds=5)  # ~09:31
    due = get_due_jobs()
    assert [d["id"] for d in due] == [job["id"]]
    advance_next_runs([d["id"] for d in due])
    _fire(job["id"])  # the delayed run-now

    after = get_job(job["id"])
    assert (after["next_run_at"], after["repeat"]) == (slot, {"times": 3, "completed": 0})
