"""A stopped cron run stays stopped in every reader of the ledger.

Real store, real execution ledger, real cancel markers under the test HERMES_HOME: a cancel must not
be replayed by receipt recovery, read as a crashed worker, counted as a failure, spend an extra
repeat, or hide behind the due scan's ``missed`` rows.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest

from cron import executions, jobs, scheduler, scheduler_authority


@pytest.fixture
def store(tmp_path):
    with jobs.use_cron_store(tmp_path / "cron"):
        yield tmp_path


def _cancelled_run(job: dict) -> str:
    from cron.cancellation import finish_cancelled_run, request_cancel

    fire = executions.create_execution(job["id"], source="test")["id"]
    executions.mark_execution_running(fire)
    request_cancel(fire, reason="operator asked")
    finish_cancelled_run(dict(jobs.get_job(job["id"])), fire, None)
    return fire


def test_receipt_recovery_does_not_replay_a_cancelled_agent_run(store, monkeypatch):
    from hermes_cli import gateway_client

    calls: list = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            return {"status": "terminal", "job": job, "result": [False, "doc", "", "stopped"]}

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
    fire = _cancelled_run(job)
    journal = scheduler_authority.journal_path(job["id"], fire)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({
        "params": {"job_id": job["id"], "request_id": fire, "extra_prompt": None},
        "receipt": {"session_id": "s", "admission_id": "a"}}))
    booked = jobs.get_job(job["id"])

    scheduler_authority.reconcile_pending()

    after = jobs.get_job(job["id"])
    assert calls == []
    assert (after["last_status"], after["repeat"], after.get("failure_streak")) == (
        booked["last_status"], booked["repeat"], booked.get("failure_streak"))
    assert executions.get_execution(fire)["status"] == "cancelled"
    assert not journal.exists()


def test_the_external_worker_waiter_accepts_a_cancelled_row(store):
    """The detached worker books ``cancelled`` and exits 0; that is a finished attempt, not a
    worker that died without terminalizing."""
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
    fire = _cancelled_run(job)

    class Exited:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    assert scheduler._wait_for_external_cron_worker_body(
        Exited(), execution_id=fire, stderr_path=None) is True


def test_a_missed_row_does_not_hide_a_live_run(store):
    job = jobs.create_job(prompt="long", schedule="every 1h", deliver="local")
    live = executions.create_execution(job["id"], source="test")["id"]
    executions.mark_execution_running(live)
    executions.record_missed_occurrences(job["id"], count=3, first_instant=None, detail="slept")

    assert executions.live_inflight_execution(job["id"])["id"] == live


def test_stopping_a_ladder_rerun_does_not_spend_another_repeat(store):
    from cron.cancellation import finish_cancelled_run, request_cancel
    from cron.unreachable_retry import STATE_KEY

    job = jobs.create_job(prompt="ping", schedule="every 1h", deliver="local", repeat=2)
    stored = jobs.load_jobs()
    stored[0]["repeat"]["completed"] = 1  # the original occurrence already counted
    jobs.save_jobs(stored)
    instant = jobs.get_job(job["id"])["next_run_at"]
    rerun = dict(jobs.get_job(job["id"]), _scheduled_instant=instant,
                 **{STATE_KEY: {"at": instant, "expr": job["schedule"].get("expr")}})
    fire = executions.create_execution(job["id"], source="test")["id"]
    request_cancel(fire)

    finish_cancelled_run(rerun, fire, None)

    after = jobs.get_job(job["id"])
    assert after["repeat"]["completed"] == 1
    assert after["enabled"]


def test_removal_before_the_run_starts_books_cancelled_not_claim_loss(store, monkeypatch):
    """remove_job writes the marker and drops the claim; the worker's first claim check must book
    the operator's stop, not "Fire claim ownership lost before execution started"."""
    from cron.cancellation import request_cancel

    job = jobs.create_job(prompt="long", schedule="every 1h", deliver="local")
    fire = executions.create_execution(job["id"], source="test")["id"]
    request_cancel(fire, reason="job removed while running")
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *a, **k: False)
    ran: list = []

    claimed = dict(job, execution_id=fire, fire_claim={"by": "me", "at": "now"})
    assert scheduler._run_with_fire_claim_heartbeat(claimed, ran.append) is True

    assert ran == []
    row = executions.get_execution(fire)
    assert (row["status"], row["error"]) == ("cancelled", "job removed while running")


def test_overlapping_cancel_requests_keep_the_first_reason(store, monkeypatch):
    """Both callers can pass the existence check before either publishes its marker."""
    from pathlib import Path

    from cron.cancellation import CancelRequest, request_cancel

    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda self: False if self.name == "exec-race.cancel" else real_exists(self))

    request_cancel("exec-race", reason="first ask")
    request_cancel("exec-race", reason="second ask")

    assert CancelRequest("exec-race").reason == "first ask"


def test_monitoring_reports_stops_and_misses_as_themselves(monkeypatch):
    from agent.monitoring.cron_health import project_execution_event
    from hermes_cli.observability import shared_metrics_gateway as metrics

    assert project_execution_event({"status": "cancelled", "job_id": "j"}).status == "cancelled"
    assert project_execution_event({"status": "missed", "job_id": "j"}).status == "missed"

    sent: list = []
    monkeypatch.setattr(metrics, "_submit", lambda emit, mark, fields, **kw: sent.append(fields(**kw)))
    metrics.record_cron_finish({"id": "e1", "status": "cancelled", "started_at": "2026-01-01T00:00:00",
                                "finished_at": "2026-01-01T00:00:05"})
    assert sent[0]["outcome"] == "cancelled"


def test_a_cancelled_agent_run_names_its_reason_in_history():
    from hermes_cli.web_routers.cron import _reconcile_cron_runs

    session = {"id": "s1", "title": "Morning brief", "cron_execution_id": "ex1",
               "started_at": 100.0}
    attempts = [{"id": "ex1", "status": "cancelled", "error": "cancelled on request (Stop)",
                 "claimed_at": 99.0, "finished_at": 120.0}]

    rows = _reconcile_cron_runs([session], [], attempts, "job", 10)

    assert [r["title"] for r in rows] == ["CANCELLED · cancelled on request (Stop)"]


def test_a_capped_count_says_so_after_a_catch_up(store):
    job = jobs.create_job(prompt="tick", schedule="every 1m", deliver="local", catch_up=True)
    stored = jobs.load_jobs()
    stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(minutes=1100)).isoformat()
    jobs.save_jobs(stored)

    jobs.get_due_jobs()

    missed = [r for r in executions.list_executions(job_id=job["id"]) if r["status"] == "missed"]
    assert len(missed) == 1
    assert "counting stopped at 1000" in missed[0]["error"]


def test_the_job_list_still_shows_a_live_run_behind_a_refused_trigger(store):
    """Desktop's Stop reads the job list: a refused Run-now's newer row must not hide the live run."""
    job = jobs.create_job(prompt="long", schedule="every 1h", deliver="local")
    live = executions.create_execution(job["id"], source="test")["id"]
    executions.mark_execution_running(live)
    refused = executions.create_execution(job["id"], source="test")["id"]
    executions.finish_execution(refused, success=False, error="Fire claim was not acquired")
    with executions._transaction() as conn:  # the refused row is the job's newest attempt
        conn.execute("UPDATE executions SET claimed_at='2999-01-01T00:00:00+00:00' WHERE id=?", (refused,))

    [listed] = jobs.list_jobs(include_disabled=True)
    assert listed["latest_execution"]["id"] == refused
    assert listed["live_execution"]["id"] == live

    executions.finish_execution(live, success=True)
    assert jobs.list_jobs(include_disabled=True)[0]["live_execution"] is None
