"""Delivery failures get incident parity, keyed by lane (#112712 lever 1).

A failed delivery (of a successful result or of a failure notice) opens a lane-keyed
``failure_type='delivery'`` incident through the same store the fire errors use; a clean delivery
resolves it; an operator ack survives repeats; the signature is the LANE, not the raw error text,
so run-to-run error variance cannot mint new incidents. The compose-phase recovery close skips
delivery rows (it runs before this run's delivery outcome is known); only a clean delivery
resolves them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.executions as executions
import cron.incidents as incidents
import cron.scheduler as sched
from cron.incidents import DELIVERY_FAILURE_TYPE
from cron.scheduler import _RunDelivery


def _point_db(monkeypatch, tmp_path):
    # cron.executions.EXECUTIONS_FILE takes precedence over incidents.EXECUTIONS_FILE; patch both.
    path = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", path)
    return incidents


JOB = {"id": "job-1", "name": "volantini"}


def _delivery_rows(inc, job_id="job-1"):
    return [r for r in inc.list_incidents() if r["job_id"] == job_id
            and r.get("failure_type") == DELIVERY_FAILURE_TYPE]


def test_lane_keyed_signature_dedups_across_varying_error_texts(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    sched._record_delivery_incident(JOB, "telegram:-1003701271596:234", "Chat not found")
    sched._record_delivery_incident(JOB, "telegram:-1003701271596:234", "retry after 31")
    sched._record_delivery_incident(JOB, "telegram:-1003701271596:234",
                                    "httpx.ReadTimeout on bot-api:443, receipt=sha256:9f2c")

    rows = _delivery_rows(inc)
    assert len(rows) == 1  # the lane is the signature; error variance dedups
    assert rows[0]["state"] == "detected"


def test_delivery_failure_type_survives_timeout_and_rate_limit_texts(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)

    for text in ("Read timed out", "429 Too Many Requests", "401 Unauthorized"):
        sched._record_delivery_incident(JOB, "discord:123", text)

    rows = _delivery_rows(inc)
    assert len(rows) == 1  # same lane
    assert rows[0]["failure_type"] == DELIVERY_FAILURE_TYPE  # never classified timeout/auth/rate_limit


def test_clean_delivery_resolves_and_an_ack_survives_repeats(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    acked_job = {"id": "job-2", "name": "acked"}
    sched._record_delivery_incident(JOB, "telegram:1", "Chat not found")

    sched._record_delivery_incident(JOB, None, None)
    assert _delivery_rows(inc) and _delivery_rows(inc)[0]["state"] == "resolved"

    sched._record_delivery_incident(acked_job, "telegram:2", "Chat not found")
    acked_id = _delivery_rows(inc, "job-2")[0]["id"]
    inc.ack_incident(acked_id)
    sched._record_delivery_incident(acked_job, "telegram:2", "Chat not found again")
    assert inc.get_incident(acked_id)["state"] == "closed"


def test_compose_phase_recovery_close_skips_delivery_rows(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    fire_id, _ = inc.upsert_incident("job-1", "provider 503")
    sched._record_delivery_incident(JOB, "telegram:1", "Chat not found")

    content, *_ = sched._compose_run_delivery(
        JOB, success=True, error=None, final_response="all good", output_file=None)

    assert content == "all good"
    assert inc.get_incident(fire_id)["state"] == "resolved"  # fire rows close as before
    assert _delivery_rows(inc)[0]["state"] == "detected"  # delivery rows wait for the outcome


def _finish(monkeypatch, tmp_path, d, marked=True):
    inc = _point_db(monkeypatch, tmp_path)
    monkeypatch.setattr(sched, "self_removal_delivery_allowed", lambda _jid: False)
    monkeypatch.setattr(sched, "mark_job_run", lambda *a, **k: marked)
    finished = {}
    monkeypatch.setattr(sched, "finish_execution",
                        lambda eid, **k: finished.update(k) or finished.setdefault("eid", eid))
    ok = sched._finish_completed_run(d, None, "exec-1")
    return inc, ok, finished


def test_finish_completed_run_records_delivery_incident_on_success(monkeypatch, tmp_path):
    job = dict(JOB, deliver="telegram:-1003701271596:234")
    d = _RunDelivery(job=job, success=True, error=None, delivery_error="Chat not found",
                     should_deliver=True)

    inc, ok, finished = _finish(monkeypatch, tmp_path, d)

    assert ok is True and finished["delivery_outcome"] == "failed"
    rows = _delivery_rows(inc)
    assert len(rows) == 1 and "telegram:-1003701271596:234" in rows[0]["error"]


def test_finish_completed_run_records_delivery_incident_on_failed_notice(monkeypatch, tmp_path):
    job = dict(JOB, deliver="telegram:1", failure_deliver="telegram:2")
    d = _RunDelivery(job=job, success=False, error="provider 503",
                     delivery_error="Chat not found", should_deliver=True)

    inc, ok, _ = _finish(monkeypatch, tmp_path, d)

    assert ok is True
    assert len(_delivery_rows(inc)) == 1  # the failure lane's delivery failure is recorded too
    assert "telegram:2" in _delivery_rows(inc)[0]["error"]


def test_finish_completed_run_resolves_after_a_clean_delivery(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    sched._record_delivery_incident(JOB, "telegram:1", "Chat not found")
    job = dict(JOB, deliver="telegram:1")
    d = _RunDelivery(job=job, success=True, error=None, delivery_error=None, should_deliver=True)

    inc, ok, _ = _finish(monkeypatch, tmp_path, d)

    assert ok is True and _delivery_rows(inc)[0]["state"] == "resolved"


def test_claim_lost_before_tail_records_nothing(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    monkeypatch.setattr(sched, "self_removal_delivery_allowed", lambda _jid: False)
    monkeypatch.setattr(sched, "mark_job_run", lambda *a, **k: False)  # claim lost
    monkeypatch.setattr(sched, "finish_execution", lambda eid, **k: None)
    job = dict(JOB, deliver="telegram:1")
    d = _RunDelivery(job=job, success=True, error=None, delivery_error="Chat not found",
                     should_deliver=True)

    ok = sched._finish_completed_run(d, "owner-1", "exec-1")  # fire_owner set + not marked

    assert ok is True and _delivery_rows(inc) == []  # the early return skips the recording
