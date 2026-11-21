"""A successful run whose ``deliver='origin'`` resolves to no target must not read as ``ok``.

#43014 keeps such a run a non-failure — a CLI/script-created job legitimately has no captured
origin and the machine may have no home channel — but the record used to say ``ok``, which reads as
"the user was told", while the output only ever reached ``last_output``. The status now says what
happened, and the CLI renders it as a warning instead of green.
"""

from datetime import timedelta

from cron import scheduler as sched
from cron.jobs import create_job, get_job
from cron.scheduler_bookkeeping import _classify_delivery_outcome, finish_completed_run
from hermes_time import now


def test_classify_keeps_origin_without_target_non_fatal():
    assert _classify_delivery_outcome(
        delivery_error=None,
        should_deliver=True,
        unresolved_origin=True,
        normalized_deliver="origin",
        incident_acked=False,
        success=True,
    ) == "not_configured"


def _spy_mark(monkeypatch):
    real_mark = sched.mark_job_run
    seen: dict = {}

    def spy(job_id, success, error=None, **kwargs):
        seen.update(kwargs)
        seen["success"] = success
        return real_mark(job_id, success, error, **kwargs)

    monkeypatch.setattr(sched, "mark_job_run", spy)
    monkeypatch.setattr(sched, "finish_execution", lambda *a, **k: True)
    return seen


def test_origin_without_target_records_delivery_not_configured(monkeypatch):
    seen = _spy_mark(monkeypatch)
    job = create_job("origin job", "every 24h", deliver="origin")
    run = dict(job, execution_id="exec-origin-1")
    delivery = sched._RunDelivery(job=run, success=True, error=None)
    delivery.should_deliver = True
    delivery.unresolved_origin = True

    assert finish_completed_run(delivery, None, "exec-origin-1") is True

    assert seen["status"] == "delivery_not_configured"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_status"] == "delivery_not_configured"
    # Still a non-failure: the agent run itself succeeded and no delivery error is claimed.
    assert not stored.get("last_delivery_error")
    assert stored.get("last_error") in (None, "")


def test_run_with_a_resolved_target_still_reports_ok(monkeypatch):
    seen = _spy_mark(monkeypatch)
    job = create_job("targeted job", "every 24h", deliver="telegram:42")
    run = dict(job, execution_id="exec-target-1")
    delivery = sched._RunDelivery(job=run, success=True, error=None)
    delivery.should_deliver = True
    delivery.unresolved_origin = False

    assert finish_completed_run(delivery, None, "exec-target-1") is True

    assert "status" not in seen
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_status"] == "ok"


def test_last_run_display_warns_instead_of_green():
    from hermes_cli.cron import _last_run_display

    text = _last_run_display({"last_status": "delivery_not_configured", "deliver": "origin",
                              "id": "abc123"})
    plain = text.replace("\x1b[33m", "").replace("\x1b[0m", "")
    assert "nothing was delivered" in plain
    assert "cron edit abc123" in plain


def test_doctor_does_not_report_it_as_a_failed_run():
    from hermes_cli.cron import _cron_doctor_issues_for_job

    job = {
        "id": "abc123",
        "name": "origin job",
        "deliver": "origin",
        "last_status": "delivery_not_configured",
        "last_error": None,
        "enabled": True,
        "state": "scheduled",
        "next_run_at": (now() + timedelta(hours=1)).isoformat(),
        "repeat": {"times": None, "completed": 1},
    }
    issues = _cron_doctor_issues_for_job(job)
    assert not any(i.startswith("last run failed") for i in issues)
    assert any("delivered nowhere" in i for i in issues)
