"""Production-seam contract tests for cron profile-activity events."""


def test_cron_activity_uses_ledger_contract_fields(monkeypatch, tmp_path):
    from cron.jobs import _record_cron_activity
    from hermes_cli import profile_activity_ledger as ledger

    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    # R2-6: ledger path helpers now take an optional explicit_home argument.
    monkeypatch.setattr(ledger, "ledger_db_path", lambda explicit_home=None: tmp_path / "activity.sqlite")
    monkeypatch.setattr(ledger, "ledger_jsonl_dir", lambda explicit_home=None: tmp_path / "logboard")

    _record_cron_activity(
        "job_run_error",
        {
            "id": "job-42",
            "name": "Nightly reconciliation",
            "profile": "kensei",
            "schedule_display": "every 1h",
        },
        idempotency_key="cron:job-42:run:fixed",
        delivery_error="adapter unavailable",
    )

    events = ledger.query_events(event_types=["job_run_error"])
    assert len(events) == 1
    event = events[0]
    assert event["event_id"] == "cron:job-42:run:fixed"
    assert event["actor_profile"] == "kensei"
    assert event["payload"]["severity"] == "error"
    assert event["payload"]["correlation_id"] == "job-42"
    assert event["payload"]["delivery_error"] == "adapter unavailable"
