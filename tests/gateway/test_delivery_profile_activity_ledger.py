"""Production-seam contract tests for gateway profile-activity events."""


def test_delivery_activity_uses_ledger_contract_fields(monkeypatch, tmp_path):
    from gateway.config import Platform
    from gateway.delivery import DeliveryRouter, DeliveryTarget
    from hermes_cli import profile_activity_ledger as ledger
    from hermes_cli import profiles

    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    # R2-6: ledger path helpers now take an optional explicit_home argument.
    monkeypatch.setattr(ledger, "ledger_db_path", lambda explicit_home=None: tmp_path / "activity.sqlite")
    monkeypatch.setattr(ledger, "ledger_jsonl_dir", lambda explicit_home=None: tmp_path / "logboard")
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "kensei")

    router = object.__new__(DeliveryRouter)
    router._record_delivery_activity(
        DeliveryTarget(platform=Platform.TELEGRAM, chat_id="123", is_explicit=True),
        False,
        "job-42",
        "Nightly reconciliation",
        {"request_id": "req-7"},
        error="adapter unavailable",
    )

    events = ledger.query_events(event_types=["delivery_error"])
    assert len(events) == 1
    event = events[0]
    assert event["actor_profile"] == "kensei"
    assert event["payload"]["severity"] == "error"
    assert event["payload"]["correlation_id"] == "job-42"
    assert event["payload"]["error"] == "adapter unavailable"
