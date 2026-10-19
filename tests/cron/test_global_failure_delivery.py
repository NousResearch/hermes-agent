"""Fleet failure routing without changing any job's success delivery or job record."""

import pytest

from cron import scheduler
from cron import scheduler_delivery as delivery
from cron import scheduler_preflight as preflight
from cron.incidents import has_alerted_incident_for_job, set_incident_state


@pytest.fixture
def fleet_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("cron:\n  failure_deliver: telegram\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_fleet_failure_route_overrides_local_and_job_opt_out(fleet_home, monkeypatch):
    monkeypatch.setattr(delivery, "_get_home_target_chat_id", lambda p: "home-123")
    for job in (
        {"id": "local", "deliver": "local"},
        {"id": "explicit", "deliver": "slack:D1", "failure_deliver": "local"},
    ):
        assert delivery._delivery_lane_value(job, for_failure=True) == "telegram"
        assert [(t["platform"], t["chat_id"]) for t in delivery._resolve_delivery_targets(
            job, for_failure=True
        )] == [("telegram", "home-123")]
        assert delivery._delivery_lane_value(job) == job["deliver"]
        assert delivery._resolve_delivery_targets(job) == (
            [] if job["deliver"] == "local" else delivery._resolve_delivery_targets(
                {"deliver": job["deliver"]})
        )


def test_fleet_failure_route_is_checked_by_preflight(fleet_home, monkeypatch):
    monkeypatch.setattr(preflight._delivery, "_is_known_delivery_platform", lambda p: p == "telegram")

    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: type("GatewayConfig", (), {"get_connected_platforms": lambda self: []})(),
    )
    assert preflight._preflight_check_delivery({"id": "local", "deliver": "local"}) is not None


def test_one_alert_per_job_across_errors_until_success(fleet_home):
    job = {"id": "fleet-canary", "name": "fleet-canary", "deliver": "local"}
    withheld, first = scheduler._upsert_incident_for_failure(job, "provider error")
    assert not withheld and first
    assert set_incident_state(first, "alerted")
    assert has_alerted_incident_for_job(job["id"])
    withheld, second = scheduler._upsert_incident_for_failure(job, "different provider error")
    assert withheld and second != first
    assert scheduler._upsert_incident_for_failure(job, "provider error")[0]
    scheduler._resolve_incidents_for_recovered_job(job)
    assert not has_alerted_incident_for_job(job["id"])
    assert not scheduler._upsert_incident_for_failure(job, "provider error")[0]
