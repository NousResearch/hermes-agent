from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _metric(snapshot, name):
    return next(metric for metric in snapshot.metrics if metric.name == name)


def test_cron_snapshot_projects_freshness_counts_and_overdue_without_content(monkeypatch):
    from agent.monitoring import cron_health

    now = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    secret = "Quarterly payroll for alice@example.com"
    monkeypatch.setattr(cron_health, "_now", lambda: now)
    monkeypatch.setattr(cron_health, "get_ticker_heartbeat_age", lambda: 4.5)
    monkeypatch.setattr(cron_health, "get_ticker_success_age", lambda: 9.0)
    monkeypatch.setattr(cron_health, "get_running_job_ids", lambda: frozenset({"job-private-1"}))
    monkeypatch.setattr(
        cron_health,
        "load_jobs",
        lambda: [
            {
                "id": "job-private-1",
                "name": secret,
                "prompt": secret,
                "enabled": True,
                "schedule": {"kind": "interval", "minutes": 10},
                "next_run_at": (now - timedelta(minutes=6)).isoformat(),
            },
            {
                "id": "job-private-2",
                "name": "disabled private job",
                "enabled": False,
                "schedule": {"kind": "interval", "minutes": 10},
                "next_run_at": (now - timedelta(days=1)).isoformat(),
            },
        ],
    )

    snapshot = cron_health.build_cron_health_snapshot()

    assert _metric(snapshot, "hermes.cron.scheduler.heartbeat_age_seconds").value == 4.5
    assert _metric(snapshot, "hermes.cron.scheduler.last_success_age_seconds").value == 9.0
    assert _metric(snapshot, "hermes.cron.jobs.enabled").value == 1
    assert _metric(snapshot, "hermes.cron.jobs.running").value == 1
    assert _metric(snapshot, "hermes.cron.jobs.overdue").value == 1
    assert secret not in str(snapshot)
    assert "job-private-1" not in str(snapshot)


def test_cron_snapshot_omits_unknown_freshness_instead_of_inventing_values(monkeypatch):
    from agent.monitoring import cron_health

    monkeypatch.setattr(cron_health, "get_ticker_heartbeat_age", lambda: None)
    monkeypatch.setattr(cron_health, "get_ticker_success_age", lambda: None)
    monkeypatch.setattr(cron_health, "get_running_job_ids", lambda: frozenset())
    monkeypatch.setattr(cron_health, "load_jobs", lambda: [])

    names = {metric.name for metric in cron_health.build_cron_health_snapshot().metrics}

    assert "hermes.cron.scheduler.heartbeat_age_seconds" not in names
    assert "hermes.cron.scheduler.last_success_age_seconds" not in names


def test_execution_projection_is_opaque_bounded_and_content_free():
    from agent.monitoring.cron_health import project_execution_event

    event = project_execution_event(
        {
            "id": "execution-private-id",
            "job_id": "Payroll for alice@example.com and token top-secret-token",
            "source": "builtin",
            "status": "failed",
            "claimed_at": "2026-07-24T12:00:00+00:00",
            "started_at": "2026-07-24T12:00:01+00:00",
            "finished_at": "2026-07-24T12:00:03.250000+00:00",
            "error": "Bearer top-secret-token rejected for alice@example.com",
        },
        delivery_outcome="failed",
    ).to_dict()

    assert event["event"] == "cron_execution"
    assert event["status"] == "failed"
    assert event["job_key"].startswith("sha256:")
    assert len(event["job_key"]) == len("sha256:") + 24
    assert event["duration_ms"] == 2250
    assert event["delivery_outcome"] == "failed"
    assert event["error_class"] == "auth_failed"
    assert "job_id" not in event
    assert "error" not in event
    assert "alice@example.com" not in str(event)
    assert "top-secret-token" not in str(event)


def test_execution_projection_omits_duration_and_delivery_when_not_known():
    from agent.monitoring.cron_health import project_execution_event

    event = project_execution_event(
        {
            "job_id": "private",
            "source": "external-value-must-not-leak",
            "status": "claimed",
            "claimed_at": "2026-07-24T12:00:00+00:00",
        }
    ).to_dict()

    assert event["status"] == "claimed"
    assert event["source"] == "external"
    assert event["duration_ms"] is None
    assert event["delivery_outcome"] is None


def test_external_provider_source_is_normalized_to_external():
    from agent.monitoring.cron_health import project_execution_event

    event = project_execution_event(
        {"job_id": "private", "source": "Chronos", "status": "claimed"}
    )

    assert event.source == "external"


@pytest.mark.parametrize("message", ["oauth refresh failed", "tokenizer crashed", "HTTP 4015"])
def test_error_classification_avoids_auth_substring_false_positives(message):
    from agent.monitoring.cron_health import classify_cron_error

    assert classify_cron_error(message) == "unknown"


@pytest.mark.parametrize(
    "message",
    ["authentication failed", "not authorized", "access token expired", "HTTP 401"],
)
def test_error_classification_recognizes_auth_terms_and_status_tokens(message):
    from agent.monitoring.cron_health import classify_cron_error

    assert classify_cron_error(message) == "auth_failed"


def test_cron_snapshot_exports_catch_up_occurrence_counter(monkeypatch):
    from agent.monitoring import cron_health

    monkeypatch.setattr(cron_health, "get_ticker_heartbeat_age", lambda: None)
    monkeypatch.setattr(cron_health, "get_ticker_success_age", lambda: None)
    monkeypatch.setattr(cron_health, "get_running_job_ids", lambda: frozenset())
    monkeypatch.setattr(cron_health, "load_jobs", lambda: [])
    monkeypatch.setattr(cron_health, "get_catch_up_occurrence_count", lambda: 3)

    snapshot = cron_health.build_cron_health_snapshot()

    assert _metric(snapshot, "hermes.cron.scheduler.catch_up_occurrences").value == 3


def test_terminal_execution_emission_flushes_and_failures_are_fail_open(monkeypatch):
    from agent.monitoring import cron_health, emitter

    calls = []

    class FakeEmitter:
        def emit(self, event):
            calls.append(("emit", event.to_dict()["status"]))

        def flush(self, timeout):
            calls.append(("flush", timeout))
            raise RuntimeError("collector unavailable")

    monkeypatch.setattr(emitter, "get_emitter", lambda: FakeEmitter())

    cron_health.emit_execution_state(
        {"job_id": "private", "source": "builtin", "status": "completed"}
    )

    assert calls == [("emit", "completed"), ("flush", 1.0)]


def test_gateway_export_includes_cron_metrics_and_only_accepted_event_planes(monkeypatch):
    from agent.monitoring import gateway_health_export

    gateway_snapshot = type("Snapshot", (), {"metrics": []})()
    cron_snapshot = type(
        "Snapshot",
        (),
        {"metrics": [type("Metric", (), {"name": "hermes.cron.jobs.enabled", "value": 2, "attributes": {}})()]},
    )()
    monkeypatch.setattr(gateway_health_export, "_read_gateway_snapshot", lambda config: gateway_snapshot)
    monkeypatch.setattr(gateway_health_export, "_read_cron_snapshot", lambda: cron_snapshot)

    snapshot = gateway_health_export._read_runtime_snapshot({})

    assert [metric.name for metric in snapshot.metrics] == ["hermes.cron.jobs.enabled"]
    assert gateway_health_export._gateway_health_event({"event": "cron_execution"}) is True
    assert gateway_health_export._gateway_health_event({"event": "gateway_health"}) is True
    assert gateway_health_export._gateway_health_event({"event": "run"}) is False


def test_monitoring_docs_distinguish_relay_health_scope_and_terminal_flush():
    from pathlib import Path

    text = Path("docs/observability/monitoring.md").read_text(encoding="utf-8")

    assert "Hermes Agent-owned Relay transport health" in text
    assert "authoritative shared connector/platform state" in text
    assert "up to one second" in text
    assert "terminal" in text
