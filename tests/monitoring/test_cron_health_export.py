from __future__ import annotations

import pytest

def _metric(snapshot, name):
    return next(metric for metric in snapshot.metrics if metric.name == name)

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

def test_cron_store_writability_is_exported(monkeypatch, tmp_path):
    """A degraded store exports writable=0 with its skipped runs, a cleared one writable=1."""
    import errno
    from agent.monitoring.cron_health import build_cron_health_snapshot
    from cron import store_health
    monkeypatch.setattr(store_health, "_degraded", {})
    monkeypatch.setattr(store_health, "_listener", None)
    monkeypatch.setattr(store_health, "_recovered", {})
    from cron import scheduler_ownership
    monkeypatch.setattr(scheduler_ownership, "_ticked_homes", {})
    store_health.note_unwritable(OSError(errno.ENOSPC, "full"), "x", "advance",
                                 [{"id": "job", "next_run_at": "2026-06-22T12:00:00+00:00"}])
    degraded = build_cron_health_snapshot()
    assert (_metric(degraded, "hermes.cron.store.writable").value,
            _metric(degraded, "hermes.cron.store.skipped_runs").value) == (0, 1)
    store_health._degraded.clear()
    assert _metric(build_cron_health_snapshot(), "hermes.cron.store.writable").value == 1
    # A secondary profile's store (not the exporter's scope) must not read as healthy.
    from cron.scheduler_ownership import register_ticked_homes
    register_ticked_homes([tmp_path / "other"])
    store_health.note_unwritable(OSError(errno.ENOSPC, "full"), "x", "scan", cron_dir=tmp_path / "other" / "cron")
    assert _metric(build_cron_health_snapshot(), "hermes.cron.store.writable").value == 0
    register_ticked_homes([])  # the profile left this gateway: its store no longer counts
    assert _metric(build_cron_health_snapshot(), "hermes.cron.store.writable").value == 1

@pytest.mark.parametrize("message", ["oauth refresh failed", "tokenizer crashed", "HTTP 4015"])
def test_error_classification_avoids_auth_substring_false_positives(message):
    from agent.monitoring.cron_health import classify_cron_error

    assert classify_cron_error(message) == "unknown"

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

def test_registered_observable_metric_names_cover_snapshot_metrics(monkeypatch):
    """Every gauge emitted in the runtime snapshot must also be registered in the
    observable-gauge metric_names list, or the OTLP exporter never observes it.

    This asserts the vocabulary-registration invariant documented in
    website/docs/developer-guide/gateway-monitoring.md: an emitted-but-unregistered gauge is
    silently dropped. Regression guard for background_work / cron additions.
    """
    from agent.monitoring import gateway_health_export

    # Build a representative snapshot (gateway + cron + background_work) without
    # a live gateway by stubbing the gateway snapshot to the real metric names.
    class _M:
        def __init__(self, name):
            self.name = name
            self.value = 0
            self.attributes = {}

    gateway_snapshot = type("S", (), {"metrics": [
        _M("hermes.gateway.up"), _M("hermes.gateway.active_agents"),
        _M("hermes.gateway.busy"), _M("hermes.gateway.drainable"),
        _M("hermes.gateway.restart_requested"),
        _M("hermes.platform.up"), _M("hermes.platform.degraded"),
    ]})()
    cron_snapshot = type("S", (), {"metrics": [
        _M("hermes.cron.scheduler.heartbeat_age_seconds"),
        _M("hermes.cron.scheduler.last_success_age_seconds"),
        _M("hermes.cron.scheduler.catch_up_occurrences"),
        _M("hermes.cron.jobs.enabled"), _M("hermes.cron.jobs.running"),
        _M("hermes.cron.jobs.overdue"),
        _M("hermes.cron.store.writable"), _M("hermes.cron.store.skipped_runs"),
    ]})()
    monkeypatch.setattr(gateway_health_export, "_read_gateway_snapshot", lambda config: gateway_snapshot)
    monkeypatch.setattr(gateway_health_export, "_read_cron_snapshot", lambda: cron_snapshot)

    snapshot_names = {m.name for m in gateway_health_export._read_runtime_snapshot({}).metrics}

    registered = set(gateway_health_export._OBSERVABLE_METRIC_NAMES)

    missing = snapshot_names - registered
    assert not missing, f"gauges emitted but NOT registered in metric_names (will be silently dropped): {sorted(missing)}"
