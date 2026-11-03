"""Previous-exit and memory-pressure gauges on the gateway monitoring plane (#148527)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from agent.monitoring import gateway_health_export, gateway_memory_health
from agent.monitoring.gateway_health import GatewayMetric
from gateway.lifecycle_ledger import get_lifecycle_sentinel_path
from gateway.shutdown_watchdog import get_loop_heartbeat_path

_BASE = {"service.instance.id": "sha256:abc", "service.version": "1.0", "hermes.supervision_mode": "systemd"}
_STARVED = {"prior_exit_reason": "loop_liveness_watchdog", "prior_host_memory_pressure": "critical"}
_SUSPECTED = {"prior_unclean_exit": True, "prior_suspected_oom": True}
# Each case also carries the next-weaker evidence, so a swapped precedence pair picks the wrong label.
_PRIOR_FLAGS = {
    "oom_kill_cgroup": {**_SUSPECTED, **_STARVED, "prior_oom_kill_seen": True, "prior_oom_kill_scope": "cgroup"},
    "watchdog_memory_starved": _STARVED,
    "watchdog": {"prior_exit_reason": "shutdown_watchdog", "prior_oom_kill_seen": True, "prior_oom_kill_scope": "host"},
    "oom_kill_host": {**_SUSPECTED, "prior_oom_kill_seen": True, "prior_oom_kill_scope": "host"},
    "suspected_oom": _SUSPECTED,
    "unclean": {"prior_unclean_exit": True},
    "none": {},
}


@pytest.fixture(autouse=True)
def _fresh_memo():
    gateway_memory_health._status_memo.clear()
    yield
    gateway_memory_health._status_memo.clear()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    def snapshot(config):  # fresh per call, like the real reader: the export extends its list
        return type("S", (), {"metrics": [GatewayMetric("hermes.gateway.up", 1, dict(_BASE))], "events": []})()

    monkeypatch.setattr(gateway_health_export, "_read_gateway_snapshot", snapshot)
    monkeypatch.setattr(gateway_health_export, "_read_cron_snapshot",
                        lambda: type("S", (), {"metrics": [], "events": []})())
    return home


def _claim(home: Path, prior: dict) -> None:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": "running", "pid": 1, "started_at": "2026-11-05T01:00:00+00:00", **prior}),
                    encoding="utf-8")


def _heartbeat(home: Path, *, available_kib: int, age_s: float = 5.0) -> None:
    path = get_loop_heartbeat_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": 1, "updated_at": (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat(), "monotonic": 1.0,
        "mem": {"rss_kib": 100_000, "mem_total_kib": 8_000_000, "mem_available_kib": available_kib},
    }), encoding="utf-8")


def _only(name: str) -> GatewayMetric:
    [metric] = [m for m in gateway_health_export._read_runtime_snapshot({}).metrics if m.name == name]
    return metric


@pytest.mark.parametrize("expected", list(_PRIOR_FLAGS))
def test_previous_exit_gauge_names_the_most_specific_verdict(home: Path, expected: str) -> None:
    _claim(home, _PRIOR_FLAGS[expected])

    metric = _only("hermes.gateway.previous_exit")

    assert metric.value == 1
    assert metric.attributes == {**_BASE, "hermes.previous_exit": expected}


def test_previous_exit_is_unknown_without_a_ledger_claim(home: Path) -> None:
    assert _only("hermes.gateway.previous_exit").attributes["hermes.previous_exit"] == "unknown"


@pytest.mark.parametrize(("available_kib", "age_s", "expected"), [
    pytest.param(4_000_000, 5.0, "ok", id="ok"),
    pytest.param(1_000_000, 5.0, "elevated", id="elevated"),
    pytest.param(32_000, 5.0, "critical", id="critical"),
    pytest.param(32_000, 600.0, "unknown", id="stale-heartbeat"),
])
def test_memory_pressure_gauge_is_the_live_heartbeat_level(home: Path, available_kib: int, age_s: float,
                                                           expected: str) -> None:
    _claim(home, {})
    _heartbeat(home, available_kib=available_kib, age_s=age_s)

    metric = _only("hermes.gateway.memory_pressure")

    assert metric.value == 1
    assert metric.attributes == {**_BASE, "hermes.memory_pressure": expected}


def test_memory_pressure_is_unknown_without_a_heartbeat(home: Path) -> None:
    assert _only("hermes.gateway.memory_pressure").attributes["hermes.memory_pressure"] == "unknown"


def test_a_failing_reader_warns_with_the_type_name_and_the_rest_still_exports(
        home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _boom(home=None, **_kw):
        raise PermissionError("/secret/path/state/gateway.lifecycle.json")

    monkeypatch.setattr("gateway.memory_status.collect_memory_status", _boom)
    with caplog.at_level(logging.WARNING, logger=gateway_health_export.logger.name):
        names = [m.name for m in gateway_health_export._read_runtime_snapshot({}).metrics]

    assert "hermes.gateway.up" in names and "hermes.gateway.background_work" in names
    assert "hermes.gateway.previous_exit" not in names
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING and "memory health" in r.getMessage()]
    assert "PermissionError" in record.getMessage() and "/secret" not in record.getMessage()
    assert record.exc_info is None


def test_events_only_pass_skips_the_memory_files(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("gateway.memory_status.collect_memory_status", lambda *a, **k: calls.append(1) or {})
    gateway_health_export._emit_snapshot_events({})
    assert calls == []
    gateway_health_export._read_runtime_snapshot({})
    assert calls == [1]


def test_gauge_callbacks_within_one_second_read_the_files_once(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway import memory_status

    calls: list[int] = []
    real = memory_status.collect_memory_status
    monkeypatch.setattr(memory_status, "collect_memory_status", lambda *a, **k: calls.append(1) or real(*a, **k))
    clock = [1000.0]
    monkeypatch.setattr(gateway_memory_health.time, "monotonic", lambda: clock[0])
    _claim(home, {})

    for _ in range(3):  # one export: every gauge callback rebuilds the snapshot
        _only("hermes.gateway.previous_exit")
    assert calls == [1]
    clock[0] += 1.5
    _claim(home, _PRIOR_FLAGS["unclean"])
    assert _only("hermes.gateway.previous_exit").attributes["hermes.previous_exit"] == "unclean"
    assert calls == [1, 1]


def test_memory_gauges_reach_loopback_collector(home: Path) -> None:
    """Real SDK + OTLP transport: both gauges arrive with exactly the bounded attribute."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pytest.importorskip("opentelemetry.sdk.metrics", reason="otlp extra not installed")
    request_type = pytest.importorskip("opentelemetry.proto.collector.metrics.v1.metrics_service_pb2",
                                       reason="otlp extra not installed").ExportMetricsServiceRequest

    _claim(home, _PRIOR_FLAGS["oom_kill_cgroup"])
    _heartbeat(home, available_kib=32_000)
    received: list = []
    delivered = threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(request_type.FromString(payload))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            delivered.set()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = None
    try:
        runtime = gateway_health_export.start_gateway_health_export({"monitoring": {
            "install_id": "test-install",
            "export": {"otlp": {"enabled": True, "endpoint": f"http://127.0.0.1:{server.server_port}/v1/traces"}},
            "gateway_health_export": {"enabled": True, "metrics_enabled": True, "diagnostic_events_enabled": False},
        }})
        assert runtime.enabled, runtime.reason
        assert runtime.metric_provider.force_flush(timeout_millis=10000)
        assert delivered.wait(timeout=10)
        points = {metric.name: [{kv.key: kv.value.string_value for kv in point.attributes}
                                for point in metric.gauge.data_points]
                  for request in received for resource in request.resource_metrics
                  for scope in resource.scope_metrics for metric in scope.metrics}
        [exit_point] = points["hermes.gateway.previous_exit"]
        [pressure_point] = points["hermes.gateway.memory_pressure"]
        assert exit_point == {**_BASE, "hermes.previous_exit": "oom_kill_cgroup"}
        assert pressure_point == {**_BASE, "hermes.memory_pressure": "critical"}
    finally:
        if runtime is not None:
            runtime.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
