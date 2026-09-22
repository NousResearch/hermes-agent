"""``/api/status`` agrees with ``hermes gateway status`` on the two #113372 shapes:

* PID alive but the heartbeat stamp is past the freshness TTL (loop/housekeeping wedged while
  ``gateway_state.json`` still says ``running``) -> ``gateway_heartbeat_stale_s`` is set.
* A watchdog hard-exited the process (``degraded`` + watchdog ``exit_reason``, PID gone) -> the
  retained verdict stays ``degraded`` with its ``gateway_exit_reason`` instead of a bare ``stopped``.
"""
from datetime import datetime, timedelta, timezone

import pytest

import gateway.status as _gw_status
from gateway import code_skew


def _iso_age(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


@pytest.fixture
def client(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(_gw_status, "_pid_exists", lambda pid: False)
    monkeypatch.setattr(_gw_status, "_get_process_start_time", lambda pid: None)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_status_reports_stale_heartbeat_when_pid_alive(client, monkeypatch):
    record = {"gateway_state": "running", "pid": 1234, "start_time": 111.0,
              "updated_at": _iso_age(900), "platforms": {}, "active_agents": 0}
    monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: 1234)
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda: record)

    data = client.get("/api/status").json()
    assert data["gateway_running"] is True
    assert data["gateway_state"] == "running"
    assert 900 <= data["gateway_heartbeat_stale_s"] <= 930

    record["updated_at"] = _iso_age(5)
    assert client.get("/api/status").json()["gateway_heartbeat_stale_s"] is None


def test_status_keeps_watchdog_degraded_verdict_and_reason_for_dead_pid(client, monkeypatch):
    record = {"gateway_state": "degraded", "exit_reason": "loop_liveness_watchdog",
              "pid": 999_999_999, "start_time": 1.0, "updated_at": _iso_age(30), "platforms": {}}
    monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: None)
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda: record)

    data = client.get("/api/status").json()
    assert data["gateway_running"] is False
    assert data["gateway_state"] == "degraded"
    assert data["gateway_exit_reason"] == "loop_liveness_watchdog"
    assert data["gateway_heartbeat_stale_s"] is None

    # ``hermes gateway stop`` afterwards records the operator's intent: no longer a current failure.
    record["desired_state"] = "stopped"
    data = client.get("/api/status").json()
    assert data["gateway_state"] == "stopped"
    assert data["gateway_exit_reason"] is None


def test_status_exposes_only_a_proven_checkout_skew(client, monkeypatch):
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
    assert client.get("/api/status").json()["code_skew"] == {
        "boot_rev": "abc1234567", "disk_rev": "def4567890"}

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
    assert "code_skew" not in client.get("/api/status").json()
