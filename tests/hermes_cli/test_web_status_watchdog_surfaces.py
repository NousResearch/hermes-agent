"""``/api/status`` agrees with ``hermes gateway status`` on the two #113372 shapes:

* A watchdog hard-exited the process (``degraded`` + watchdog ``exit_reason``, PID gone) -> the
  retained verdict stays ``degraded`` with its ``gateway_exit_reason`` instead of a bare ``stopped``.
"""
from datetime import datetime, timedelta, timezone

import pytest

import gateway.status as _gw_status


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


def test_status_keeps_watchdog_degraded_verdict_and_reason_for_dead_pid(client, monkeypatch):
    record = {"gateway_state": "degraded", "exit_reason": "loop_liveness_watchdog",
              "pid": 999_999_999, "start_time": 1.0, "updated_at": _iso_age(30), "platforms": {}}
    monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: None)
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda: record)

    data = client.get("/api/status").json()
    assert data["gateway_running"] is False
    assert data["gateway_state"] == "degraded"
    assert data["gateway_exit_reason"] == "loop_liveness_watchdog"

    # ``hermes gateway stop`` afterwards records the operator's intent: no longer a current failure.
    record["desired_state"] = "stopped"
    data = client.get("/api/status").json()
    assert data["gateway_state"] == "stopped"
    assert data["gateway_exit_reason"] is None
