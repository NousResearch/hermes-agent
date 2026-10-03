"""Regression for #26859: platform state belongs to its writer, not the next PID."""

import json
from unittest.mock import AsyncMock

from starlette.testclient import TestClient


def test_snapshot_identity_survives_http_projection_and_startup_cleanup(
    tmp_path, monkeypatch
):
    import gateway.status as status
    import hermes_cli.web_server as server
    import hermes_cli.web_routers.status as route
    import hermes_cli.web_server_gateway as gateway_web

    snapshot = {
        "gateway_state": "running",
        "pid": 4242,
        "start_time": 111.0,
        "platforms": {"discord": {"state": "connected"}},
    }
    live = {"pid": 4242, "start": 111.0}
    monkeypatch.setattr(status, "get_running_pid_cached", lambda *a, **k: live["pid"])
    monkeypatch.setattr(status, "read_runtime_status", lambda *a, **k: snapshot)
    monkeypatch.setattr(status, "get_runtime_status_running_pid", lambda *a, **k: None)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: live["start"])
    monkeypatch.setattr(
        gateway_web, "_load_configured_gateway_platforms", lambda: {"discord"}
    )
    monkeypatch.setattr(
        gateway_web,
        "_collect_profile_gateway_topology_cached",
        lambda: {
            "profile_platforms": {},
            "profiles": ["default"],
            "gateway_mode": "single",
            "gateways": [],
        },
    )
    monkeypatch.setattr(route, "_status_active_sessions", AsyncMock(return_value=0))
    monkeypatch.setattr(route, "_advisory_pressure", AsyncMock())
    monkeypatch.setattr(route, "_component_health", AsyncMock(return_value={}))
    monkeypatch.setattr(server, "_GATEWAY_HEALTH_URL", None)
    client = TestClient(server.app)
    client.headers[server._SESSION_HEADER_NAME] = server._SESSION_TOKEN
    problems = []
    for pid, start, expected in (
        (4242, 111.0, {"discord": {"state": "connected"}}),
        (9999, 222.0, {}),
        (4242, 222.0, {}),
    ):
        live.update(pid=pid, start=start)
        response = client.get("/api/status")
        assert response.status_code == 200
        actual = response.json()
        assert actual["gateway_running"] is True
        if actual["gateway_platforms"] != expected:
            problems.append(("HTTP", pid, start, actual["gateway_platforms"], expected))

    # The startup call already requests profile cleanup on base. It must also discard
    # primary-platform state from a different writer *before* re-stamping identity.
    path = tmp_path / "gateway_state.json"
    monkeypatch.setattr(status, "_get_runtime_status_path", lambda: path)
    for pid, start, expected in (
        (4242, 111.0, {"discord": {"state": "connected"}}),
        (9999, 222.0, {}),
        (4242, 222.0, {}),
    ):
        path.write_text(
            json.dumps({
                **snapshot,
                "platforms": {
                    "discord": {"state": "connected"},
                    "reviewer:slack": {"state": "fatal"},
                },
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            status,
            "_build_pid_record",
            lambda pid=pid, start=start: {
                "kind": "gateway",
                "pid": pid,
                "start_time": start,
                "argv": [],
            },
        )
        assert status.write_runtime_status(
            gateway_state="starting",
            clear_profile_platforms=True,
            reload_existing=True,
            wait_timeout=2.0,
        )
        actual = json.loads(path.read_text())
        assert (actual["pid"], actual["start_time"]) == (pid, start)
        if actual["platforms"] != expected:
            problems.append(("startup", pid, start, actual["platforms"], expected))
    assert problems == []
