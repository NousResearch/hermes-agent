from hermes_cli.telegram_miniapp.status import build_status_snapshot


def test_build_status_snapshot_redacts_paths_and_uses_safe_fields(monkeypatch):
    from hermes_cli.telegram_miniapp import status as status_mod

    runtime = {
        "gateway_state": "running",
        "active_agents": "2",
        "restart_requested": True,
        "updated_at": "2026-07-01T10:00:00+00:00",
        "pid": 12345,
        "argv": ["python", "gateway"],
        "secret": "SHOULD_NOT_LEAK",
    }
    monkeypatch.setattr(status_mod, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: 12345)
    monkeypatch.setattr(status_mod, "get_runtime_status_running_pid", lambda payload: 12345)

    snapshot = build_status_snapshot(hermes_home_configured=True)

    assert snapshot["ok"] is True
    assert snapshot["hermes_home"] == "configured"
    assert snapshot["gateway"] == {
        "running": True,
        "state": "running",
        "busy": True,
        "drainable": True,
        "active_agents": 2,
        "restart_requested": True,
    }
    serialized = str(snapshot)
    assert "/Volumes/Diver Pro/hermes" not in serialized
    assert "SHOULD_NOT_LEAK" not in serialized
    assert "argv" not in serialized
    assert "12345" not in serialized


def test_build_status_snapshot_handles_missing_runtime(monkeypatch):
    from hermes_cli.telegram_miniapp import status as status_mod

    monkeypatch.setattr(status_mod, "read_runtime_status", lambda: None)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: None)

    snapshot = build_status_snapshot(hermes_home_configured=False)

    assert snapshot["ok"] is True
    assert snapshot["hermes_home"] == "missing"
    assert snapshot["gateway"]["running"] is False
    assert snapshot["gateway"]["state"] == "unknown"
    assert snapshot["gateway"]["active_agents"] == 0
    assert snapshot["gateway"]["busy"] is False
    assert snapshot["gateway"]["drainable"] is False


def test_build_status_snapshot_clamps_invalid_active_agents(monkeypatch):
    from hermes_cli.telegram_miniapp import status as status_mod

    runtime = {"gateway_state": "running", "active_agents": "-100", "restart_requested": False}
    monkeypatch.setattr(status_mod, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: 111)
    monkeypatch.setattr(status_mod, "get_runtime_status_running_pid", lambda payload: 111)

    snapshot = build_status_snapshot(hermes_home_configured=True)

    assert snapshot["gateway"]["active_agents"] == 0
    assert snapshot["gateway"]["busy"] is False
