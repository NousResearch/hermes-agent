from hermes_cli.gateway import _runtime_health_lines


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "startup_failed",
            "exit_reason": "telegram conflict",
            "platforms": {
                "telegram": {
                    "state": "fatal",
                    "error_message": "another poller is active",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ telegram: another poller is active" in lines
    assert "⚠ Last startup issue: telegram conflict" in lines


def test_runtime_status_running_pid_validates_live_gateway_record(monkeypatch):
    from gateway import status as status_mod

    runtime = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--replace"],
        "start_time": None,
        "gateway_state": "running",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: pid == 12345)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status_mod, "_looks_like_gateway_process", lambda pid: False)

    assert status_mod.get_runtime_status_running_pid(runtime) == 12345


def test_runtime_status_running_pid_rejects_stopped_record(monkeypatch):
    from gateway import status as status_mod

    runtime = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--replace"],
        "gateway_state": "stopped",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: True)

    assert status_mod.get_runtime_status_running_pid(runtime) is None
