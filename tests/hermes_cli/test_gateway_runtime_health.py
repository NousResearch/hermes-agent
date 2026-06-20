import hermes_cli.gateway as gateway_cli
from hermes_cli.gateway import _runtime_health_lines


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(gateway_cli, "_recent_gateway_log_transport_issues", lambda classifier: [])
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


def test_runtime_health_lines_surface_paused_telegram_transport(monkeypatch):
    """Incident regression: gateway process alive, Telegram transport paused.

    Before the fix, _runtime_health_lines only surfaced platforms in `fatal`
    state. A `paused` platform after repeated reconnect failures was silent,
    so operators saw a green gateway while conductors were unreachable.
    """
    monkeypatch.setattr(gateway_cli, "_recent_gateway_log_transport_issues", lambda classifier: [])
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "exit_reason": None,
            "platforms": {
                "telegram": {
                    "state": "paused",
                    "error_message": "auto-paused after 10 consecutive reconnect failures",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert any(
        "telegram" in line.lower() and "paused" in line.lower()
        for line in lines
    ), lines


def test_runtime_health_lines_surface_stale_telegram_poll(monkeypatch):
    """When the telegram transport hasn't polled successfully past the
    staleness threshold, _runtime_health_lines must surface it even if
    `state` is still nominally `connected`."""
    monkeypatch.setattr(gateway_cli, "_recent_gateway_log_transport_issues", lambda classifier: [])
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "exit_reason": None,
            "platforms": {
                "telegram": {
                    "state": "connected",
                    "last_successful_poll_at": "2020-01-01T00:00:00+00:00",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert any(
        "telegram" in line.lower() and "stale" in line.lower()
        for line in lines
    ), lines


def test_runtime_health_lines_surface_telegram_pause_log_signature(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "gateway.log").write_text(
        "2026-06-20T12:00:00 [WARNING] gateway: "
        "Telegram paused after 10 consecutive reconnect failures\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "exit_reason": None,
            "platforms": {},
        },
    )

    lines = _runtime_health_lines()

    assert any("HERMES_TELEGRAM_PAUSED" in line for line in lines), lines
