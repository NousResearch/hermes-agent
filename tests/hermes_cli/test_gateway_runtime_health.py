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


def test_runtime_health_lines_include_discord_websocket_summary(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "discord": {
                    "state": "degraded",
                    "health": {
                        "transport": "websocket",
                        "heartbeat_ack_age_seconds": 181.2,
                        "latency_ms": 42.0,
                        "last_health_reason": "ack_stale",
                    },
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert any("WebSocket degraded" in line for line in lines)
    assert any("ACK age 181.2s" in line for line in lines)
    assert any("latency 42ms" in line for line in lines)


def test_runtime_health_lines_redact_and_flatten_untrusted_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "discord": {
                    "state": "retrying",
                    "health": {
                        "transport": "websocket",
                        "last_health_reason": "Authorization: Bearer " + ("x" * 32) + chr(10) + "extra",
                    },
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert len(lines) == 1
    assert "x" * 32 not in lines[0]
    assert "\\n" not in lines[0]


def test_runtime_health_lines_strip_url_credentials_and_query_values(monkeypatch):
    diagnostic = (
        "request failed at https://user:pass@example.invalid:8443/path?access_token=opaque#fragment "
        "Bearer x"
    )
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "discord": {
                    "state": "retrying",
                    "health": {
                        "transport": "websocket",
                        "last_health_reason": diagnostic,
                    },
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "user:pass@" not in lines[0]
    assert ":8443" not in lines[0]
    assert "access_token=opaque" not in lines[0]
    assert "opaque-token-value-1234567890" not in lines[0]
    assert "#fragment" not in lines[0]


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
