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


def test_runtime_health_lines_include_cron_failure(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "cron": {
                "state": "failing",
                "consecutive_failures": 3,
                "last_error": "OSError: too many open files",
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ Cron ticker failing (3 consecutive failures): OSError: too many open files" in lines
