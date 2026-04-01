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
    monkeypatch.setattr(
        "hermes_cli.gateway.load_config",
        lambda: {"platforms": {"telegram": {"enabled": True}}},
    )

    lines = _runtime_health_lines()

    assert "telegram: another poller is active" in lines
    assert "Last startup issue: telegram conflict" in lines


def test_runtime_health_lines_include_reconnecting_platform(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "telegram": {
                    "state": "reconnecting",
                    "error_message": "Telegram polling reconnect failed on attempt 1/10: Timed out",
                }
            },
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.load_config",
        lambda: {"platforms": {"telegram": {"enabled": True}}},
    )

    lines = _runtime_health_lines()

    assert "telegram: reconnecting — Telegram polling reconnect failed on attempt 1/10: Timed out" in lines


def test_runtime_health_lines_ignore_disconnected_and_unconfigured_platforms(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "telegram": {"state": "disconnected", "error_message": ""},
                "webhook": {"state": "fatal", "error_message": "stale webhook issue"},
                "whatsapp": {"state": "fatal", "error_message": "stale whatsapp issue"},
            },
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.load_config",
        lambda: {"platforms": {"telegram": {"enabled": True}}},
    )

    assert _runtime_health_lines() == []
