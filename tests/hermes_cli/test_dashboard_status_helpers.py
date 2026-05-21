"""Dashboard status/repair helper coverage."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli import web_server


def test_dashboard_sanitizes_gateway_platform_errors_again():
    platforms = {
        "telegram": {
            "state": "fatal",
            "error_code": "unauthorized",
            "error_message": (
                "Telegram rejected token "
                "`123456789:abcdefghijklmnopqrstuvwxyzABCDE123456789:moreSECRET`"
            ),
            "updated_at": "2026-05-20T00:00:00Z",
        }
    }

    sanitized = web_server._sanitize_gateway_platforms(platforms)

    msg = sanitized["telegram"]["error_message"]
    assert "abcdefghijklmnopqrstuvwxyz" not in msg
    assert "moreSECRET" not in msg
    assert "[REDACTED]" in msg


def test_dashboard_status_summary_includes_service_health_and_gateway_repair_hint():
    status = web_server._build_dashboard_health_summary(
        gateway_running=False,
        gateway_state="stopped",
        gateway_platforms={},
        active_sessions=2,
    )

    assert status["overall"] == "degraded"
    assert {item["id"] for item in status["services"]} >= {"dashboard", "gateway", "sessions"}
    gateway = next(item for item in status["services"] if item["id"] == "gateway")
    assert gateway["state"] == "stopped"
    assert gateway["repair_action"] == "repair-stack"


def test_dashboard_status_summary_degrades_when_any_configured_platform_is_down():
    status = web_server._build_dashboard_health_summary(
        gateway_running=True,
        gateway_state="running",
        gateway_platforms={
            "telegram": {"state": "connected"},
            "discord": {"state": "startup_failed"},
        },
        active_sessions=0,
    )

    assert status["overall"] == "degraded"
    platforms = next(item for item in status["services"] if item["id"] == "platforms")
    assert platforms["state"] == "degraded"
    assert platforms["repair_action"] == "repair-stack"


def test_repair_stack_spawns_single_safe_repair_action():
    with patch.object(web_server, "_spawn_hermes_action") as spawn:
        proc = spawn.return_value
        proc.pid = 4242
        resp = web_server._start_repair_stack_action()

    spawn.assert_called_once_with(["dashboard", "repair-stack"], "repair-stack")
    assert resp == {"ok": True, "pid": 4242, "name": "repair-stack"}
