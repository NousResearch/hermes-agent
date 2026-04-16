from datetime import datetime, timedelta, timezone

import pytest


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def test_evaluate_runtime_health_reports_stale_qq_and_stuck_work():
    from gateway.runtime_canary import evaluate_runtime_health

    now = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    runtime_status = {
        "gateway_state": "running",
        "updated_at": _iso(now - timedelta(seconds=20)),
        "platforms": {
            "qq_napcat": {
                "state": "connected",
                "updated_at": _iso(now - timedelta(minutes=20)),
            }
        },
        "runtime_summary": {
            "active_sessions": [
                {
                    "platform": "qq_napcat",
                    "chat_id": "726109087",
                    "current_tool": "delegate_task",
                    "age_seconds": 901,
                }
            ],
            "background_jobs": {
                "active": [
                    {
                        "task_id": "bg_123",
                        "status": "running",
                        "age_seconds": 1201,
                    }
                ]
            },
        },
    }

    result = evaluate_runtime_health(runtime_status, now=now)

    assert result["healthy"] is False
    assert {issue["code"] for issue in result["issues"]} == {
        "qq_connectivity_stale",
        "active_session_stuck",
        "background_work_stuck",
    }
    assert "qq_napcat connectivity stale" in result["summary"]


def test_evaluate_runtime_health_reports_provider_degradation_when_present():
    from gateway.runtime_canary import evaluate_runtime_health

    now = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    runtime_status = {
        "gateway_state": "running",
        "updated_at": _iso(now),
        "runtime_summary": {
            "model": {
                "active_provider": "openrouter",
                "active_model": "gpt-5.4-mini",
                "degraded_provider": "openrouter",
                "degraded_model": "gpt-5.4-mini",
                "degraded_reason": "stream_drop",
                "degraded_failures": 3,
                "degraded_cooldown_until": _iso(now + timedelta(minutes=10)),
            }
        },
    }

    result = evaluate_runtime_health(runtime_status, now=now)

    assert result["healthy"] is False
    issue = next(issue for issue in result["issues"] if issue["code"] == "provider_degraded")
    assert issue["provider"] == "openrouter"
    assert issue["model"] == "gpt-5.4-mini"
    assert issue["failure_count"] == 3
    assert "stream_drop" in issue["message"]


def test_run_runtime_canary_throttles_repeated_alerts():
    from gateway.runtime_canary import run_runtime_canary

    now = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    runtime_status = {
        "gateway_state": "running",
        "updated_at": _iso(now),
        "platforms": {
            "qq_napcat": {
                "state": "connected",
                "updated_at": _iso(now - timedelta(minutes=20)),
            }
        },
        "runtime_summary": {},
    }

    first = run_runtime_canary(
        runtime_status=runtime_status,
        alert_state={},
        alert_target="qq_napcat:dm:179033731",
        now=now,
        throttle_seconds=900,
        gateway_stale_seconds=3600,
    )
    second = run_runtime_canary(
        runtime_status=runtime_status,
        alert_state=first["alert_state"],
        alert_target="qq_napcat:dm:179033731",
        now=now + timedelta(minutes=5),
        throttle_seconds=900,
        gateway_stale_seconds=3600,
    )

    assert first["should_alert"] is True
    assert first["alert_target"] == "qq_napcat:dm:179033731"
    assert "qq_napcat connectivity stale" in first["alert_text"]
    assert second["should_alert"] is False
    assert second["throttled"] is True
    assert second["evaluation"]["issues"][0]["code"] == "qq_connectivity_stale"
    assert second["evaluation"]["status"] == "critical"


def test_run_runtime_canary_realerts_when_target_changes():
    from gateway.runtime_canary import run_runtime_canary

    now = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    runtime_status = {
        "gateway_state": "running",
        "updated_at": _iso(now),
        "platforms": {
            "qq_napcat": {
                "state": "connected",
                "updated_at": _iso(now - timedelta(minutes=20)),
            }
        },
        "runtime_summary": {},
    }

    first = run_runtime_canary(
        runtime_status=runtime_status,
        alert_state={},
        alert_target="qq_napcat:dm:179033731",
        now=now,
        throttle_seconds=900,
        gateway_stale_seconds=3600,
    )
    second = run_runtime_canary(
        runtime_status=runtime_status,
        alert_state=first["alert_state"],
        alert_target="telegram:dm:board",
        now=now + timedelta(minutes=5),
        throttle_seconds=900,
        gateway_stale_seconds=3600,
    )

    assert first["should_alert"] is True
    assert second["should_alert"] is True
    assert second["alert_target"] == "telegram:dm:board"
    assert second["alert_state"]["last_alert_target"] == "telegram:dm:board"
    assert second["alert_state"]["last_alerts"]["qq_connectivity_stale"]["target"] == "telegram:dm:board"
