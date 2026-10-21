"""A stopped gateway's per-platform verdict is history, not current state.

``gateway_state.json`` preserves platform entries across restarts, so a gateway that once ran
WITHOUT a Telegram token and then stopped leaves ``fatal / No bot token configured`` behind. The
Channels payload must not repeat that after the user saved credentials: with no live gateway the
platform reads ``gateway_stopped`` and carries no error (Desktop Messaging page report).
"""
import json

import pytest


_VALID_BOT_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\nTELEGRAM_ALLOWED_USERS=42\n", encoding="utf-8")
    (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    (home / "gateway_state.json").write_text(json.dumps({
        "kind": "gateway", "pid": 999_999_999, "start_time": 1.0, "gateway_state": "stopped",
        "exit_reason": "shutdown", "updated_at": "2026-01-01T00:00:00+00:00",
        "platforms": {"telegram": {
            "state": "fatal", "error_code": "missing_credentials",
            "error_message": "No bot token configured",
            "writer_pid": 999_999_999, "writer_start_time": 1.0,
        }},
    }), encoding="utf-8")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_stopped_gateway_does_not_report_stale_platform_error(client):
    payload = client.get("/api/messaging/platforms").json()
    telegram = next(p for p in payload["platforms"] if p["id"] == "telegram")

    assert telegram["configured"] is True
    assert telegram["gateway_running"] is False
    # The saved token is current; the dead gateway's "no token" verdict is not.
    assert telegram["state"] == "gateway_stopped"
    assert telegram["error_code"] is None
    assert telegram["error_message"] is None


def test_operator_stopped_gateway_does_not_report_retained_startup_failure(client):
    """``hermes gateway stop`` keeps the last ``startup_failed`` + ``exit_reason`` on disk with
    ``desired_state: stopped``; the Channels page must read that as stopped, exactly like
    ``/api/status`` does, not wear a "Start failed" badge with the stale reason (#112517)."""
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "gateway_state.json").write_text(json.dumps({
        "kind": "gateway", "pid": 999_999_999, "start_time": 1.0,
        "gateway_state": "startup_failed", "desired_state": "stopped",
        "exit_reason": "Port 8642 already in use", "updated_at": "2026-01-01T00:00:00+00:00",
        "platforms": {},
    }), encoding="utf-8")

    payload = client.get("/api/messaging/platforms").json()
    telegram = next(p for p in payload["platforms"] if p["id"] == "telegram")

    assert telegram["gateway_running"] is False
    assert telegram["state"] == "gateway_stopped"
    assert telegram["error_code"] is None
    assert telegram["error_message"] is None
