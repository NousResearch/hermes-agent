"""A stopped gateway's per-platform verdict is history, not current state.

``gateway_state.json`` preserves platform entries across restarts, so a gateway that once ran
WITHOUT a Telegram token and then stopped leaves ``fatal / No bot token configured`` behind. The
Channels payload must not repeat that after the user saved credentials: with no live gateway the
platform reads ``gateway_stopped`` and carries no error (Desktop Messaging page report).
"""
import json
import os

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


def _write_hosted_running_state(home):
    """A fresh, live runtime record written by THIS (non-``gateway run``) process — the
    #116416 in-process deployment shape: messaging is served, the heartbeat is current, but
    no strict command-line identity will ever match."""
    from datetime import datetime, timezone
    from gateway import status as gateway_status

    now = datetime.now(timezone.utc).isoformat()
    (home / "gateway_state.json").write_text(json.dumps({
        "kind": "gateway", "pid": os.getpid(),
        "start_time": gateway_status._get_process_start_time(os.getpid()),
        "argv": ["hermes", "dashboard", "--host", "127.0.0.1", "--no-open"],
        "gateway_state": "running", "exit_reason": None,
        "updated_at": now,
        "platforms": {"telegram": {
            "state": "connected", "writer_pid": os.getpid(), "writer_start_time": 1.0,
            "updated_at": now,
        }},
    }), encoding="utf-8")


def test_fresh_hosted_loop_keeps_live_platform_verdict(client):
    """#116416: when a live non-``gateway run`` host carries the loop (fresh heartbeat, live
    PID), the Channels page repeats its live verdict instead of flattening to gateway_stopped."""
    from hermes_constants import get_hermes_home

    _write_hosted_running_state(get_hermes_home())

    payload = client.get("/api/messaging/platforms").json()
    telegram = next(p for p in payload["platforms"] if p["id"] == "telegram")

    # The strict ladder stays down — lifecycle surfaces (stop/restart/drain) keep their
    # command-line safety — but the presentation trusts the live writer.
    assert telegram["gateway_running"] is False
    assert telegram["state"] == "connected"
    assert telegram["error_code"] is None
    assert telegram["error_message"] is None


def test_api_status_reports_hosted_loop_running(client):
    """>/api/status keeps the hosted loop's real state and platform map, with the host's PID
    in the display-only ``hosted_pid`` field — ``gateway_pid`` stays lifecycle-manageable
    only (#116416, #116445 review)."""
    from hermes_constants import get_hermes_home

    _write_hosted_running_state(get_hermes_home())

    payload = client.get("/api/status").json()

    assert payload["gateway_running"] is False
    assert payload["gateway_state"] == "running"
    # No ``gateway run`` process exists, so the lifecycle PID stays null and the live host
    # is reported separately — an API consumer must not mistake it for a stoppable PID.
    assert payload["gateway_pid"] is None
    assert payload["hosted_pid"] == os.getpid()
    telegram = payload["gateway_platforms"].get("telegram")
    assert isinstance(telegram, dict) and telegram["state"] == "connected"


def test_hosted_loop_with_empty_platform_entry_is_pending_restart(client):
    """Same arm as a live gateway without a runtime entry for the platform: the hosted host is
    serving, so a missing entry reads pending_restart, not gateway_stopped."""
    from hermes_constants import get_hermes_home

    _write_hosted_running_state(get_hermes_home())
    state = json.loads((get_hermes_home() / "gateway_state.json").read_text(encoding="utf-8"))
    state["platforms"] = {}
    (get_hermes_home() / "gateway_state.json").write_text(json.dumps(state), encoding="utf-8")

    payload = client.get("/api/messaging/platforms").json()
    telegram = next(p for p in payload["platforms"] if p["id"] == "telegram")

    assert telegram["gateway_running"] is False
    assert telegram["state"] == "pending_restart"
