"""A stopped gateway's per-platform verdict is history, not current state.

``gateway_state.json`` preserves platform entries across restarts, so a gateway that once ran
WITHOUT a Telegram token and then stopped leaves ``fatal / No bot token configured`` behind. The
Channels payload must not repeat that after the user saved credentials: with no live gateway the
platform reads ``gateway_stopped`` and carries no error (Desktop Messaging page report).
"""
import json
import time

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


# --- live_health passthrough with server-computed age_seconds -----------------
# PR #92616 writes a per-platform ``live_health`` record (websocket_state/healthy/
# checked_at, often latency/ack_age) into gateway_state.json. The payload surfaces
# it annotated with ``age_seconds`` so consumers never re-implement freshness math.

def _entry(platform_id="discord"):
    return {"id": platform_id, "name": "Discord", "description": "", "docs_url": "",
            "env_vars": [], "required_env": []}


@pytest.fixture
def live_gateway(monkeypatch):
    """The messaging router with liveness forced running (else the payload wipes
    the runtime record as stale history) and enablement short-circuited."""
    from types import SimpleNamespace

    from hermes_cli.web_routers import messaging

    monkeypatch.setattr(messaging, "resolve_gateway_liveness",
                        lambda **kw: SimpleNamespace(running=True))
    monkeypatch.setattr(messaging, "_platform_enablement", lambda *a, **k: (True, True, None))
    return messaging


_sentinel = object()


def _runtime_with(live_health=_sentinel) -> dict:
    platform: dict = {"state": "connected"}
    if live_health is not _sentinel:
        platform["live_health"] = live_health
    return {"gateway_state": "running", "platforms": {"discord": platform}}


def test_live_health_surfaces_with_server_computed_age(live_gateway):
    from datetime import datetime, timedelta, timezone

    t0 = time.monotonic()
    checked_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    health = {"websocket_state": "connected", "healthy": True, "checked_at": checked_at,
              "ack_age": 3.2}
    payload = live_gateway._messaging_platform_payload(_entry(), {}, _runtime_with(health))
    elapsed = time.monotonic() - t0

    assert payload["live_health"]["websocket_state"] == "connected"
    assert payload["live_health"]["healthy"] is True
    assert payload["live_health"]["checked_at"] == checked_at
    assert payload["live_health"]["ack_age"] == 3.2
    age = payload["live_health"]["age_seconds"]
    assert isinstance(age, float)
    # Upper bound is 120 + actually-elapsed wall time + 1 (rounding), not a bet on runner speed.
    assert 119.0 <= age <= 120 + elapsed + 1, age


def test_future_dated_checked_at_clamps_age_to_zero(live_gateway):
    from datetime import datetime, timedelta, timezone

    # The reader's clock is behind the writer's (NTP step, VM skew); a negative
    # age is meaningless to a consumer that renders it.
    checked_at = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    health = {"websocket_state": "connected", "healthy": True, "checked_at": checked_at,
              "ack_age": 3.2}
    payload = live_gateway._messaging_platform_payload(_entry(), {}, _runtime_with(health))

    assert payload["live_health"]["websocket_state"] == "connected"
    assert payload["live_health"]["healthy"] is True
    assert payload["live_health"]["checked_at"] == checked_at
    assert payload["live_health"]["ack_age"] == 3.2
    assert payload["live_health"]["age_seconds"] == 0.0


def test_platform_without_live_health_yields_none(live_gateway):
    payload = live_gateway._messaging_platform_payload(_entry(), {}, _runtime_with())
    assert payload["live_health"] is None


@pytest.mark.parametrize("broken", [
    {"websocket_state": "connected", "healthy": True},              # checked_at missing
    {"checked_at": ""},                                             # empty
    {"checked_at": "not-a-timestamp"},                              # unparseable
    {"checked_at": "2026-01-01T00:00:00"},                          # naive: TypeError on subtract
    {"checked_at": 1767225600},                                     # mis-typed
    "just-a-string",                                                # not a dict
])
def test_unusable_checked_at_passes_through_unchanged(live_gateway, broken):
    payload = live_gateway._messaging_platform_payload(_entry(), {}, _runtime_with(broken))
    assert payload["live_health"] == broken
    if isinstance(broken, dict):
        assert "age_seconds" not in payload["live_health"]
