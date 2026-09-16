"""A profile served by the shared multiplexer must read its platform state from the
multiplexer's record even when a leftover standalone ``gateway_state.json`` sits in the
profile's own home.

The multiplexer fallback used to fire only when the profile had NO runtime file of its own.
A profile that once ran standalone keeps a stopped ``gateway_state.json`` forever: the file
is a dict, so the fallback never fired, the bare ``telegram`` lookup missed, and with the
gateway alive (the multiplexer serves the profile) the state ladder answered
``pending_restart`` — "Restart needed" for a channel that was connected and working
(Desktop Messaging report).
"""

import json
from types import SimpleNamespace

import pytest


_VALID_BOT_TOKEN = "123456789:***"


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_cli.web_routers import messaging

    home = get_hermes_home()
    profiles_root = home / "profiles"
    for name in ("vera", "max"):
        profile_home = profiles_root / name
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / ".env").write_text(
            f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\n", encoding="utf-8"
        )
        (profile_home / "config.yaml").write_text(
            "platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8"
        )
        # Leftover from the profile's standalone days: stopped, no platforms, never rewritten.
        (profile_home / "gateway_state.json").write_text(
            json.dumps({
                "kind": "gateway",
                "pid": 999_999_999,
                "gateway_state": "stopped",
                "exit_reason": "shutdown",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "platforms": {},
            }),
            encoding="utf-8",
        )

    # The shared multiplexer record: vera's telegram is live under the namespaced key.
    shared_runtime = {
        "kind": "gateway",
        "pid": 4242,
        "gateway_state": "running",
        "updated_at": "2026-09-16T08:14:21+00:00",
        "served_profiles": ["default", "vera", "max"],
        "platforms": {
            "webhook": {"state": "connected"},
            "vera:telegram": {
                "state": "connected",
                "error_code": None,
                "error_message": None,
                "updated_at": "2026-09-16T08:14:21+00:00",
            },
        },
    }

    monkeypatch.setattr(
        messaging,
        "multiplexer_liveness_for_profile",
        lambda profile_dir: (4242, shared_runtime),
    )
    monkeypatch.setattr(
        messaging,
        "resolve_gateway_liveness",
        lambda **kwargs: SimpleNamespace(running=True, pid=4242, source="multiplexer"),
    )

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _telegram(client, profile):
    payload = client.get("/api/messaging/platforms", params={"profile": profile}).json()
    return next(p for p in payload["platforms"] if p["id"] == "telegram")


def test_stale_own_record_yields_to_live_multiplexer_state(client):
    telegram = _telegram(client, "vera")

    assert telegram["configured"] is True
    assert telegram["gateway_running"] is True
    assert telegram["state"] == "connected"
    assert telegram["updated_at"] == "2026-09-16T08:14:21+00:00"


def test_sibling_profile_state_does_not_leak_across_scopes(client):
    # max has its own stale record and no ``max:telegram`` entry in the shared one —
    # vera's connected verdict must not bleed into max's scope.
    telegram = _telegram(client, "max")

    assert telegram["configured"] is True
    assert telegram["gateway_running"] is True
    assert telegram["state"] == "pending_restart"
