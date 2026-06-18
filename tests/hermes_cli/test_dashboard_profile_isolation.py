"""Regression tests for dashboard profile isolation.

Machine dashboards intentionally aggregate all local profiles. Desktop-spawned
or --isolated backends are different: they are the backend for exactly one
profile and must not expose sibling profile sessions through cross-profile API
query parameters.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_state import SessionDB


pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")


def _restore_attr(state, name: str, previous):
    if previous is None:
        if hasattr(state, name):
            delattr(state, name)
    else:
        setattr(state, name, previous)


@pytest.fixture
def profile_homes(monkeypatch, tmp_path):
    homes = {
        "default": tmp_path / "default-home",
        "prime": tmp_path / "profiles" / "prime",
        "architect": tmp_path / "profiles" / "architect",
    }
    for name, home in homes.items():
        home.mkdir(parents=True)
        db = SessionDB(db_path=home / "state.db")
        try:
            db.create_session(f"{name}-session", source="cli")
        finally:
            db.close()

    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in homes)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [SimpleNamespace(name=name, path=home) for name, home in homes.items()],
    )
    monkeypatch.setenv("HERMES_HOME", str(homes["prime"]))
    return homes


@pytest.fixture
def dashboard_client(profile_homes):
    state = web_server.app.state
    prev_auth_required = getattr(state, "auth_required", None)
    prev_bound_host = getattr(state, "bound_host", None)
    prev_bound_port = getattr(state, "bound_port", None)
    prev_isolated_profile = getattr(state, "isolated_profile", None)

    state.auth_required = False
    state.bound_host = "127.0.0.1"
    state.bound_port = 9119
    if hasattr(state, "isolated_profile"):
        delattr(state, "isolated_profile")

    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client
    finally:
        client.close()
        _restore_attr(state, "auth_required", prev_auth_required)
        _restore_attr(state, "bound_host", prev_bound_host)
        _restore_attr(state, "bound_port", prev_bound_port)
        _restore_attr(state, "isolated_profile", prev_isolated_profile)


def test_machine_dashboard_profiles_sessions_all_still_aggregates_all_profiles(dashboard_client):
    response = dashboard_client.get("/api/profiles/sessions", params={"profile": "all"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["profile_totals"]) == {"default", "prime", "architect"}
    assert body["total"] == 3
    assert {session["profile"] for session in body["sessions"]} == {
        "default",
        "prime",
        "architect",
    }


def test_isolated_dashboard_profiles_sessions_all_returns_only_isolated_profile(dashboard_client):
    web_server.app.state.isolated_profile = "prime"

    response = dashboard_client.get("/api/profiles/sessions", params={"profile": "all"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile_totals"] == {"prime": 1}
    assert body["total"] == 1
    assert [session["profile"] for session in body["sessions"]] == ["prime"]
    assert [session["id"] for session in body["sessions"]] == ["prime-session"]


def test_isolated_dashboard_rejects_explicit_sibling_profile_session_reads(dashboard_client):
    web_server.app.state.isolated_profile = "prime"

    aggregate = dashboard_client.get(
        "/api/profiles/sessions",
        params={"profile": "architect"},
    )
    direct = dashboard_client.get(
        "/api/sessions",
        params={"profile": "architect"},
    )

    assert aggregate.status_code == 403
    assert direct.status_code == 403


def test_isolated_dashboard_rejects_explicit_sibling_profile_status_and_channels(dashboard_client):
    web_server.app.state.isolated_profile = "prime"

    status = dashboard_client.get("/api/status", params={"profile": "architect"})
    channels = dashboard_client.get(
        "/api/messaging/platforms",
        params={"profile": "architect"},
    )
    channel_update = dashboard_client.put(
        "/api/messaging/platforms/telegram",
        json={"profile": "architect", "enabled": False},
    )
    channel_test = dashboard_client.post(
        "/api/messaging/platforms/telegram/test",
        params={"profile": "architect"},
    )

    assert status.status_code == 403
    assert channels.status_code == 403
    assert channel_update.status_code == 403
    assert channel_test.status_code == 403
