"""Phase 7 — /api/status exposes auth-gate state + AuthWidget integration.

The dashboard's status endpoint now reports ``auth_required`` and
``auth_providers`` so the AuthWidget + StatusPage can render the
correct "gated / loopback" badge without a separate round trip. This
test asserts both shapes (gated and loopback).

The AuthWidget itself is .tsx — no Python test here. The widget's
behaviour (renders nothing on 401, shows truncated user_id, etc.) is
documented in AuthWidget.tsx; covered manually via the Phase 4.2
smoke test against staging Portal.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_client():
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def loopback_client():
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_status_reports_auth_required_in_gated_mode(gated_client):
    # No ``_login()`` call — ``/api/status`` is in the shared
    # ``PUBLIC_API_PATHS`` allowlist precisely so external probes (and
    # the SPA's pre-login bootstrap) can read the gate's shape without
    # a cookie. Hit it cold.
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is True
    assert body["auth_providers"] == ["stub"]




# Host-local detail (absolute paths, PID, internal gateway URL) is deployment
# recon a liveness probe never needs. ``/api/status`` bypasses dashboard auth
# (it is in ``PUBLIC_API_PATHS``), so on a network-exposed bind it must not
# leak that detail to anonymous callers.
_HOST_DETAIL_FIELDS = frozenset({
    "hermes_home", "config_path", "env_path", "gateway_pid",
    "gateway_health_url",
})


def test_status_withholds_host_detail_in_gated_mode(gated_client):
    """On a gated (non-loopback) bind, the public ``/api/status`` probe must
    expose only the liveness + auth-gate shape — never absolute host paths,
    the gateway PID, or the internal gateway health URL. The endpoint
    bypasses dashboard auth, so anyone who can reach the host hits it cold."""
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    # Liveness / auth-gate shape stays public.
    for key in ("version", "gateway_state", "auth_required", "auth_providers"):
        assert key in body, f"liveness field {key!r} must stay public"
    # Deployment recon must be withheld from the anonymous public probe.
    leaked = _HOST_DETAIL_FIELDS & set(body.keys())
    assert not leaked, f"/api/status leaked host detail under the gate: {leaked}"




# ``webapp_url`` drives the SPA's "your dashboard is moving" modal: only a hosted (Hermes Cloud)
# agent's signed-in standalone dashboard gets it, never a local one and never the Webapp itself.
_WEBAPP_URL = "https://otter--app.agents.nousresearch.com"


def _login(client) -> None:
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False)


_READY = {"state": "ready", "url": _WEBAPP_URL}


@pytest.fixture
def hosted_webapp(monkeypatch):
    from hermes_cli.web_routers import status
    monkeypatch.setattr(status, "_hosted_webapp_move", lambda: _READY)


def test_status_names_the_webapp_to_a_signed_in_hosted_dashboard(gated_client, monkeypatch):
    from hermes_cli.web_routers import status
    monkeypatch.setattr(status, "_hosted_webapp_move", lambda: pytest.fail("asked for a signed-out caller"))
    signed_out = gated_client.get("/api/status").json()  # signed out: gateway not queried
    assert "webapp_url" not in signed_out and "webapp_state" not in signed_out
    monkeypatch.setattr(status, "_hosted_webapp_move", lambda: _READY)
    _login(gated_client)
    body = gated_client.get("/api/status").json()
    assert (body["webapp_state"], body["webapp_url"]) == ("ready", _WEBAPP_URL)


@pytest.mark.parametrize("state", ["starting", "down", "not_available"])
def test_status_reports_a_webapp_that_is_not_up_without_an_address(gated_client, monkeypatch, state):
    from hermes_cli.web_routers import status
    monkeypatch.setattr(status, "_hosted_webapp_move", lambda: {"state": state, "url": None})
    _login(gated_client)
    body = gated_client.get("/api/status").json()
    assert body["webapp_state"] == state and "webapp_url" not in body


def test_status_omits_the_webapp_on_a_local_dashboard(loopback_client, hosted_webapp):
    body = loopback_client.get("/api/status").json()
    assert "webapp_url" not in body and "webapp_state" not in body


def test_status_omits_the_webapp_on_the_webapp_itself(gated_client, hosted_webapp, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    _login(gated_client)
    body = gated_client.get("/api/status").json()
    assert "webapp_url" not in body and "webapp_state" not in body


def _access(state="serving", *, enabled=True, public_url=_WEBAPP_URL, served_port=9120):
    """``webapp_access.access_status()`` for a config asking for the Webapp on port 9120."""
    url = f"http://127.0.0.1:{served_port}" if state == "serving" else None
    return {"enabled": enabled, "port": 9120, "public_url": public_url,
            "gateway": {"state": state, "url": url, "reason": None}}


def _move(state, url=None):
    return {"state": state, "url": url}


# ``layout`` is the hosted ``/opt/data`` home; ``stamp`` is the Portal's ``HERMES_HOSTING``.
@pytest.mark.parametrize("layout,stamp,access,expected", [
    (True, None, _access(), _move("ready", _WEBAPP_URL)),
    (False, "nous_cloud", _access(), _move("ready", _WEBAPP_URL)),  # stamped, another layout
    (False, None, _access(), None),  # not Hermes Cloud
    (False, "self_hosted", _access(), None),
    (True, None, _access("starting"), _move("starting")),  # gateway coming up
    (True, None, _access(served_port=9999), _move("starting")),  # not restarted onto the config yet
    (True, None, _access("stopped"), _move("down")),  # no gateway answers
    (True, None, _access("unavailable"), _move("down")),
    (True, None, _access("off"), _move("down")),  # a gateway older than the Webapp
    # Not turned on: only the Portal's stamp says this is Cloud, since the Docker image shares
    # the /opt/data layout.
    (True, "nous_cloud", _access(enabled=False), _move("not_available")),
    (True, "nous_cloud", _access(public_url=""), _move("not_available")),
    (True, None, _access(enabled=False), None),
    (True, None, _access(public_url=""), None),
])
def test_hosted_webapp_move(monkeypatch, layout, stamp, access, expected):
    from hermes_cli import web_server_files, webapp_access
    from hermes_cli.web_routers import status
    monkeypatch.setattr(web_server_files, "_default_hermes_root_is_opt_data", lambda: layout)
    monkeypatch.delenv("HERMES_HOSTING", raising=False)
    if stamp:
        monkeypatch.setenv("HERMES_HOSTING", stamp)
    monkeypatch.setattr(webapp_access, "access_status", lambda: access)
    assert status._hosted_webapp_move() == expected


def test_local_dashboard_never_asks_the_gateway_about_the_webapp(monkeypatch):
    from hermes_cli import web_server_files, webapp_access
    from hermes_cli.web_routers import status
    monkeypatch.setattr(web_server_files, "_default_hermes_root_is_opt_data", lambda: False)
    monkeypatch.delenv("HERMES_HOSTING", raising=False)
    monkeypatch.setattr(webapp_access, "access_status", lambda: pytest.fail("queried the gateway"))
    assert status._hosted_webapp_move() is None
