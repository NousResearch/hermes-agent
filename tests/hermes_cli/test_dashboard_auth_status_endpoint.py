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



# ---------------------------------------------------------------------------
# dashboard.public_status_detail: operator detail on the anonymous probe
# ---------------------------------------------------------------------------

# Profile names, the gateway topology mode, and the host memory / disk
# rollups describe the deployment, not its liveness. They are public by
# default because Hermes Cloud renders them in the Portal from an
# unauthenticated read. A self-hosted dashboard exposed to the internet has
# no such reader, so ``dashboard.public_status_detail: minimal`` withholds
# them from anonymous callers while keeping them for a signed-in operator.
_OPERATOR_DETAIL_FIELDS = frozenset({
    "profiles", "gateway_mode", "memory", "disk",
})


def _login(client):
    """Walk the stub IDP round trip so the client holds a session cookie."""
    to_idp = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert to_idp.status_code in (302, 307), to_idp.status_code
    callback = to_idp.headers["location"]
    landed = client.get(callback, follow_redirects=False)
    assert landed.status_code in (302, 307), landed.status_code


def test_status_publishes_operator_detail_by_default(gated_client, monkeypatch):
    """Default stays ``full``: the Cloud Portal's profile list must not
    regress on an install that never opts in."""
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    body = gated_client.get("/api/status").json()
    assert "profiles" in body
    assert "memory" in body


def test_status_withholds_operator_detail_when_minimal(gated_client, monkeypatch):
    """``minimal`` + gated bind + anonymous caller: the deployment detail
    must be absent from the public probe."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "minimal"}},
    )
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    # Liveness + auth-gate shape stays public: the probe contract holds.
    for key in ("version", "gateway_state", "auth_required", "auth_providers"):
        assert key in body, f"liveness field {key!r} must stay public"
    leaked = _OPERATOR_DETAIL_FIELDS & set(body.keys())
    assert not leaked, f"/api/status leaked operator detail: {leaked}"


def test_status_keeps_operator_detail_for_signed_in_caller(gated_client, monkeypatch):
    """``minimal`` must not blind the operator's own dashboard: a request
    carrying a valid session cookie still gets the full payload, so the
    Status page keeps rendering profiles and the resource banners."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "minimal"}},
    )
    _login(gated_client)
    body = gated_client.get("/api/status").json()
    assert "profiles" in body, "signed-in caller lost the profile list"
    assert "memory" in body, "signed-in caller lost the memory rollup"


def test_status_probe_survives_a_broken_session_cookie(gated_client, monkeypatch):
    """The public probe must never fail because a cookie could not be
    verified. A liveness endpoint that 500s on a stale cookie is worse
    than the disclosure it was hardening against."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "minimal"}},
    )
    gated_client.cookies.set("hermes_session_at", "not-a-valid-token")
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    assert _OPERATOR_DETAIL_FIELDS.isdisjoint(r.json().keys())


def _session_token(client) -> str:
    """Return the access token of a logged-in client's session cookie.

    The cookie name carries a ``__Host-`` prefix over HTTPS, so match on the
    bare suffix rather than hardcoding the prefixed name.
    """
    for name, value in client.cookies.items():
        if name.endswith("hermes_session_at"):
            return value
    raise AssertionError(f"no session cookie among {list(client.cookies.keys())}")


def test_status_keeps_operator_detail_for_bearer_caller(gated_client, monkeypatch):
    """The desktop authenticates REST with ``Authorization: Bearer`` and holds
    no cookie (RFC 8252 native flow). It must not read as anonymous, or
    ``minimal`` would blind the very cockpit that manages the gateway."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "minimal"}},
    )
    _login(gated_client)
    token = _session_token(gated_client)
    gated_client.cookies.clear()
    body = gated_client.get(
        "/api/status", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert "profiles" in body, "bearer caller lost the profile list"
    assert "memory" in body, "bearer caller lost the memory rollup"


def test_status_minimal_is_a_no_op_on_a_loopback_bind(loopback_client, monkeypatch):
    """``minimal`` describes what an anonymous NETWORK caller sees. A loopback
    dashboard has no gate and no anonymous reader, so the payload is whole."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "minimal"}},
    )
    body = loopback_client.get("/api/status").json()
    assert "profiles" in body
    assert "memory" in body


def test_status_detail_env_override_wins(gated_client, monkeypatch):
    """Hosting platforms inject settings as env, matching the precedence the
    other dashboard keys already follow."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "full"}},
    )
    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_STATUS_DETAIL", "minimal")
    body = gated_client.get("/api/status").json()
    assert _OPERATOR_DETAIL_FIELDS.isdisjoint(body.keys())


def test_status_detail_unknown_value_falls_back_to_full(gated_client, monkeypatch):
    """An unrecognised value must not silently mean ``minimal``: the payload
    Hermes Cloud reads stays whole unless the operator asked otherwise."""
    monkeypatch.setattr(
        web_server, "load_config",
        lambda: {"dashboard": {"public_status_detail": "moderate"}},
    )
    body = gated_client.get("/api/status").json()
    assert "profiles" in body


def test_status_survives_an_unreadable_config(gated_client, monkeypatch):
    """A config read that raises must degrade to the public default rather
    than failing the liveness probe."""
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(web_server, "load_config", _boom)
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    assert "version" in r.json()


def test_unreadable_config_warns_once_not_per_probe(gated_client, monkeypatch, caplog):
    """The warning must not scale with probe traffic. /api/status is polled on a
    timer by portals, uptime monitors and the desktop, so a per-call warning
    turns one broken config into a log flood that buries the line the operator
    needs."""
    import logging

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(web_server, "load_config", _boom)
    monkeypatch.setattr(web_server, "_warned_public_status_detail_unreadable", False)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        for _ in range(3):
            assert gated_client.get("/api/status").status_code == 200
    hits = [r for r in caplog.records if "public_status_detail" in r.message]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"
