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


def test_status_reports_auth_disabled_in_loopback_mode(loopback_client):
    r = loopback_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is False
    # Loopback mode has no registered providers (the Nous plugin's env
    # vars aren't set in test).
    assert body["auth_providers"] == []


def test_status_preserves_existing_fields(loopback_client):
    """Defence-in-depth: adding auth_required/auth_providers must not
    have dropped any previous field (the dashboard's React StatusPage
    relies on the full payload shape)."""
    r = loopback_client.get("/api/status")
    body = r.json()
    expected_keys = {
        "version", "hermes_home", "config_path", "env_path",
        "config_version", "latest_config_version", "gateway_running",
        "gateway_pid", "gateway_health_url", "gateway_state",
        "gateway_platforms", "gateway_exit_reason", "gateway_updated_at",
        "active_sessions", "auth_required", "auth_providers",
    }
    missing = expected_keys - set(body.keys())
    assert not missing, f"/api/status dropped fields: {missing}"
    # gateway_updated_at is a typed contract (web/src/lib/api.ts declares
    # string | null): it must never be a number, and any string must
    # round-trip through fromisoformat as a timezone-aware timestamp.
    updated_at = body["gateway_updated_at"]
    assert isinstance(updated_at, (str, type(None))), (
        f"gateway_updated_at must be str|None, got {type(updated_at).__name__}: {updated_at!r}"
    )
    if updated_at is not None:
        from datetime import datetime
        parsed = datetime.fromisoformat(updated_at)
        assert parsed.tzinfo is not None
        assert parsed.isoformat() == updated_at, "gateway_updated_at is not canonical isoformat"


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


def test_status_includes_host_detail_in_loopback_mode(loopback_client):
    """Counterpart to the gated case: a loopback bind is local-only, so the
    full payload (including host paths and PID) is still served — preserving
    the StatusPage / ``hermes status`` experience for local operators."""
    r = loopback_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    missing = _HOST_DETAIL_FIELDS - set(body.keys())
    assert not missing, f"loopback /api/status should keep host detail: {missing}"
