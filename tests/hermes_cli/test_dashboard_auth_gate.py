"""Regression harness for the dashboard auth gate.

Phase 0 — establish a baseline pin on the current (pre-OAuth) behavior so
later phases can prove they didn't break loopback mode.
"""
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client_loopback():
    # Pin the bound-host state for host_header_middleware so requests with
    # default Host: testclient pass the DNS-rebinding check.  TestClient
    # sends Host: testserver by default, but our middleware accepts the
    # loopback aliases when bound_host is loopback.
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port


def test_loopback_status_is_public(client_loopback):
    """`/api/status` must remain reachable without a token in loopback mode."""
    r = client_loopback.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body


def test_loopback_protected_route_requires_token(client_loopback):
    """Any non-public /api/ route must require the session token."""
    # /api/sessions exists and is auth-gated by auth_middleware.
    r = client_loopback.get("/api/sessions")
    assert r.status_code == 401


def test_loopback_protected_route_accepts_session_token(client_loopback):
    """The injected SPA token unlocks protected /api/ routes."""
    r = client_loopback.get(
        "/api/sessions",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    # 200 or 404 (no sessions yet) both prove the auth layer let it through.
    # 500 is also acceptable if there's a downstream issue unrelated to auth.
    assert r.status_code != 401, (
        f"Expected auth to succeed but got 401; body: {r.text}"
    )


def test_loopback_index_injects_session_token(client_loopback):
    """Loopback mode keeps injecting the SPA token into index.html.

    This is the property that the new auth gate MUST disable once a gated
    bind is detected. Phase 3 will add an inverse test for the gated path.
    """
    r = client_loopback.get("/")
    if r.status_code == 404:
        pytest.skip("WEB_DIST not built in this env")
    assert "__HERMES_SESSION_TOKEN__" in r.text


def test_loopback_host_header_validation_still_enforced(client_loopback):
    """DNS-rebinding protection: a foreign Host header is rejected."""
    r = client_loopback.get("/api/status", headers={"Host": "evil.test"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# should_require_auth predicate (Task 0.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host,allow_public,expected", [
    ("127.0.0.1", False, False),
    ("127.0.0.1", True,  False),
    ("localhost", False, False),
    ("::1",       False, False),
    ("0.0.0.0",   True,  False),    # --insecure escape hatch
    ("0.0.0.0",   False, True),
    ("192.168.1.5", False, True),
    ("10.0.0.1",  True,  False),
    ("100.64.0.1", False, True),    # Tailscale CGNAT — treated as public
    ("hermes-agent-prod-abc.fly.dev", False, True),
])
def test_should_require_auth_truth_table(host, allow_public, expected):
    from hermes_cli.web_server import should_require_auth
    assert should_require_auth(host, allow_public) is expected
