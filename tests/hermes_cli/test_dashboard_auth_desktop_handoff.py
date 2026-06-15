"""End-to-end tests for the desktop system-browser sign-in handoff.

Covers the gateway half of the fix for issue #42448: ``/auth/login`` carrying
the loopback ``app_redirect``/``app_state`` across the IdP round trip,
``/auth/callback`` minting a one-time handoff code (and NOT setting browser
cookies) when the request is app-driven, and ``/api/auth/desktop-exchange``
trading that code for the session cookies. Also pins the loopback-only
validation that keeps the handoff code from leaking to an attacker host.

The gated_app harness mirrors ``test_dashboard_auth_middleware.py``: register
the stub provider, flip ``app.state.auth_required = True``, drive a
``TestClient``.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

# Shares the dashboard-auth app-state xdist group: these tests mutate
# ``web_server.app.state.auth_required`` at module level.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.app_handoff import _reset_for_tests
from hermes_cli.dashboard_auth.routes import _validate_app_redirect
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

_LOOPBACK = "http://127.0.0.1:54321/callback"
_NONCE = "app-nonce-abc123"


@pytest.fixture
def gated_app():
    clear_providers()
    register_provider(StubAuthProvider())
    _reset_for_tests()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    _reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _drive_to_handoff(client: TestClient) -> str:
    """Walk /auth/login → /auth/callback with the app params, returning the
    one-time handoff code the callback bounced to the loopback URL."""
    r1 = client.get(
        f"/auth/login?provider=stub&app_redirect={_LOOPBACK}&app_state={_NONCE}",
        follow_redirects=False,
    )
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]

    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302
    # The callback must NOT set browser session cookies on the app path —
    # the session belongs in the app, reached via the handoff code only.
    set_cookies = r2.headers.get_list("set-cookie")
    assert not any("hermes_session_at" in c for c in set_cookies)
    assert not any("hermes_session_rt" in c for c in set_cookies)

    loc = urlparse(r2.headers["location"])
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == _LOOPBACK
    q = parse_qs(loc.query)
    assert q["state"] == [_NONCE]
    return q["code"][0]


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


def test_providers_endpoint_advertises_handoff(gated_app):
    r = gated_app.get("/api/auth/providers")
    assert r.status_code == 200
    assert r.json().get("app_handoff") is True


# ---------------------------------------------------------------------------
# Full handoff round trip
# ---------------------------------------------------------------------------


def test_handoff_round_trip_establishes_session(gated_app):
    code = _drive_to_handoff(gated_app)

    # The app trades the code for cookies. No prior session cookie exists,
    # proving /api/auth/desktop-exchange is allowlisted past the gate.
    r = gated_app.post("/api/auth/desktop-exchange", json={"code": code, "state": _NONCE})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True}
    set_cookies = r.headers.get_list("set-cookie")
    assert any("hermes_session_at" in c for c in set_cookies)
    assert any("hermes_session_rt" in c for c in set_cookies)

    # The client now holds the cookies → a gated API route succeeds.
    r2 = gated_app.get("/api/sessions")
    assert r2.status_code == 200, r2.text


def test_handoff_code_is_single_use(gated_app):
    code = _drive_to_handoff(gated_app)
    first = gated_app.post("/api/auth/desktop-exchange", json={"code": code})
    assert first.status_code == 200
    # A replay (or a stolen code re-presented) must be rejected.
    gated_app.cookies.clear()
    second = gated_app.post("/api/auth/desktop-exchange", json={"code": code})
    assert second.status_code == 400


def test_exchange_rejects_unknown_code(gated_app):
    r = gated_app.post("/api/auth/desktop-exchange", json={"code": "nope"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Loopback-only guard: a non-loopback app_redirect must NOT trigger handoff
# ---------------------------------------------------------------------------


def test_non_loopback_app_redirect_falls_back_to_browser_login(gated_app):
    evil = "https://evil.example/grab"
    r1 = gated_app.get(
        f"/auth/login?provider=stub&app_redirect={evil}&app_state={_NONCE}",
        follow_redirects=False,
    )
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]

    r2 = gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    # No handoff: this is an ordinary browser login — cookies set, lands on /.
    assert r2.status_code == 302
    assert r2.headers["location"] == "/"
    set_cookies = r2.headers.get_list("set-cookie")
    assert any("hermes_session_at" in c for c in set_cookies)


@pytest.mark.parametrize(
    "raw,expected_ok",
    [
        ("http://127.0.0.1:54321/callback", True),
        ("http://localhost:8080/callback", True),
        ("http://[::1]:9000/cb", True),
        ("https://127.0.0.1:54321/callback", False),  # https never accepted
        ("http://evil.example:80/callback", False),   # non-loopback host
        ("http://127.0.0.1/callback", False),         # no explicit port
        ("http://user:pw@127.0.0.1:5000/cb", False),  # credentials in authority
        ("ftp://127.0.0.1:21/cb", False),             # wrong scheme
        ("", False),
    ],
)
def test_validate_app_redirect(raw, expected_ok):
    assert bool(_validate_app_redirect(raw)) is expected_ok
