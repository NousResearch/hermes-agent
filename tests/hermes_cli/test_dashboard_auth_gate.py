"""Regression harness for the dashboard auth gate.

Phase 0 — establish a baseline pin on the current (pre-OAuth) behavior so
later phases can prove they didn't break loopback mode.
"""
import pytest

# Phase 5 / Phase 6: these tests mutate ``web_server.app.state.auth_required``
# at module level. Run them in the same xdist worker so they don't race
# against each other (and against any other file that also touches
# ``app.state``) — the marker name is shared across all dashboard-auth test
# files that gate the app.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")
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
    # --insecure (allow_public=True) NO LONGER bypasses the gate on a public
    # bind (June 2026 hermes-0day hardening). Non-loopback always requires auth.
    ("0.0.0.0",   True,  True),
    ("0.0.0.0",   False, True),
    ("192.168.1.5", False, True),
    ("10.0.0.1",  True,  True),     # allow_public ignored — LAN IP is public
    ("100.64.0.1", False, True),    # Tailscale CGNAT — treated as public
    ("hermes-agent-prod-abc.fly.dev", False, True),
])
def test_should_require_auth_truth_table(host, allow_public, expected):
    from hermes_cli.web_server import should_require_auth
    assert should_require_auth(host, allow_public) is expected


# ---------------------------------------------------------------------------
# start_server stashes auth_required on app.state (Task 0.3)
# ---------------------------------------------------------------------------


def _stub_uvicorn_run(monkeypatch):
    """Replace uvicorn.Config/Server with no-op fakes so start_server
    returns immediately (rather than blocking on the event loop). Returns the dict
    that will capture the keyword args.
    """
    import asyncio
    import contextlib
    import uvicorn
    captured: dict = {"kwargs": {}}

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000

        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def load(self):
            pass

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *a, **kw):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", lambda config: _FakeServer())
    return captured


def test_start_server_loopback_sets_auth_required_false(monkeypatch):
    """Loopback bind: app.state.auth_required is False after start_server."""
    _stub_uvicorn_run(monkeypatch)
    # Force a fresh state to detect that start_server actually set it.
    web_server.app.state.auth_required = None
    web_server.start_server(
        host="127.0.0.1", port=9119,
        open_browser=False, allow_public=False,
    )
    assert web_server.app.state.auth_required is False


def test_start_server_insecure_public_no_longer_bypasses_gate(monkeypatch):
    """``--insecure`` (allow_public=True) on a public host: gate now ENGAGES.

    June 2026 hardening: --insecure no longer disables auth. With no providers
    registered, the bind fails closed (SystemExit) and auth_required is True.
    """
    from hermes_cli.dashboard_auth import clear_providers
    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=True,
        )
    assert web_server.app.state.auth_required is True


def test_start_server_public_without_insecure_records_auth_required(monkeypatch):
    """Public bind without --insecure: the gate engages and auth_required=True.

    With no providers registered, this fails closed with SystemExit. The
    flag-stashing happens BEFORE the exit so the rest of the system can
    branch on it. (See task 3.5 tests below for the with-provider path.)
    """
    from hermes_cli.dashboard_auth import clear_providers
    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )
    assert web_server.app.state.auth_required is True


# ---------------------------------------------------------------------------
# Task 3.5: start_server fail-closed + proxy_headers + index-token suppression
# ---------------------------------------------------------------------------


def test_start_server_gate_with_provider_proceeds_and_sets_proxy_headers(monkeypatch):
    """With at least one provider, public bind + no --insecure starts the server.

    The SystemExit-refusing-to-bind guard is REPLACED in gated mode by
    "the gate engages", so as long as a provider is registered the bind
    succeeds.  uvicorn is called with proxy_headers=True so X-Forwarded-Proto
    from Fly's TLS terminator is honoured for cookie Secure-flag decisions.
    """
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    clear_providers()
    register_provider(StubAuthProvider())
    captured = _stub_uvicorn_run(monkeypatch)
    try:
        web_server.app.state.auth_required = None
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )
        assert web_server.app.state.auth_required is True
        assert captured["kwargs"].get("host") == "0.0.0.0"
        assert captured["kwargs"].get("proxy_headers") is True
    finally:
        clear_providers()


def test_start_server_gate_without_provider_fails_closed(monkeypatch):
    """No providers + gate would activate → SystemExit with a clear message."""
    from hermes_cli.dashboard_auth import clear_providers

    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit, match=r"no auth providers"):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )


def test_start_server_surfaces_nous_skip_reason_when_unconfigured(monkeypatch):
    """When the bundled Nous plugin loaded but skipped registration (no
    env vars set), the gate's fail-closed message should surface the
    plugin's LAST_SKIP_REASON so the operator knows the config fix is
    'set HERMES_DASHBOARD_OAUTH_CLIENT_ID', not 'install a plugin'."""
    from hermes_cli.dashboard_auth import clear_providers
    from plugins.dashboard_auth import nous as nous_plugin

    # Simulate the plugin running and skipping for "no client_id".
    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
    from unittest.mock import MagicMock
    nous_plugin.register(MagicMock())  # populates LAST_SKIP_REASON
    assert "HERMES_DASHBOARD_OAUTH_CLIENT_ID" in nous_plugin.LAST_SKIP_REASON

    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit) as exc_info:
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )
    # The error message embeds the plugin's specific skip reason rather
    # than the generic "Install the default Nous provider" boilerplate.
    msg = str(exc_info.value)
    assert "HERMES_DASHBOARD_OAUTH_CLIENT_ID" in msg
    assert "nous:" in msg


def test_start_server_loopback_keeps_proxy_headers_off(monkeypatch):
    """Loopback bind: proxy_headers stays False (no TLS terminator in front)."""
    captured = _stub_uvicorn_run(monkeypatch)
    web_server.start_server(
        host="127.0.0.1", port=9119,
        open_browser=False, allow_public=False,
    )
    assert captured["kwargs"].get("proxy_headers") is False


def test_start_server_insecure_public_engages_gate_and_fails_closed(monkeypatch):
    """--insecure on a public host: gate engages now; no provider → fail closed.

    Replaces the old "insecure keeps gate off" test. --insecure is a no-op for
    auth as of the June 2026 hardening, so a public bind with no provider
    refuses to start.
    """
    from hermes_cli.dashboard_auth import clear_providers

    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=True,
        )
    assert web_server.app.state.auth_required is True
