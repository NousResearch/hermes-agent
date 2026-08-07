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


# ---------------------------------------------------------------------------
# extra_hosts + headless serve: Host acceptance and auth scoping stay atomic
# (#75907 review — teknium1: thread headless into the interactive preflight;
# egilewski: a headless serve must not admit proxy Hosts while the gate is
# off, or /api/status leaks local-only fields to remote callers)
# ---------------------------------------------------------------------------


@pytest.fixture
def extra_hosts(monkeypatch):
    """Pin the operator extra-hosts allowlist (bypasses env+config+cache)."""
    def _set(hosts):
        monkeypatch.setattr(
            web_server, "_extra_allowed_hosts_cache", frozenset(hosts)
        )
    return _set


def test_extra_hosts_engage_gate_for_interactive_dashboard(extra_hosts):
    """The PR's feature: declared proxy hosts re-engage the gate on loopback."""
    from hermes_cli.web_server import should_require_auth
    extra_hosts({"proxy.example"})
    assert should_require_auth("127.0.0.1", headless=False) is True


def test_headless_serve_skips_gate_even_with_extra_hosts(extra_hosts):
    """Headless serve keeps its loopback posture — and (below) rejects the
    proxy Host at the boundary instead of admitting it unauthenticated."""
    from hermes_cli.web_server import should_require_auth
    extra_hosts({"proxy.example"})
    assert should_require_auth("127.0.0.1", headless=True) is False


def test_non_loopback_headless_bind_still_engages_gate(extra_hosts):
    from hermes_cli.web_server import should_require_auth
    extra_hosts({"proxy.example"})
    assert should_require_auth("0.0.0.0", headless=True) is True


def test_is_accepted_host_rejects_extra_hosts_when_headless(extra_hosts):
    from hermes_cli.web_server import _is_accepted_host
    extra_hosts({"proxy.example"})
    # Interactive (gated) dashboard: the feature works.
    assert _is_accepted_host("proxy.example:443", "127.0.0.1", headless=False) is True
    # Headless serve: the same Host is refused — no unauthenticated exposure.
    assert _is_accepted_host("proxy.example:443", "127.0.0.1", headless=True) is False
    # Loopback aliases are unaffected by the headless flag.
    assert _is_accepted_host("127.0.0.1:9119", "127.0.0.1", headless=True) is True
    # Unlisted names stay rejected either way (rebinding floor preserved).
    assert _is_accepted_host("evil.example", "127.0.0.1", headless=False) is False


@pytest.fixture
def client_headless_loopback():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_headless = getattr(web_server.app.state, "headless", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.headless = True
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.headless = prev_headless


def test_headless_serve_rejects_proxy_host_at_http_boundary(
    client_headless_loopback, extra_hosts
):
    """egilewski's split-invariant repro: a proxy-routed request must NOT
    reach /api/status on a headless serve. The 400 body carries none of the
    local-only deployment fields."""
    extra_hosts({"proxy.example"})
    r = client_headless_loopback.get(
        "/api/status", headers={"host": "proxy.example:443"}
    )
    assert r.status_code == 400
    for field in (
        "hermes_home", "config_path", "env_path",
        "gateway_pid", "gateway_health_url",
    ):
        assert field not in r.text


def test_interactive_dashboard_accepts_proxy_host_at_http_boundary(
    client_loopback, extra_hosts, monkeypatch
):
    """The feature keeps working where it belongs: the gated interactive
    dashboard accepts the configured proxy Host (400 is the Host rejection;
    any other status means the request passed the boundary)."""
    extra_hosts({"proxy.example"})
    monkeypatch.setattr(web_server.app.state, "headless", False, raising=False)
    r = client_loopback.get("/api/status", headers={"host": "proxy.example:443"})
    assert r.status_code != 400


class _FakeWS:
    def __init__(self, headers):
        self.headers = headers


def test_ws_host_origin_reason_rejects_proxy_host_when_headless(
    extra_hosts, monkeypatch
):
    from hermes_cli.web_server import _ws_host_origin_reason
    extra_hosts({"proxy.example"})
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "headless", True, raising=False)
    reason = _ws_host_origin_reason(_FakeWS({"host": "proxy.example:443"}))
    assert reason is not None and reason.startswith("host_mismatch")


def test_ws_host_origin_reason_accepts_proxy_host_when_interactive(
    extra_hosts, monkeypatch
):
    from hermes_cli.web_server import _ws_host_origin_reason
    extra_hosts({"proxy.example"})
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "headless", False, raising=False)
    assert _ws_host_origin_reason(_FakeWS({"host": "proxy.example:443"})) is None


def test_preflight_threads_headless_into_should_require_auth(monkeypatch):
    """teknium1: the interactive preflight must evaluate the gate with the
    same headless value start_server uses, so `hermes serve` doesn't enter
    auth setup for a gate that will be off, and `hermes dashboard` does."""
    import types
    from hermes_cli import main as main_mod

    calls = []

    def _spy(host, allow_public=False, headless=False):
        calls.append({"host": host, "headless": headless})
        return False  # gate off → preflight no-ops before any prompt

    monkeypatch.setattr(
        "hermes_cli.web_server.should_require_auth", _spy
    )

    main_mod._maybe_setup_dashboard_auth_interactively(
        types.SimpleNamespace(host="127.0.0.1", headless_backend=True)
    )
    assert calls == [{"host": "127.0.0.1", "headless": True}]

    calls.clear()
    main_mod._maybe_setup_dashboard_auth_interactively(
        types.SimpleNamespace(host="127.0.0.1", headless_backend=False)
    )
    assert calls == [{"host": "127.0.0.1", "headless": False}]


