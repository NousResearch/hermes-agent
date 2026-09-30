"""Tests for GHSA-ppp5-vxwm-4cf7 — Host-header validation.

DNS rebinding defence: a victim browser that has the dashboard open
could be tricked into fetching from an attacker-controlled hostname
that TTL-flips to 127.0.0.1. Same-origin / CORS checks won't help —
the browser now treats the attacker origin as same-origin. Validating
the Host header at the application layer rejects the attack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


class TestHostHeaderValidator:
    """Unit test the _is_accepted_host helper directly — cheaper and
    more thorough than spinning up the full FastAPI app."""



    def test_zero_zero_bind_accepts_anything(self):
        """0.0.0.0 means operator explicitly opted into all-interfaces
        (requires --insecure). No Host-layer defence is possible — rely
        on operator network controls."""
        from hermes_cli.web_server import _is_accepted_host

        for host in ("10.0.0.5", "evil.example", "my-server.corp.net"):
            assert _is_accepted_host(host, "0.0.0.0")
            assert _is_accepted_host(host + ":9119", "0.0.0.0")

    def test_explicit_non_loopback_bind_requires_exact_match(self):
        """If the operator bound to a specific non-loopback hostname,
        the Host header must match exactly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("my-server.corp.net", "my-server.corp.net")
        assert _is_accepted_host("my-server.corp.net:9119", "my-server.corp.net")
        # Different host — reject
        assert not _is_accepted_host("evil.example", "my-server.corp.net")
        # Loopback — reject (we bound to a specific non-loopback name)
        assert not _is_accepted_host("localhost", "my-server.corp.net")


    def test_trusted_public_host_is_exact_match_only(self):
        """A declared proxy host is accepted without weakening rebinding checks."""
        from hermes_cli.web_server import _is_accepted_host

        trusted = frozenset({"dashboard.example.test"})
        assert _is_accepted_host(
            "dashboard.example.test:9443", "127.0.0.1", trusted
        )
        assert not _is_accepted_host(
            "dashboard.example.test.evil.test", "127.0.0.1", trusted
        )
        assert not _is_accepted_host("evil.test", "127.0.0.1", trusted)

    def test_malformed_host_authorities_fail_closed(self):
        """Ports, IPv6 brackets, and authority syntax must be unambiguous."""
        from hermes_cli.web_server import _is_accepted_host

        trusted = frozenset({"dashboard.example.test"})
        for malformed in (
            "http://dashboard.example.test:9443",
            "dashboard.example.test:",
            "dashboard.example.test:notaport",
            "[::1].evil.test",
            "[::1]:notaport",
            "[localhost]",
        ):
            assert not _is_accepted_host(malformed, "127.0.0.1", trusted)


class TestHostHeaderMiddleware:
    """End-to-end test via the FastAPI app — verify the middleware
    rejects bad Host headers with 400."""

    def test_rebinding_request_rejected(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Simulate start_server having set the bound_host
        app.state.bound_host = "127.0.0.1"
        try:
            client = TestClient(app)
            # The TestClient sends Host: testserver by default — which is
            # NOT a loopback alias, so the middleware must reject it.
            resp = client.get(
                "/api/status",
                headers={"Host": "evil.example"},
            )
            assert resp.status_code == 400
        finally:
            # Clean up so other tests don't inherit the bound_host
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host


    def test_trusted_public_host_request_accepted(self):
        """A loopback backend may accept its declared reverse-proxy host."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        app.state.trusted_public_hosts = frozenset({"dashboard.example.test"})
        try:
            client = TestClient(app)
            resp = client.get(
                "/api/status",
                headers={"Host": "dashboard.example.test:9443"},
            )
            assert resp.status_code != 400
        finally:
            del app.state.bound_host
            del app.state.trusted_public_hosts

    def test_no_bound_host_skips_validation(self):
        """If app.state.bound_host isn't set (e.g. running under test
        infra without calling start_server), middleware must pass through
        rather than crash."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Make sure bound_host isn't set
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host

        client = TestClient(app)
        resp = client.get("/api/status")
        # Should get through to the status endpoint, not a 400
        assert resp.status_code != 400


class TestWebSocketHostOriginGuard:
    """WebSocket upgrades must enforce the same dashboard boundary as HTTP."""

    def test_rebinding_websocket_host_is_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "evil.example",
                    "Origin": "http://evil.example",
                },
            ):
                pass

        assert exc.value.code == 4403


    def test_loopback_websocket_host_and_origin_are_accepted(self, monkeypatch):
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "localhost:9119",
                "Origin": "http://localhost:9119",
            },
        ):
            pass

    def test_trusted_public_websocket_host_and_origin_are_accepted(self, monkeypatch):
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state,
            "trusted_public_hosts",
            frozenset({"dashboard.example.test"}),
            raising=False,
        )
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "dashboard.example.test:9443",
                "Origin": "https://dashboard.example.test:9443",
            },
        ):
            pass

    def test_trusted_public_websocket_rejects_cross_site_origin(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state,
            "trusted_public_hosts",
            frozenset({"dashboard.example.test"}),
            raising=False,
        )
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "dashboard.example.test:9443",
                    "Origin": "https://evil.test",
                },
            ):
                pass

        assert exc.value.code == 4403


_DESKTOP_PORT = 52817  # an ephemeral port, as a packaged Desktop backend binds


@pytest.mark.parametrize("dev_server, bound_port, origin, trusted", [
    # Packaged Desktop backend: file:// renderer, ephemeral port, no dev renderer.
    (None, _DESKTOP_PORT, f"http://127.0.0.1:{_DESKTOP_PORT}", True),  # its own origin
    (None, _DESKTOP_PORT, "http://localhost:5173", False),   # any Vite project's default port
    (None, _DESKTOP_PORT, "http://127.0.0.1:5174", False),   # Vite's next port; the desktop dev port
    (None, _DESKTOP_PORT, "http://127.0.0.1:4174", False),   # the desktop renderer's preview port
    (None, _DESKTOP_PORT, "http://localhost:9119", False),   # the dashboard default, not bound here
    (None, _DESKTOP_PORT, "http://localhost:3000", False),   # the usual local web app port
    (None, _DESKTOP_PORT, "http://localhost:3999", False),   # a page previewed from any other port
    (None, _DESKTOP_PORT, "http://127.0.0.1:5500", False),   # e.g. VS Code Live Server
    # `hermes dashboard` on its default port.
    (None, 9119, "http://localhost:9119", True),
    (None, 9119, "http://127.0.0.1:5173", False),
    # Backend spawned by a desktop dev build (hgui slot 1): that renderer, and only that one.
    ("http://127.0.0.1:5175", _DESKTOP_PORT, "http://127.0.0.1:5175", True),
    ("http://127.0.0.1:5175", _DESKTOP_PORT, "http://127.0.0.1:5174", False),  # another renderer
    ("http://127.0.0.1:5175", _DESKTOP_PORT, "http://127.0.0.1:5176", False),
])
def test_other_local_ports_can_neither_read_nor_open_a_socket(
        monkeypatch, dev_server, bound_port, origin, trusted):
    """A loopback page on another port is a different origin. It must get no CORS grant (else it
    reads index.html's session token) and no WebSocket upgrade (else it drives /api/pty with that
    token). Only the bound port and the desktop renderer this backend was spawned for (Electron
    passes HERMES_DESKTOP_DEV_SERVER down; worktree-ui-dev.md runs slot N's Vite on 5174+N) are
    trusted — no fixed dev port, so a Vite app on 5173/5174 cannot reach a packaged backend."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import hermes_cli.web_server as ws

    if dev_server:
        monkeypatch.setenv("HERMES_DESKTOP_DEV_SERVER", dev_server)
    else:
        monkeypatch.delenv("HERMES_DESKTOP_DEV_SERVER", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", bound_port, raising=False)
    monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

    client = TestClient(ws.app)

    host = f"127.0.0.1:{bound_port}"
    grant = client.get("/api/status", headers={"Host": host, "Origin": origin})
    assert (grant.headers.get("access-control-allow-origin") == origin) is trusted

    url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
    headers = {"Host": host, "Origin": origin}
    if trusted:
        with client.websocket_connect(url, headers=headers):
            pass
    else:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url, headers=headers):
                pass
        assert exc.value.code == 4403


@pytest.mark.parametrize("origin, trusted", [
    ("http://127.0.0.1:9119", True),    # the dev page's own upgrade, as the proxy presents it
    ("http://localhost:5173", False),   # the same upgrade without the rewrite
    ("http://localhost:3000", False),   # another local page using the proxy: passed through as is
])
def test_dashboard_vite_dev_proxy_needs_no_trusted_dev_port(monkeypatch, origin, trusted):
    """``web/`` `npm run dev` serves the SPA from Vite and proxies /api (ws: true) without changing
    Host. The browser's Origin is Vite's, so web/vite.config.ts rewrites it to the backend's own
    origin for upgrades from Vite's own page (Origin host == Host) and leaves every other page's
    Origin alone. The backend then admits the dev page as same-origin with no :5173 grant."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import hermes_cli.web_server as ws

    monkeypatch.delenv("HERMES_DESKTOP_DEV_SERVER", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    client = TestClient(ws.app)

    url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
    headers = {"Host": "localhost:5173", "Origin": origin}
    if trusted:
        with client.websocket_connect(url, headers=headers):
            pass
    else:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url, headers=headers):
                pass
        assert exc.value.code == 4403


@pytest.mark.parametrize("host, origin, allowed", [
    ("127.0.0.1:9119", "http://localhost:3999", False),   # a page on another local port
    ("127.0.0.1:9119", "https://evil.example", False),    # a remote site
    ("evil.example", "http://evil.example", False),       # DNS rebinding
    ("127.0.0.1:9119", "http://127.0.0.1:9119", True),    # the dashboard itself
    ("127.0.0.1:9119", None, True),                       # non-browser client
])
def test_plugin_websockets_run_the_core_host_origin_gate(monkeypatch, host, origin, allowed):
    """HTTP middleware never sees a WebSocket upgrade, so plugin sockets need the gate the core
    sockets run before accept. The kanban plugin's /events only checked the credential: a token
    read by a page on another local port still opened it. The gate is attached when plugin routers
    are mounted, so the real kanban socket is covered without the plugin opting in."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import hermes_cli.web_server as ws

    assert "/api/plugins/kanban/events" in {getattr(r, "path", "") for r in ws.app.routes}
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
    client = TestClient(ws.app)

    url = f"/api/plugins/kanban/events?token={ws._SESSION_TOKEN}"
    headers = {"Host": host, **({"Origin": origin} if origin else {})}
    if allowed:
        with client.websocket_connect(url, headers=headers):
            pass
    else:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url, headers=headers):
                pass
        assert exc.value.code == 4403


def test_every_mounted_plugin_websocket_gets_the_gate(tmp_path, monkeypatch):
    """The gate lives on the mount, not in each plugin: a plugin that declares a socket and checks
    nothing itself is still refused for a foreign origin and served for the dashboard's own."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import hermes_cli.web_server as ws
    import hermes_cli.web_server_dashboard as dashboard

    api_dir = tmp_path / "wsprobe" / "dashboard"
    api_dir.mkdir(parents=True)
    (api_dir / "api.py").write_text(
        "from fastapi import APIRouter, WebSocket\n"
        "router = APIRouter()\n"
        "@router.websocket('/socket')\n"
        "async def socket(ws: WebSocket):\n"
        "    await ws.accept()\n"
        "    await ws.send_text('open')\n"
        "    await ws.close()\n"
    )
    fresh = FastAPI()
    fresh.state.bound_host, fresh.state.bound_port, fresh.state.auth_required = "127.0.0.1", 9119, False
    monkeypatch.setattr(ws, "app", fresh)
    monkeypatch.setattr(ws, "_dashboard_plugins_cache", [{
        "name": "wsprobe", "source": "bundled", "_dir": str(api_dir), "_api_file": "api.py"}])
    monkeypatch.delitem(sys.modules, "hermes_dashboard_plugin_wsprobe", raising=False)
    dashboard._mount_plugin_api_routes()
    client = TestClient(fresh)

    with client.websocket_connect(
            "/api/plugins/wsprobe/socket",
            headers={"Host": "127.0.0.1:9119", "Origin": "http://127.0.0.1:9119"}) as sock:
        assert sock.receive_text() == "open"
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
                "/api/plugins/wsprobe/socket",
                headers={"Host": "127.0.0.1:9119", "Origin": "http://localhost:3999"}):
            pass
    assert exc.value.code == 4403


@pytest.mark.parametrize("path", [
    "/",                    # index.html / the headless page: carries the session token
    "/api/files/download",  # the ?token= route: serves files under $HOME
    "/api/status",          # hermes_home / config_path / env_path
    "/openapi.json",        # the full route map
    "/docs",
    "/redoc",
])
def test_token_page_and_route_map_are_not_cors_readable_from_another_local_port(monkeypatch, path):
    """Every unauthenticated or query-token surface a page on another local port could use to take
    the token, read files or fingerprint the backend gets no CORS grant; the dashboard's own origin
    still does, so the check is not vacuous."""
    from fastapi.testclient import TestClient

    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
    client = TestClient(ws.app)
    url = f"{path}?token={ws._SESSION_TOKEN}&path=missing.txt" if path == "/api/files/download" else path

    for origin, granted in (("http://localhost:3999", False), ("http://127.0.0.1:9119", True)):
        r = client.get(url, headers={"Host": "127.0.0.1:9119", "Origin": origin})
        assert (r.headers.get("access-control-allow-origin") == origin) is granted, (origin, r.status_code)
